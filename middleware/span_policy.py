"""Span-selection policy: chooses which messages become reusable PIC spans.

Configured by ``processing.span_policy``, consumed by
``processors.build_pic_prompt``. Register new policies in :data:`_POLICIES`.
"""
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Type

logger = logging.getLogger(__name__)


class SpanPolicy(ABC):
    """Maps a request's messages to the indices that become PIC spans."""

    @abstractmethod
    def select_spans(self, messages: List[dict]) -> Set[int]: ...


class AllMessagesSpanPolicy(SpanPolicy):
    """Every message is its own span (default).

    Note this spans the system prompt and the final user turn too, which makes
    the live query attention-isolated from the context it is asking about. Use
    :class:`ContextMessagesSpanPolicy` for retrieval/agent shapes where the
    context recurs but the query must still read it.
    """

    def select_spans(self, messages: List[dict]) -> Set[int]:
        return set(range(len(messages)))


class ContextMessagesSpanPolicy(SpanPolicy):
    """Span = a context-carrying message, never the system prompt or the live query.

    For retrieval and web-agent shapes the reusable content arrives as ordinary
    user/tool messages (a retrieved passage, a serialized page) rather than as a
    bash tool result, so :class:`ToolFileReadSpanPolicy` selects nothing. Spanning
    *everything* instead is worse than useless: a span attends only to itself, so
    making the final user turn a span leaves the question unable to read the
    passages, and the model answers from nothing.

    Keep the system prompt and the last message in-context; everything between
    them is reusable content.
    """

    def select_spans(self, messages: List[dict]) -> Set[int]:
        spans = {
            i for i, m in enumerate(messages[:-1])
            if i > 0 and m.get("role") in ("user", "tool")
        }
        logger.debug("context_messages: %d/%d messages selected as spans",
                     len(spans), len(messages))
        return spans


class PastContextSpanPolicy(SpanPolicy):
    """Span = context that is NOT the message the model is currently reasoning over.

    A sliding-window web agent sends ``[system, obs_t-k .. obs_t-1, obs_t, task]``. The
    reusable content is the RETAINED observations: they recur verbatim at a shifted offset
    every step. ``obs_t`` never recurs - it is new by construction - so spanning it buys no
    reuse at all, while costing exactly what a span costs: it attends only to itself, and it
    is the page the agent must read to choose its next action.

    So keep the system prompt, the current observation and the task in-context, and span only
    what came before. Same reasoning as ContextMessagesSpanPolicy, applied one message
    further left because in this shape the live query spans TWO trailing messages.
    """

    def select_spans(self, messages: List[dict]) -> Set[int]:
        spans = {
            i for i, m in enumerate(messages[:-2])
            if i > 0 and m.get("role") in ("user", "tool")
        }
        logger.debug("past_context: %d/%d messages selected as spans",
                     len(spans), len(messages))
        return spans


class ToolFileReadSpanPolicy(SpanPolicy):
    """Span = a tool response produced by a deterministic code-file read.

    Two read seams qualify:

    * ``bash`` -- the command must pair a read verb (cat/head/tail/grep/rg/sed/
      awk/less/more) with a code-file extension, and match none of the
      :attr:`_REJECTERS` (writes, ``sed -i``, interpreter at command position,
      ``grep -l``). Pure regex; no shell parsing.
    * ``view`` -- the structured file-editor read tool. EVERY view response is
      accepted, whatever the path: the tool is read-only by construction, so the
      code-extension gate that `bash` needs (to separate reads from writes) only
      discarded reusable bytes here. Ranged views (``view_range``) count
      too, matching the existing acceptance of ``sed -n '1,40p' main.go``.

    ``view`` was added to the agent after this policy was written, and a
    bash-only gate silently excluded it -- the agent's main file-read path,
    carrying most of the reusable bytes -- from ever becoming a span.
    """

    _CODE_EXTS: Tuple[str, ...] = (
        "py", "js", "ts", "tsx", "jsx", "java", "go", "rs", "c", "cpp", "h",
        "hpp", "rb", "php", "cs", "kt", "swift", "scala", "sh", "sql",
        "json", "yaml", "yml", "toml", "md", "txt", "csv",
    )
    _EXTS_ALT = "|".join(_CODE_EXTS)

    # Positive gates: a read verb AND a code-file extension must both appear.
    _READ_RE = re.compile(r"\b(cat|head|tail|grep|rg|sed|awk|less|more)\b")
    _EXT_RE = re.compile(rf"\.({_EXTS_ALT})\b", re.IGNORECASE)

    # Rejecters: each is a precision filter on top of the positive gates.
    _WRITE_RE = re.compile(
        # `> file.py` / `>> file.py` redirects targeting a code file.
        rf">>?\s*\S+\.({_EXTS_ALT})\b"
        # Heredoc introducer: `<<` (optional `-`) followed by an identifier,
        # anchored to a redirection position so literal `<<` inside quoted
        # grep patterns (e.g. `grep '<<' file.py`) does not match.
        rf"|(?:^|[\s;&|])<<-?\s*['\"]?\w"
        # `tee` writing to a tracked file, at command position. Excludes
        # `| tee build.log` (untracked target) and path segments named `tee`.
        rf"|(?:^|[\s;&|(])tee\s+(?:-[aA]\s+)?\S+\.({_EXTS_ALT})\b",
        re.IGNORECASE,
    )
    _SED_INPLACE_RE = re.compile(r"\bsed\b\s+-[a-zA-Z]*i|\bsed\b[^|]*--in-place")
    # Interpreter/runner names must be at command position (start of a
    # pipeline/sequence segment), so they do not match when appearing as
    # search terms or as substrings of file paths.
    _EXEC_RE = re.compile(
        r"(?:^|[;&|(]\s*)"
        r"(?:python3?|pytest|runtests|manage\.py|django-admin"
        r"|unittest|tox|nox|node)\b"
    )
    # `grep -l` (or any flag cluster containing `l`) and `--files-with-matches`
    # both produce filename-list output rather than stable file content.
    _GREP_LIST_RE = re.compile(
        r"\bgrep\b[^|]*\s-[A-Za-z]*l[A-Za-z]*\b|--files-with-matches"
    )

    _REJECTERS: Tuple[re.Pattern, ...] = (
        _WRITE_RE, _SED_INPLACE_RE, _EXEC_RE, _GREP_LIST_RE,
    )

    def select_spans(self, messages: List[dict]) -> Set[int]:
        # Map each tool_call id -> a description of the deterministic read it made.
        # Two seams qualify: `bash` read commands, and the structured `view` tool.
        # Gating on bash alone missed `view` entirely, which is the agent's main
        # file-read path and carries most of the reusable bytes.
        read_by_id: Dict[str, str] = {}
        for m in messages:
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                tid = tc.get("id")
                if not tid:
                    continue
                name = fn.get("name")
                if self._is_bash(name):
                    cmd = self._command(fn.get("arguments"))
                    if cmd and self._is_code_read(cmd):
                        read_by_id[tid] = "code-file read: " + cmd
                elif self._is_view(name):
                    # EVERY view response is a span. The extension gate exists to
                    # tell reads from writes in `bash`; `view` is read-only by
                    # construction, so filtering it only dropped reusable bytes --
                    # extensionless files (Makefile, Dockerfile), unlisted types
                    # (.rst, .xml, .html, .cfg, .lock) and directory listings all
                    # fell through despite being deterministic reads.
                    path = self._arg(fn.get("arguments"), "path")
                    read_by_id[tid] = "view: " + (path or "<no path>")

        spans: Set[int] = set()
        for i, m in enumerate(messages):
            if m.get("role") != "tool":
                continue
            src = read_by_id.get(m.get("tool_call_id"))
            if src:
                spans.add(i)
                logger.info(
                    "tool_file_read: caught msg[%d] as PIC span — %s", i, src[:120],
                )
        if spans and logger.isEnabledFor(logging.INFO):
            logger.info(
                "tool_file_read: selected %d/%d messages as PIC spans: %s",
                len(spans), len(messages), sorted(spans),
            )
        return spans

    @staticmethod
    def _is_bash(name: object) -> bool:
        # Accept "bash" and namespaced variants (e.g. mcp__environment__bash).
        return isinstance(name, str) and name.endswith("bash")

    @staticmethod
    def _is_view(name: object) -> bool:
        # The structured file-editor read tool. Exact match plus namespaced
        # variants only -- endswith("view") alone would also catch "preview".
        return isinstance(name, str) and (name == "view" or name.endswith("__view"))

    @staticmethod
    def _arg(arguments: object, key: str) -> Optional[str]:
        """One string argument off a tool call, whose args may be a JSON string."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (ValueError, TypeError):
                return None
        if isinstance(arguments, dict):
            v = arguments.get(key)
            return v if isinstance(v, str) else None
        return None

    @classmethod
    def _command(cls, arguments: object) -> Optional[str]:
        return cls._arg(arguments, "command")

    @classmethod
    def _is_code_read(cls, command: str) -> bool:
        if not (cls._READ_RE.search(command) and cls._EXT_RE.search(command)):
            return False
        return not any(r.search(command) for r in cls._REJECTERS)


class NoSpansPolicy(SpanPolicy):
    """No spans: the ``full`` full-recompute baseline. Avoids the huge-GAP_LENGTH path, which livelocks the scheduler."""

    def select_spans(self, messages: List[dict]) -> Set[int]:
        return set()


class AllToolResponsesSpanPolicy(SpanPolicy):
    """Every tool-response message (``role == "tool"``) is a reusable span,
    regardless of which tool produced it or what it contains.

    Tool outputs (command results, fetched docs, retrieved snippets) are stable
    text that tends to recur verbatim across requests, so caching each as a
    prefix-independent span is broadly useful. Coarser than
    :class:`ToolFileReadSpanPolicy` (no command inspection) — it just spans them all.
    """

    def select_spans(self, messages: List[dict]) -> Set[int]:
        return {i for i, m in enumerate(messages) if m.get("role") == "tool"}


# Available policies by config name. Add new policies here.
_POLICIES: Dict[str, Type[SpanPolicy]] = {
    "all_messages": AllMessagesSpanPolicy,
    "context_messages": ContextMessagesSpanPolicy,
    "past_context": PastContextSpanPolicy,
    "tool_file_read": ToolFileReadSpanPolicy,
    "tool_responses": AllToolResponsesSpanPolicy,
    "none": NoSpansPolicy,
}


def get_span_policy(name: str) -> SpanPolicy:
    try:
        return _POLICIES[name]()
    except KeyError:
        raise ValueError(
            f"unknown span_policy {name!r}; available: {sorted(_POLICIES)}"
        )
