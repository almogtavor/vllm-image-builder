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
    """Every message is its own span (default)."""

    def select_spans(self, messages: List[dict]) -> Set[int]:
        return set(range(len(messages)))


class ToolFileReadSpanPolicy(SpanPolicy):
    """Span = ``bash`` tool response whose invoking command reads a code file.

    A command qualifies when a read verb (cat/head/tail/grep/rg/sed/awk/
    less/more) and a code-file extension both appear, and none of the
    :attr:`_REJECTERS` (writes, ``sed -i``, interpreter at command position,
    ``grep -l``) match. Pure regex; no shell parsing.
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
        # Map each bash tool_call id -> its command string.
        cmd_by_id: Dict[str, str] = {}
        for m in messages:
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                if not self._is_bash(fn.get("name")):
                    continue
                cmd = self._command(fn.get("arguments"))
                if cmd and tc.get("id"):
                    cmd_by_id[tc["id"]] = cmd

        spans: Set[int] = set()
        for i, m in enumerate(messages):
            if m.get("role") != "tool":
                continue
            cmd = cmd_by_id.get(m.get("tool_call_id"))
            if cmd and self._is_code_read(cmd):
                spans.add(i)
                logger.info(
                    "tool_file_read: caught msg[%d] as PIC span — code-file read: %s",
                    i, cmd[:120],
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
    def _command(arguments: object) -> Optional[str]:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (ValueError, TypeError):
                return None
        if isinstance(arguments, dict):
            cmd = arguments.get("command")
            return cmd if isinstance(cmd, str) else None
        return None

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
