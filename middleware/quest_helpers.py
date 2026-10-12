"""Quest-only middleware helpers."""

import json
from typing import Any, Dict, List, Optional


def coerce_tool_call_args(
    tool_calls: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Retype parser-stringified non-string tool arguments from declared schemas."""
    if not tool_calls or not tools:
        return tool_calls

    non_string = {"array", "object", "integer", "number", "boolean"}

    def declared_types(spec: Any) -> set[str]:
        if not isinstance(spec, dict):
            return set()
        out: set[str] = set()
        typ = spec.get("type")
        if isinstance(typ, str):
            out.add(typ)
        elif isinstance(typ, list):
            out.update(x for x in typ if isinstance(x, str))
        for sub in spec.get("anyOf") or spec.get("oneOf") or spec.get("allOf") or []:
            out |= declared_types(sub)
        return out

    schema: Dict[str, Dict[str, set[str]]] = {}
    for tool in tools:
        fn = (tool or {}).get("function") or {}
        name = fn.get("name")
        props = ((fn.get("parameters") or {}).get("properties")) or {}
        if name and isinstance(props, dict):
            schema[name] = {p: declared_types(spec) for p, spec in props.items()}

    for tool_call in tool_calls:
        fn = (tool_call or {}).get("function") or {}
        want = schema.get(fn.get("name"))
        raw = fn.get("arguments")
        if not want or not isinstance(raw, str):
            continue
        try:
            args = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(args, dict):
            continue
        changed = False
        for key, value in list(args.items()):
            types = want.get(key) or set()
            if not isinstance(value, str) or not (types & non_string) or "string" in types:
                continue
            try:
                args[key] = json.loads(value)
            except (TypeError, ValueError):
                continue
            changed = True
        if changed:
            fn["arguments"] = json.dumps(args)
    return tool_calls
