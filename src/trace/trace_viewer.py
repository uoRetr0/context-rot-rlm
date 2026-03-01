"""JSON export and pretty-print for execution traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.trace.tracer import TraceNode


def export_trace(trace: TraceNode, path: str | Path) -> None:
    """Export a trace tree to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(trace.to_dict(), f, indent=2)


def pretty_print(trace: TraceNode, indent: int = 0) -> str:
    """Pretty-print a trace tree."""
    lines = []
    prefix = "  " * indent
    conf = trace.metadata.get("confidence", "")
    conf_str = f" [conf={conf:.2f}]" if isinstance(conf, (int, float)) else ""
    dur_str = f" ({trace.duration:.1f}s)" if trace.duration else ""

    lines.append(f"{prefix}[{trace.action}]{conf_str}{dur_str}")

    if trace.input:
        inp = trace.input[:100]
        lines.append(f"{prefix}  in:  {inp}")
    if trace.output:
        out = trace.output[:100]
        lines.append(f"{prefix}  out: {out}")

    for child in trace.children:
        lines.append(pretty_print(child, indent + 1))

    return "\n".join(lines)
