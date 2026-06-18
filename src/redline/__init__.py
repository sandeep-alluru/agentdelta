"""redline — semantic diff engine for AI agent behavior."""

from redline.diff import DiffResult, ForkPoint, diff_traces
from redline.instrument import RedlineCallback, record
from redline.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode

__all__ = [
    "AgentTrace",
    "DiffResult",
    "EdgeType",
    "ForkPoint",
    "NodeType",
    "RedlineCallback",
    "TraceEdge",
    "TraceNode",
    "diff_traces",
    "record",
]

from importlib.metadata import version as _version

__version__ = _version("redline")
