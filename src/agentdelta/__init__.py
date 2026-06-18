"""agentdelta — semantic diff engine for AI agent behavior."""

from agentdelta.diff import DiffResult, ForkPoint, diff_traces
from agentdelta.instrument import AgentDeltaCallback, record
from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode

__all__ = [
    "AgentDeltaCallback",
    "AgentTrace",
    "DiffResult",
    "EdgeType",
    "ForkPoint",
    "NodeType",
    "TraceEdge",
    "TraceNode",
    "diff_traces",
    "record",
]

from importlib.metadata import version as _version

__version__ = _version("agentdelta")
