"""agentdelta — semantic diff engine for AI agent behavior."""

from agentdelta.batch import BatchDiffResult, batch_diff, batch_from_directory
from agentdelta.diff import DiffResult, ForkPoint, diff_traces
from agentdelta.html_report import to_html
from agentdelta.instrument import AgentdeltaCallback, record
from agentdelta.score import RegressionScore, compute_score
from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode

__all__ = [
    "AgentTrace",
    "AgentdeltaCallback",
    "BatchDiffResult",
    "DiffResult",
    "EdgeType",
    "ForkPoint",
    "NodeType",
    "RegressionScore",
    "TraceEdge",
    "TraceNode",
    "batch_diff",
    "batch_from_directory",
    "compute_score",
    "diff_traces",
    "record",
    "to_html",
]

from importlib.metadata import version as _version

__version__ = _version("agentdelta")
