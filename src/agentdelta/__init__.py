"""agentdelta — semantic diff engine for AI agent behavior."""

from agentdelta.batch import BatchDiffResult, batch_diff, batch_from_directory
from agentdelta.closed_loop import ClosedLoopError, GateOutcome, assert_no_regression, gate_traces
from agentdelta.diff import DiffResult, ForkPoint, diff_traces
from agentdelta.html_report import to_html
from agentdelta.instrument import AgentdeltaCallback, record
from agentdelta.score import RegressionScore, compute_score
from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode

__all__ = [
    "AgentTrace",
    "AgentdeltaCallback",
    "BatchDiffResult",
    "ClosedLoopError",
    "DiffResult",
    "EdgeType",
    "ForkPoint",
    "GateOutcome",
    "NodeType",
    "RegressionScore",
    "TraceEdge",
    "TraceNode",
    "assert_no_regression",
    "batch_diff",
    "batch_from_directory",
    "compute_score",
    "diff_traces",
    "gate_traces",
    "record",
    "to_html",
]

from importlib.metadata import version as _version

__version__ = _version("agentdelta")
