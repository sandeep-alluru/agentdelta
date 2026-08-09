"""agentdelta - semantic diff engine for AI agent behavior."""

from agentdelta.batch import BatchDiffResult, batch_diff, batch_from_directory
from agentdelta.closed_loop import (
    ClosedLoopError,
    ErrorLifecycle,
    GateOutcome,
    TrajectoryStep,
    analyze_error_lifecycle,
    answer_fingerprint,
    assert_error_lifecycle_ok,
    assert_no_regression,
    e2e_content_swap_rejudges,
    e2e_reader_after_write,
    gate_error_lifecycle,
    gate_from_disk,
    gate_traces,
    node_is_failed,
    path_fingerprint,
)
from agentdelta.diff import DiffResult, ForkPoint, diff_traces
from agentdelta.html_report import to_html
from agentdelta.instrument import AgentdeltaCallback, record
from agentdelta.score import RegressionScore, compute_score
from agentdelta.tool_calls import (
    ToolCallAnalysis,
    ToolCallEvent,
    ToolResultEvent,
    analyze_tool_calls,
    assert_tool_calls_ok,
    extract_tool_events,
    gate_tool_calls,
)
from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode

__all__ = [
    "AgentTrace",
    "AgentdeltaCallback",
    "BatchDiffResult",
    "ClosedLoopError",
    "DiffResult",
    "EdgeType",
    "ErrorLifecycle",
    "ForkPoint",
    "GateOutcome",
    "NodeType",
    "RegressionScore",
    "ToolCallAnalysis",
    "ToolCallEvent",
    "ToolResultEvent",
    "TraceEdge",
    "TraceNode",
    "TrajectoryStep",
    "analyze_error_lifecycle",
    "analyze_tool_calls",
    "answer_fingerprint",
    "assert_error_lifecycle_ok",
    "assert_no_regression",
    "assert_tool_calls_ok",
    "batch_diff",
    "batch_from_directory",
    "compute_score",
    "diff_traces",
    "e2e_content_swap_rejudges",
    "e2e_reader_after_write",
    "extract_tool_events",
    "gate_error_lifecycle",
    "gate_from_disk",
    "gate_tool_calls",
    "gate_traces",
    "node_is_failed",
    "path_fingerprint",
    "record",
    "to_html",
]

from importlib.metadata import version as _version

__version__ = _version("agentdelta")
