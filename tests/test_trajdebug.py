"""TRAJDEBUG - error lifecycle on long-horizon trajectories.

Public case (Track B 20260807T201237Z):
  arXiv 2608.06346 TRAJDEBUG: Tracing Error Lifecycle to Identify Critical
  Failures in Long-Horizon LLM Agents. Final-answer-only scoring hides
  intermediate tool/LLM failures.

Also maps: DiagChain intermediate stages; Bitter Lesson of Tool Calling.
"""

from __future__ import annotations

import pytest

from agentdelta.closed_loop import (
    ClosedLoopError,
    TrajectoryStep,
    analyze_error_lifecycle,
    assert_error_lifecycle_ok,
    gate_error_lifecycle,
    node_is_failed,
)
from agentdelta.trace import AgentTrace, NodeType, TraceNode


def test_empty_trajectory_fails_loud() -> None:
    out = gate_error_lifecycle([])
    assert out.ok is False
    assert out.verdict == "FAIL_LOUD"
    assert out.exit_code == 2
    assert out.human_required is True
    assert "TRAJDEBUG" in out.reason


def test_clean_trajectory_passes() -> None:
    steps = [
        TrajectoryStep(1, "start", "ok"),
        TrajectoryStep(2, "tool:search", "ok"),
        TrajectoryStep(3, "end", "ok"),
    ]
    out = gate_error_lifecycle(steps)
    assert out.ok is True
    assert out.verdict == "PASS"
    assert out.error_step_count == 0
    assert out.critical_step is None


def test_unrecovered_error_fails() -> None:
    steps = [
        {"step": 1, "name": "plan", "status": "ok"},
        {"step": 2, "name": "tool:db", "status": "error", "message": "timeout"},
        {"step": 3, "name": "end", "status": "ok"},
    ]
    out = gate_error_lifecycle(steps, claimed_success=True)
    assert out.ok is False
    assert out.verdict == "FAIL"
    assert out.critical_step == 2
    assert out.error_step_count >= 1
    assert out.human_required is True
    assert "TRAJDEBUG" in out.reason
    payload = out.to_dict()
    assert payload["critical_step"] == 2


def test_recovered_error_allowed() -> None:
    steps = [
        TrajectoryStep(1, "tool:a", "error", message="retryable", recovered=True),
        TrajectoryStep(2, "tool:a", "ok"),
        TrajectoryStep(3, "end", "ok"),
    ]
    out = gate_error_lifecycle(steps, claimed_success=True, max_unrecovered=0)
    assert out.ok is True
    assert out.verdict == "PASS"


def test_claimed_success_with_errors_fails() -> None:
    life = analyze_error_lifecycle(
        [
            TrajectoryStep(1, "llm", "ok"),
            TrajectoryStep(2, "tool", "fail", message="boom"),
            TrajectoryStep(3, "end", "ok"),
        ],
        claimed_success=True,
    )
    assert life.claimed_success is True
    assert life.unrecovered_count == 1
    out = gate_error_lifecycle(
        [
            TrajectoryStep(1, "llm", "ok"),
            TrajectoryStep(2, "tool", "fail", message="boom"),
            TrajectoryStep(3, "end", "ok"),
        ],
        claimed_success=True,
    )
    assert out.ok is False
    assert "claimed success" in out.reason.lower() or "TRAJDEBUG" in out.reason


def test_agent_trace_with_failed_tool_return() -> None:
    t = AgentTrace(run_id="r1")
    t.nodes = [
        TraceNode(1, NodeType.START, "begin"),
        TraceNode(2, NodeType.TOOL_CALL, "db.query()"),
        TraceNode(
            3,
            NodeType.TOOL_RETURN,
            "Error: connection refused",
            metadata={"status": "error", "error": "connection refused"},
        ),
        TraceNode(4, NodeType.END, "All good!"),
    ]
    assert node_is_failed(t.nodes[2]) is True
    out = gate_error_lifecycle(t)
    assert out.ok is False
    assert out.verdict == "FAIL"
    assert out.critical_step == 3


def test_agent_trace_clean() -> None:
    t = AgentTrace(run_id="r2")
    t.nodes = [
        TraceNode(1, NodeType.START, "begin"),
        TraceNode(2, NodeType.LLM, "think"),
        TraceNode(3, NodeType.TOOL_CALL, "search(q)"),
        TraceNode(4, NodeType.TOOL_RETURN, "results...", metadata={"status": "ok"}),
        TraceNode(5, NodeType.END, "answer"),
    ]
    out = gate_error_lifecycle(t)
    assert out.ok is True


def test_assert_error_lifecycle_ok_raises() -> None:
    with pytest.raises(ClosedLoopError):
        assert_error_lifecycle_ok(
            [TrajectoryStep(1, "x", "error")],
            claimed_success=True,
        )


def test_assert_error_lifecycle_ok_passes() -> None:
    out = assert_error_lifecycle_ok([TrajectoryStep(1, "x", "ok")])
    assert out.ok is True
