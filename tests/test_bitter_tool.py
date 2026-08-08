"""BITTER-TOOL — tool-call brittleness (arXiv 2608.06370).

Track B public research maps *The Bitter Lesson of Tool Calling* → agentdelta.
Failure class: orphan tool calls, silent tool errors under clean END, schema
gaps, parallel partial fail, context-rot stale result reuse.
"""

from __future__ import annotations

import pytest

from agentdelta.closed_loop import ClosedLoopError
from agentdelta.tool_calls import (
    ToolCallEvent,
    ToolResultEvent,
    analyze_tool_calls,
    assert_tool_calls_ok,
    extract_tool_events,
    gate_tool_calls,
)
from agentdelta.trace import AgentTrace, NodeType, TraceNode


def test_empty_require_tools_fails_loud() -> None:
    out = gate_tool_calls(require_tools=True)
    assert out.ok is False
    assert out.verdict == "FAIL_LOUD"
    assert out.exit_code == 2
    assert "BITTER-TOOL" in out.reason


def test_empty_not_required_passes() -> None:
    out = gate_tool_calls(require_tools=False)
    assert out.ok is True
    assert out.verdict == "PASS"


def test_orphan_call_fails() -> None:
    calls = [
        ToolCallEvent("c1", "search", 1, arguments={"q": "x"}, style="json"),
    ]
    out = gate_tool_calls(calls=calls, results=[], claimed_success=True)
    assert out.ok is False
    assert out.verdict == "FAIL"
    assert "orphan" in out.reason.lower()
    assert out.error_step_count == 1


def test_paired_ok_passes() -> None:
    calls = [ToolCallEvent("c1", "search", 1, {"q": "x"}, style="json")]
    results = [ToolResultEvent("c1", "search", 2, status="ok", content="hits")]
    out = gate_tool_calls(calls=calls, results=results, claimed_success=True)
    assert out.ok is True
    assert out.verdict == "PASS"
    payload = out.to_dict()
    assert payload["ok"] is True


def test_silent_tool_error_under_success_fails() -> None:
    """Paper failure: tool errors ignored while END claims success."""
    calls = [ToolCallEvent("c1", "db.query", 1, {"sql": "select 1"}, style="json")]
    results = [
        ToolResultEvent("c1", "db.query", 2, status="error", error="timeout"),
    ]
    out = gate_tool_calls(calls=calls, results=results, claimed_success=True)
    assert out.ok is False
    assert out.verdict == "FAIL"
    assert "claimed success" in out.reason.lower() or "failed tool" in out.reason.lower()
    assert out.human_required is True


def test_failed_tool_without_success_claim_ok_if_no_other_issues() -> None:
    """If agent does not claim success, failed tools alone are not silent-success."""
    calls = [ToolCallEvent("c1", "db", 1, style="json")]
    results = [ToolResultEvent("c1", "db", 2, status="error", error="boom")]
    out = gate_tool_calls(
        calls=calls,
        results=results,
        claimed_success=False,
        refuse_failed_with_success=True,
    )
    assert out.ok is True
    assert out.verdict == "PASS"


def test_schema_invalid_fails() -> None:
    calls = [
        ToolCallEvent("c1", "search", 1, arguments={}, style="json"),  # missing q
    ]
    results = [ToolResultEvent("c1", "search", 2, status="ok")]
    out = gate_tool_calls(
        calls=calls,
        results=results,
        schemas={"search": ["q"]},
        claimed_success=True,
    )
    assert out.ok is False
    assert "schema" in out.reason.lower()


def test_schema_valid_passes() -> None:
    calls = [ToolCallEvent("c1", "search", 1, {"q": "agent"}, style="json")]
    results = [ToolResultEvent("c1", "search", 2, status="ok")]
    out = gate_tool_calls(
        calls=calls,
        results=results,
        schemas={"search": ["q"]},
        claimed_success=True,
    )
    assert out.ok is True


def test_parallel_partial_fails() -> None:
    calls = [
        ToolCallEvent("a", "fetch", 1, parallel_group="fan1", style="programmatic"),
        ToolCallEvent("b", "fetch", 1, parallel_group="fan1", style="programmatic"),
    ]
    results = [
        ToolResultEvent("a", "fetch", 2, status="ok"),
        ToolResultEvent("b", "fetch", 2, status="error", error="429"),
    ]
    # claimed_success=False so the parallel-partial rule is the load-bearing fail
    # (silent-tool-error rule would also fire under claimed_success=True).
    out = gate_tool_calls(
        calls=calls,
        results=results,
        claimed_success=False,
        refuse_failed_with_success=False,
    )
    assert out.ok is False
    assert "parallel" in out.reason.lower()


def test_context_rot_stale_reuse_fails() -> None:
    calls = [ToolCallEvent("c1", "weather", 1, {"city": "SF"}, style="json")]
    results = [ToolResultEvent("c1", "weather", 2, status="ok", content="sun")]
    out = gate_tool_calls(
        calls=calls,
        results=results,
        claimed_success=True,
        max_result_age_steps=3,
        total_steps=20,  # age = 18 > 3
    )
    assert out.ok is False
    assert "stale" in out.reason.lower() or "context-rot" in out.reason.lower()


def test_fresh_result_not_stale() -> None:
    calls = [ToolCallEvent("c1", "weather", 18, style="json")]
    results = [ToolResultEvent("c1", "weather", 19, status="ok")]
    out = gate_tool_calls(
        calls=calls,
        results=results,
        claimed_success=True,
        max_result_age_steps=5,
        total_steps=20,
    )
    assert out.ok is True


def test_prefer_programmatic_warns_on_json_only() -> None:
    calls = [ToolCallEvent("c1", "t", 1, {"x": 1}, style="json")]
    results = [ToolResultEvent("c1", "t", 2, status="ok")]
    out = gate_tool_calls(
        calls=calls,
        results=results,
        claimed_success=True,
        prefer_programmatic=True,
    )
    assert out.ok is True
    assert out.verdict == "WARN"
    assert out.exit_code == 0


def test_from_agent_trace_orphan() -> None:
    t = AgentTrace(run_id="bt1")
    t.add_node(TraceNode(1, NodeType.START, "start"))
    t.add_node(
        TraceNode(
            2,
            NodeType.TOOL_CALL,
            "search",
            metadata={"tool": "search", "call_id": "tc1", "arguments": {"q": "x"}},
        )
    )
    # no TOOL_RETURN
    t.add_node(TraceNode(3, NodeType.END, "done"))
    out = gate_tool_calls(t, claimed_success=True)
    assert out.ok is False
    assert "orphan" in out.reason.lower()
    calls, results = extract_tool_events(t)
    assert len(calls) == 1
    assert len(results) == 0


def test_from_agent_trace_paired_ok() -> None:
    t = AgentTrace(run_id="bt2")
    t.add_node(TraceNode(1, NodeType.START, "start"))
    t.add_node(
        TraceNode(
            2,
            NodeType.TOOL_CALL,
            "search",
            metadata={
                "tool": "search",
                "call_id": "tc1",
                "arguments": {"q": "x"},
                "style": "programmatic",
            },
        )
    )
    t.add_node(
        TraceNode(
            3,
            NodeType.TOOL_RETURN,
            "hits",
            metadata={"tool": "search", "call_id": "tc1", "status": "ok"},
        )
    )
    t.add_node(TraceNode(4, NodeType.END, "answer"))
    out = gate_tool_calls(t)
    assert out.ok is True
    assert out.verdict == "PASS"
    analysis = analyze_tool_calls(t)
    assert analysis.call_count == 1
    assert analysis.result_count == 1
    assert analysis.orphan_call_ids == ()

def test_dict_events() -> None:
    out = gate_tool_calls(
        calls=[{"call_id": "1", "name": "x", "step": 1, "args": {"a": 1}}],
        results=[{"call_id": "1", "name": "x", "step": 2, "status": "ok"}],
        claimed_success=True,
    )
    assert out.ok is True


def test_assert_raises() -> None:
    with pytest.raises(ClosedLoopError):
        assert_tool_calls_ok(
            calls=[ToolCallEvent("c", "t", 1)],
            results=[],
            claimed_success=True,
        )


def test_arxiv_bitter_tool_fixture() -> None:
    """End-to-end public case: silent tool failure under final success claim."""
    # Pre-fix class: JSON tool path fails; agent still ends successfully
    calls = [
        {
            "call_id": "j1",
            "name": "code_exec",
            "step": 2,
            "style": "json",
            "arguments": {"code": "print(1)"},
        },
        {
            "call_id": "j2",
            "name": "code_exec",
            "step": 4,
            "style": "json",
            "arguments": {"code": "1/0"},
        },
    ]
    results = [
        {"call_id": "j1", "name": "code_exec", "step": 3, "status": "ok", "content": "1"},
        {
            "call_id": "j2",
            "name": "code_exec",
            "step": 5,
            "status": "error",
            "error": "ZeroDivisionError",
        },
    ]
    refuse = gate_tool_calls(calls=calls, results=results, claimed_success=True)
    assert refuse.ok is False
    assert refuse.verdict == "FAIL"
    assert "BITTER-TOOL" in refuse.reason

    # Post-fix class: recovered path — no success claim with failed tools,
    # or all tools ok under PTC style
    fixed_results = [
        {"call_id": "j1", "name": "code_exec", "step": 3, "status": "ok"},
        {"call_id": "j2", "name": "code_exec", "step": 5, "status": "ok", "content": "recovered"},
    ]
    # Use programmatic style (paper: PTC robust)
    ptc_calls = [
        {**c, "style": "programmatic"} for c in calls
    ]
    accept = gate_tool_calls(
        calls=ptc_calls,
        results=fixed_results,
        claimed_success=True,
        prefer_programmatic=True,
    )
    assert accept.ok is True
    assert accept.verdict == "PASS"


def test_analyze_to_dict() -> None:
    a = analyze_tool_calls(
        calls=[ToolCallEvent("c", "t", 1)],
        results=[],
        claimed_success=False,
    )
    d = a.to_dict()
    assert d["call_count"] == 1
    assert d["orphan_call_ids"] == ["c"]
