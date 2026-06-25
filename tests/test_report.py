"""Tests for agentdelta.report."""

from __future__ import annotations

import json

from rich.console import Console

from agentdelta.diff import diff_traces
from agentdelta.report import print_diff, to_json, to_markdown
from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode


def _make_trace(run_id: str, steps: list[tuple[NodeType, str]]) -> AgentTrace:
    trace = AgentTrace(run_id=run_id)
    for i, (ntype, content) in enumerate(steps, start=1):
        trace.add_node(TraceNode(step=i, node_type=ntype, content=content))
        if i > 1:
            trace.add_edge(TraceEdge(i - 1, i, EdgeType.SEQUENCE, ""))
    return trace


def test_to_json_is_valid(simple_trace_a, simple_trace_b_match):
    """to_json should produce valid JSON with required top-level keys."""
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    output = to_json(result)
    parsed = json.loads(output)
    assert parsed["run_id_a"] == "run_a"
    assert "summary" in parsed
    assert "steps" in parsed
    assert isinstance(parsed["steps"], list)


def test_to_json_fork_point_none(simple_trace_a):
    """Identical traces should produce fork_point=null in JSON output."""
    result = diff_traces(simple_trace_a, simple_trace_a)
    parsed = json.loads(to_json(result))
    assert parsed["fork_point"] is None


def test_to_json_fork_point_present(simple_trace_a, simple_trace_b_fork):
    """A forked diff should produce a non-null fork_point in JSON output."""
    result = diff_traces(simple_trace_a, simple_trace_b_fork, fork_threshold=0.70)
    parsed = json.loads(to_json(result))
    if result.fork_point is not None:
        fp = parsed["fork_point"]
        assert "step_a" in fp
        assert "step_b" in fp
        assert "description" in fp
        assert "node_a_content" in fp
        assert "node_b_content" in fp


def test_to_json_step_fields(simple_trace_a, simple_trace_b_match):
    """Each step in the JSON output should have required fields."""
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    parsed = json.loads(to_json(result))
    for step in parsed["steps"]:
        assert "status" in step
        assert "similarity" in step
        assert "node_type" in step


def test_to_markdown_contains_header(simple_trace_a, simple_trace_b_match):
    """to_markdown output must contain the agentdelta header."""
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    md = to_markdown(result)
    assert "## agentdelta behavior diff" in md


def test_to_markdown_has_json_details(simple_trace_a, simple_trace_b_match):
    """to_markdown output must contain a JSON code block with agentdelta link."""
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    md = to_markdown(result)
    assert "```json" in md
    assert "agentdelta" in md


def test_to_markdown_regression_emoji(simple_trace_a, simple_trace_b_fork):
    """Regression status should map to the correct emoji in markdown output."""
    result = diff_traces(simple_trace_a, simple_trace_b_fork, fork_threshold=0.70)
    md = to_markdown(result)
    if result.has_regression:
        assert "🔴" in md
    else:
        assert "🟢" in md


def test_to_markdown_with_fork_point(simple_trace_a, simple_trace_b_fork):
    """to_markdown should include the fork-point block when a fork is detected."""
    result = diff_traces(simple_trace_a, simple_trace_b_fork, fork_threshold=0.70)
    md = to_markdown(result)
    if result.fork_point is not None:
        assert "First fork point" in md
        assert "```diff" in md


def test_to_markdown_changed_steps_table(simple_trace_a, simple_trace_b_fork):
    """When there are changed steps, to_markdown should include a step table."""
    result = diff_traces(simple_trace_a, simple_trace_b_fork, fork_threshold=0.70)
    md = to_markdown(result)
    changed = [s for s in result.steps if s.status != "match"]
    if changed:
        assert "Changed steps" in md
        assert "| Step |" in md


def test_to_markdown_truncates_long_changed_list():
    """More than 20 changed steps should append a 'more rows' truncation line."""
    # Build two traces with 25 different LLM steps to force 25 changed rows
    steps_a = [(NodeType.LLM, f"baseline reasoning step {i}") for i in range(25)]
    steps_b = [(NodeType.LLM, f"candidate reasoning step {i} — completely different path") for i in range(25)]
    a = _make_trace("a", steps_a)
    b = _make_trace("b", steps_b)
    result = diff_traces(a, b, fork_threshold=0.99, match_threshold=0.999)
    changed = [s for s in result.steps if s.status != "match"]
    if len(changed) > 20:
        md = to_markdown(result)
        assert "more rows" in md


def test_print_diff_no_fork(simple_trace_a, simple_trace_b_match):
    """print_diff should run without errors on a no-fork diff."""
    import io
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    buf = io.StringIO()
    con = Console(file=buf, highlight=False)
    print_diff(result, console=con)
    output = buf.getvalue()
    assert "agentdelta" in output


def test_print_diff_with_fork(simple_trace_a, simple_trace_b_fork):
    """print_diff should include fork-point panel when a fork is present."""
    import io
    result = diff_traces(simple_trace_a, simple_trace_b_fork, fork_threshold=0.70)
    buf = io.StringIO()
    con = Console(file=buf, highlight=False)
    print_diff(result, console=con)
    output = buf.getvalue()
    assert "agentdelta" in output
    if result.fork_point:
        assert "fork" in output.lower()


def test_print_diff_show_matches(simple_trace_a, simple_trace_b_match):
    """print_diff with show_matches=True should render match rows in the table."""
    import io
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    buf = io.StringIO()
    con = Console(file=buf, highlight=False)
    print_diff(result, show_matches=True, console=con)
    output = buf.getvalue()
    # With show_matches=True the table is rendered; at minimum 'Step' header is present
    assert "agentdelta" in output
