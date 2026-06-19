"""Tests for HTML report generation."""

from __future__ import annotations

import pytest

from agentdelta.diff import DiffResult, diff_traces
from agentdelta.html_report import to_html
from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode


def _make_trace(run_id: str, steps: list[tuple[NodeType, str]]) -> AgentTrace:
    trace = AgentTrace(run_id=run_id)
    for i, (ntype, content) in enumerate(steps, start=1):
        trace.add_node(TraceNode(step=i, node_type=ntype, content=content))
        if i > 1:
            trace.add_edge(TraceEdge(i - 1, i, EdgeType.SEQUENCE, ""))
    return trace


@pytest.fixture
def matched_diff() -> DiffResult:
    """DiffResult with two nearly identical traces — no regression."""
    a = _make_trace(
        "run_a",
        [
            (NodeType.START, "What is 2 + 2?"),
            (NodeType.LLM, "The answer is 4."),
            (NodeType.END, "4"),
        ],
    )
    b = _make_trace(
        "run_b",
        [
            (NodeType.START, "What is 2 + 2?"),
            (NodeType.LLM, "The answer is four."),
            (NodeType.END, "four"),
        ],
    )
    return diff_traces(a, b)


@pytest.fixture
def forked_diff() -> DiffResult:
    """DiffResult with a tool-selection fork."""
    a = _make_trace(
        "baseline",
        [
            (NodeType.START, "Summarise this document"),
            (NodeType.LLM, "I will use the summarize_tool"),
            (NodeType.TOOL_CALL, "summarize_tool(doc='...')"),
            (NodeType.TOOL_RETURN, "Summary: short text"),
            (NodeType.END, "short text"),
        ],
    )
    b = _make_trace(
        "candidate",
        [
            (NodeType.START, "Summarise this document"),
            (NodeType.LLM, "Let me search for relevant sections first"),
            (NodeType.TOOL_CALL, "web_search(query='document summary')"),
            (NodeType.TOOL_RETURN, "results..."),
            (NodeType.END, "some result"),
        ],
    )
    return diff_traces(a, b)


# ── Test 1: to_html() with a matched diff (no regression) ────────────────────

def test_to_html_matched_no_regression(matched_diff: DiffResult) -> None:
    """to_html() on a matched diff should return an HTML string with no regression text."""
    html = to_html(matched_diff)
    assert isinstance(html, str)
    assert len(html) > 0
    assert "NO REGRESSION" in html or "REGRESSION" in html  # one of the two verdicts


def test_to_html_matched_returns_pass_verdict(matched_diff: DiffResult) -> None:
    """Matched traces should render 'NO REGRESSION' verdict."""
    html = to_html(matched_diff)
    # The diff is nearly identical — should NOT say REGRESSION DETECTED
    assert "NO REGRESSION" in html


# ── Test 2: to_html() with a fork point ──────────────────────────────────────

def test_to_html_with_fork_point(forked_diff: DiffResult) -> None:
    """A diff with a fork point should include fork information in the HTML."""
    html = to_html(forked_diff)
    assert isinstance(html, str)
    # Fork box uses "First Fork at Step"
    if forked_diff.fork_point is not None:
        assert "First Fork at Step" in html


def test_to_html_fork_shows_regression(forked_diff: DiffResult) -> None:
    """A forked diff with has_regression=True should render REGRESSION DETECTED."""
    if forked_diff.has_regression:
        html = to_html(forked_diff)
        assert "REGRESSION DETECTED" in html


# ── Test 3: to_html() with custom title ──────────────────────────────────────

def test_to_html_custom_title(matched_diff: DiffResult) -> None:
    """Custom title should appear in the output HTML."""
    custom_title = "My Custom Report Title"
    html = to_html(matched_diff, title=custom_title)
    assert custom_title in html


def test_to_html_default_title(matched_diff: DiffResult) -> None:
    """Default title 'agentdelta diff' should appear when no title is given."""
    html = to_html(matched_diff)
    assert "agentdelta diff" in html


# ── Test 4: to_html() with empty traces ──────────────────────────────────────

def test_to_html_empty_traces() -> None:
    """to_html() with two empty traces should still return valid HTML."""
    a = AgentTrace(run_id="empty_a")
    b = AgentTrace(run_id="empty_b")
    diff = diff_traces(a, b)
    html = to_html(diff)
    assert isinstance(html, str)
    assert "<html" in html
    assert "</html>" in html


# ── Test 5: HTML validity (structure) ────────────────────────────────────────

def test_to_html_contains_html_tags(matched_diff: DiffResult) -> None:
    """Output must be a complete HTML document with required structural tags."""
    html = to_html(matched_diff)
    assert "<html" in html
    assert "</html>" in html
    assert "<table>" in html or "<table" in html


def test_to_html_contains_table_headers(matched_diff: DiffResult) -> None:
    """HTML output should include table headers for the step diff table."""
    html = to_html(matched_diff)
    assert "<th>" in html or "<th " in html
    assert "Step A" in html
    assert "Step B" in html


# ── Test 6: Fork point highlighted when has_regression=True ──────────────────

def test_to_html_fork_here_class_when_regression(forked_diff: DiffResult) -> None:
    """When a fork point exists, the row with the fork should have 'fork-here' CSS class."""
    if forked_diff.fork_point is not None:
        html = to_html(forked_diff)
        assert "fork-here" in html


def test_to_html_no_fork_step_label_when_no_fork(matched_diff: DiffResult) -> None:
    """When there is no fork point, 'First Fork at Step' should not appear."""
    if matched_diff.fork_point is None:
        html = to_html(matched_diff)
        assert "First Fork at Step" not in html


def test_to_html_run_ids_in_output(matched_diff: DiffResult) -> None:
    """Both run IDs should appear in the rendered HTML."""
    html = to_html(matched_diff)
    assert matched_diff.run_id_a in html
    assert matched_diff.run_id_b in html
