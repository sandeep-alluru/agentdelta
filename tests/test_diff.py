"""Tests for agentdelta.diff."""

from __future__ import annotations

from agentdelta.diff import DiffResult, ForkPoint, diff_traces


def test_no_regression_on_similar_traces(simple_trace_a, simple_trace_b_match):
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    assert isinstance(result, DiffResult)
    assert result.summary["total_steps"] > 0
    # Similar traces should have high similarity
    assert result.summary["similarity_pct"] >= 50.0


def test_fork_detected_on_diverging_traces(simple_trace_a, simple_trace_b_fork):
    result = diff_traces(
        simple_trace_a,
        simple_trace_b_fork,
        fork_threshold=0.70,
        match_threshold=0.85,
    )
    # There should be some changed or unmatched steps
    assert len(result.changed_steps) + len(result.added_steps) + len(result.removed_steps) > 0


def test_result_has_run_ids(simple_trace_a, simple_trace_b_match):
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    assert result.run_id_a == "run_a"
    assert result.run_id_b == "run_b"


def test_summary_keys_present(simple_trace_a, simple_trace_b_match):
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    required = {"total_steps", "matched", "changed", "added", "removed",
                "similarity_pct", "has_regression", "fork_step"}
    assert required.issubset(result.summary.keys())


def test_step_statuses_valid(simple_trace_a, simple_trace_b_match):
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    valid = {"match", "changed", "added", "removed"}
    for step in result.steps:
        assert step.status in valid


def test_fork_point_is_fork_type(simple_trace_a, simple_trace_b_fork):
    result = diff_traces(simple_trace_a, simple_trace_b_fork, fork_threshold=0.70)
    if result.fork_point:
        fp = result.fork_point
        assert isinstance(fp, ForkPoint)
        assert fp.node_a is not None
        assert fp.node_b is not None
        assert 0.0 <= fp.similarity <= 1.0


def test_identical_traces_no_changes(simple_trace_a):
    result = diff_traces(simple_trace_a, simple_trace_a)
    assert result.summary["changed"] == 0
    assert result.summary["added"] == 0
    assert result.summary["removed"] == 0
    assert result.summary["similarity_pct"] == 100.0


def test_has_regression_property(simple_trace_a, simple_trace_b_match):
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    assert result.has_regression == (result.fork_point is not None)
