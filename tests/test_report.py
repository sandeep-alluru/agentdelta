"""Tests for agentdelta.report."""

from __future__ import annotations

import json

from agentdelta.diff import diff_traces
from agentdelta.report import to_json, to_markdown


def test_to_json_is_valid(simple_trace_a, simple_trace_b_match):
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    output = to_json(result)
    parsed = json.loads(output)
    assert parsed["run_id_a"] == "run_a"
    assert "summary" in parsed
    assert "steps" in parsed
    assert isinstance(parsed["steps"], list)


def test_to_json_fork_point_none(simple_trace_a):
    result = diff_traces(simple_trace_a, simple_trace_a)
    parsed = json.loads(to_json(result))
    assert parsed["fork_point"] is None


def test_to_markdown_contains_header(simple_trace_a, simple_trace_b_match):
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    md = to_markdown(result)
    assert "## agentdelta behavior diff" in md


def test_to_markdown_has_json_details(simple_trace_a, simple_trace_b_match):
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    md = to_markdown(result)
    assert "```json" in md
    assert "agentdelta" in md


def test_to_markdown_regression_emoji(simple_trace_a, simple_trace_b_fork):
    result = diff_traces(simple_trace_a, simple_trace_b_fork, fork_threshold=0.70)
    md = to_markdown(result)
    if result.has_regression:
        assert "🔴" in md
    else:
        assert "🟢" in md
