"""Tests for agentdelta.batch — BatchDiffResult, batch_diff, batch_from_directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentdelta.batch import BatchDiffResult, batch_diff, batch_from_directory
from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode


def _make_trace(run_id: str, steps: list[tuple[NodeType, str]]) -> AgentTrace:
    trace = AgentTrace(run_id=run_id)
    for i, (ntype, content) in enumerate(steps, start=1):
        trace.add_node(TraceNode(step=i, node_type=ntype, content=content))
        if i > 1:
            trace.add_edge(TraceEdge(i - 1, i, EdgeType.SEQUENCE, ""))
    return trace


_IDENTICAL_STEPS: list[tuple[NodeType, str]] = [
    (NodeType.START, "user asks about weather"),
    (NodeType.LLM, "I will check the weather tool"),
    (NodeType.TOOL_CALL, "get_weather(city='Paris')"),
    (NodeType.TOOL_RETURN, '{"temp": 22}'),
    (NodeType.END, "It is 22 degrees in Paris"),
]


@pytest.fixture
def matching_pair() -> tuple[AgentTrace, AgentTrace]:
    a = _make_trace("base", _IDENTICAL_STEPS)
    b = _make_trace("cand", _IDENTICAL_STEPS)
    return a, b


@pytest.fixture
def forked_pair() -> tuple[AgentTrace, AgentTrace]:
    a = _make_trace(
        "base_fork",
        [
            (NodeType.START, "user wants a summary"),
            (NodeType.TOOL_CALL, "summarize(text='...')"),
            (NodeType.END, "summary here"),
        ],
    )
    b = _make_trace(
        "cand_fork",
        [
            (NodeType.START, "user wants a summary"),
            (NodeType.TOOL_CALL, "web_search(q='find summary')"),
            (NodeType.END, "found something"),
        ],
    )
    return a, b


# ── batch_diff ─────────────────────────────────────────────────────────────


def test_batch_diff_returns_batch_result(matching_pair: tuple[AgentTrace, AgentTrace]) -> None:
    result = batch_diff([matching_pair])
    assert isinstance(result, BatchDiffResult)


def test_batch_diff_single_matching_pair(matching_pair: tuple[AgentTrace, AgentTrace]) -> None:
    """A single identical pair should produce aggregate_score ≥ 80 and no regressions."""
    result = batch_diff([matching_pair])
    assert len(result) == 1
    assert result.aggregate_score >= 80.0
    assert not result.has_regressions


def test_batch_diff_multiple_pairs(
    matching_pair: tuple[AgentTrace, AgentTrace],
    forked_pair: tuple[AgentTrace, AgentTrace],
) -> None:
    """Multiple pairs are all processed; length matches input count."""
    result = batch_diff([matching_pair, forked_pair])
    assert len(result) == 2


def test_batch_diff_forked_pair_detected_as_regression(
    forked_pair: tuple[AgentTrace, AgentTrace],
) -> None:
    """A forked pair with strict thresholds should appear in regressions."""
    result = batch_diff([forked_pair], pass_threshold=95.0, warn_threshold=85.0)
    assert result.has_regressions


def test_batch_diff_aggregate_score_bounded() -> None:
    """aggregate_score must be in [0, 100]."""
    a = _make_trace("a", _IDENTICAL_STEPS)
    b = _make_trace("b", _IDENTICAL_STEPS)
    result = batch_diff([(a, b)])
    assert 0.0 <= result.aggregate_score <= 100.0


def test_batch_diff_empty_input() -> None:
    """Empty input yields a BatchDiffResult with default perfect aggregate score."""
    result = batch_diff([])
    assert len(result) == 0
    assert result.aggregate_score == 100.0
    assert not result.has_regressions


def test_batch_diff_pairs_carry_run_ids(
    matching_pair: tuple[AgentTrace, AgentTrace],
) -> None:
    """Each pair tuple should contain the correct run IDs."""
    a, b = matching_pair
    result = batch_diff([(a, b)])
    baseline_id, candidate_id, _ = result.pairs[0]
    assert baseline_id == a.run_id
    assert candidate_id == b.run_id


# ── batch_from_directory ───────────────────────────────────────────────────


def test_batch_from_directory_matches_files(tmp_path: Path) -> None:
    """Only files present in both directories should be diffed."""
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()

    a = _make_trace("run1", _IDENTICAL_STEPS)
    b = _make_trace("run1", _IDENTICAL_STEPS)
    a.save(baseline / "run1.jsonl")
    b.save(candidate / "run1.jsonl")

    # Extra file only in baseline — should be ignored
    extra = _make_trace("extra", [(NodeType.START, "extra")])
    extra.save(baseline / "extra.jsonl")

    result = batch_from_directory(baseline, candidate)
    assert len(result) == 1


def test_batch_from_directory_raises_for_missing_dir(tmp_path: Path) -> None:
    """FileNotFoundError if a directory does not exist."""
    with pytest.raises(FileNotFoundError):
        batch_from_directory(tmp_path / "nonexistent", tmp_path / "also_nonexistent")


def test_batch_from_directory_empty_dirs(tmp_path: Path) -> None:
    """Two empty directories yield a BatchDiffResult with 0 pairs."""
    base = tmp_path / "base"
    cand = tmp_path / "cand"
    base.mkdir()
    cand.mkdir()
    result = batch_from_directory(base, cand)
    assert len(result) == 0
