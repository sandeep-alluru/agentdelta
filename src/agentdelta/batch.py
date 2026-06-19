"""Batch diff operations across multiple trace pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agentdelta.diff import DiffResult, diff_traces
from agentdelta.score import compute_score
from agentdelta.trace import AgentTrace


@dataclass
class BatchDiffResult:
    """Results of diffing multiple trace pairs in bulk.

    Attributes:
        pairs: List of ``(baseline_id, candidate_id, DiffResult)`` triples.
        aggregate_score: Mean regression score across all pairs (0-100).
        regressions: Run-ID pairs (as ``"baseline_id vs candidate_id"`` strings)
            where a regression was detected.
    """

    pairs: list[tuple[str, str, DiffResult]] = field(default_factory=list)
    aggregate_score: float = 100.0
    regressions: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def has_regressions(self) -> bool:
        """True if any pair regressed."""
        return bool(self.regressions)


def batch_diff(
    pairs: list[tuple[AgentTrace, AgentTrace]],
    fork_threshold: float = 0.70,
    match_threshold: float = 0.85,
    pass_threshold: float = 80.0,
    warn_threshold: float = 60.0,
) -> BatchDiffResult:
    """Diff multiple trace pairs and aggregate results.

    Args:
        pairs: List of ``(baseline_trace, candidate_trace)`` tuples.
        fork_threshold: Passed to :func:`~agentdelta.diff.diff_traces`.
        match_threshold: Passed to :func:`~agentdelta.diff.diff_traces`.
        pass_threshold: Threshold for :func:`~agentdelta.score.compute_score`.
        warn_threshold: Threshold for :func:`~agentdelta.score.compute_score`.

    Returns:
        :class:`BatchDiffResult` with per-pair results and aggregate statistics.
    """
    result_pairs: list[tuple[str, str, DiffResult]] = []
    scores: list[float] = []
    regressions: list[str] = []

    for trace_a, trace_b in pairs:
        dr = diff_traces(
            trace_a,
            trace_b,
            fork_threshold=fork_threshold,
            match_threshold=match_threshold,
        )
        score = compute_score(dr, pass_threshold=pass_threshold, warn_threshold=warn_threshold)
        result_pairs.append((trace_a.run_id, trace_b.run_id, dr))
        scores.append(score.overall)
        if score.verdict == "FAIL":
            regressions.append(f"{trace_a.run_id} vs {trace_b.run_id}")

    aggregate = round(sum(scores) / len(scores), 2) if scores else 100.0

    return BatchDiffResult(
        pairs=result_pairs,
        aggregate_score=aggregate,
        regressions=regressions,
    )


def batch_from_directory(
    baseline_dir: Path,
    candidate_dir: Path,
    fork_threshold: float = 0.70,
    match_threshold: float = 0.85,
    pass_threshold: float = 80.0,
    warn_threshold: float = 60.0,
) -> BatchDiffResult:
    """Diff all matching ``*.jsonl`` trace files in two directories.

    Only files that exist in *both* directories (by filename) are diffed.
    Files present in one directory but not the other are silently skipped.

    Args:
        baseline_dir: Directory of baseline trace files.
        candidate_dir: Directory of candidate trace files.
        fork_threshold: Passed to :func:`~agentdelta.diff.diff_traces`.
        match_threshold: Passed to :func:`~agentdelta.diff.diff_traces`.
        pass_threshold: Threshold for :func:`~agentdelta.score.compute_score`.
        warn_threshold: Threshold for :func:`~agentdelta.score.compute_score`.

    Returns:
        :class:`BatchDiffResult` with per-pair results.

    Raises:
        FileNotFoundError: If either directory does not exist.
    """
    baseline_dir = Path(baseline_dir)
    candidate_dir = Path(candidate_dir)

    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")
    if not candidate_dir.is_dir():
        raise FileNotFoundError(f"Candidate directory not found: {candidate_dir}")

    baseline_files = {p.name: p for p in baseline_dir.glob("*.jsonl")}
    candidate_files = {p.name: p for p in candidate_dir.glob("*.jsonl")}
    common_names = sorted(set(baseline_files) & set(candidate_files))

    pairs: list[tuple[AgentTrace, AgentTrace]] = []
    for name in common_names:
        trace_a = AgentTrace.load(baseline_files[name])
        trace_b = AgentTrace.load(candidate_files[name])
        pairs.append((trace_a, trace_b))

    return batch_diff(
        pairs,
        fork_threshold=fork_threshold,
        match_threshold=match_threshold,
        pass_threshold=pass_threshold,
        warn_threshold=warn_threshold,
    )
