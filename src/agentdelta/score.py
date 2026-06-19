"""Regression scoring — produce a single 0-100 behavioral similarity score for CI gates."""

from __future__ import annotations

from dataclasses import dataclass

from agentdelta.diff import DiffResult


@dataclass
class RegressionScore:
    """A single composite behavioral similarity score for CI pass/fail gates.

    Attributes:
        overall: Weighted composite score in [0, 100]. Higher = more similar.
        structural: Step-count and order similarity component.
        semantic: Content similarity of matched steps component.
        tool_fidelity: Percentage of tool calls that matched in the candidate.
        fork_penalty: Penalty for an early fork. 0 = fork at step 1, 100 = no fork.
        verdict: One of ``"PASS"``, ``"WARN"``, or ``"FAIL"``.
        threshold_pass: The pass threshold used.
        threshold_warn: The warn threshold used.
    """

    overall: float
    structural: float
    semantic: float
    tool_fidelity: float
    fork_penalty: float
    verdict: str
    threshold_pass: float
    threshold_warn: float

    def __str__(self) -> str:
        return (
            f"RegressionScore(overall={self.overall:.1f}, verdict={self.verdict!r}, "
            f"structural={self.structural:.1f}, semantic={self.semantic:.1f}, "
            f"tool_fidelity={self.tool_fidelity:.1f}, fork_penalty={self.fork_penalty:.1f})"
        )


def compute_score(
    diff_result: DiffResult,
    pass_threshold: float = 80.0,
    warn_threshold: float = 60.0,
) -> RegressionScore:
    """Compute a composite regression score from a DiffResult.

    Args:
        diff_result: The result of :func:`~agentdelta.diff.diff_traces`.
        pass_threshold: Score >= this value yields ``"PASS"``.
        warn_threshold: Score >= this value (but < pass) yields ``"WARN"``.
            Score below this yields ``"FAIL"``.

    Returns:
        A :class:`RegressionScore` with component scores and a verdict.
    """
    steps = diff_result.steps
    total = len(steps)

    if total == 0:
        # No steps → trivially identical
        return RegressionScore(
            overall=100.0,
            structural=100.0,
            semantic=100.0,
            tool_fidelity=100.0,
            fork_penalty=100.0,
            verdict="PASS",
            threshold_pass=pass_threshold,
            threshold_warn=warn_threshold,
        )

    # ── Structural score ─────────────────────────────────────────────────────
    # Ratio of matched+changed steps (both present) vs total alignment entries.
    both_present = sum(1 for s in steps if s.step_a is not None and s.step_b is not None)
    structural = (both_present / total) * 100.0

    # ── Semantic score ────────────────────────────────────────────────────────
    # Mean cosine similarity of steps that have both nodes (matched + changed).
    paired_sims = [s.similarity for s in steps if s.step_a is not None and s.step_b is not None]
    semantic = (sum(paired_sims) / len(paired_sims) * 100.0) if paired_sims else 100.0

    # ── Tool fidelity ─────────────────────────────────────────────────────────
    # Among tool-call steps in the baseline, what fraction was matched in candidate?
    from agentdelta.trace import NodeType

    baseline_tool_steps = [
        s
        for s in steps
        if s.step_a is not None and s.step_a.node_type == NodeType.TOOL_CALL
    ]
    if baseline_tool_steps:
        matched_tools = sum(1 for s in baseline_tool_steps if s.status == "match")
        tool_fidelity = (matched_tools / len(baseline_tool_steps)) * 100.0
    else:
        tool_fidelity = 100.0  # no tools → perfect fidelity

    # ── Fork penalty ──────────────────────────────────────────────────────────
    # If there is no fork, penalty = 100 (no penalty). If fork is at step 1,
    # penalty = 0 (maximum penalty). Scales linearly by step index.
    fp = diff_result.fork_point
    if fp is None:
        fork_penalty = 100.0
    else:
        # fork_step is 1-based; total based on baseline nodes count
        baseline_nodes = sum(1 for s in steps if s.step_a is not None)
        if baseline_nodes <= 1:
            fork_penalty = 0.0
        else:
            fork_penalty = ((fp.step_a - 1) / (baseline_nodes - 1)) * 100.0

    # ── Overall score ─────────────────────────────────────────────────────────
    # Weighted combination
    overall = (
        0.25 * structural
        + 0.35 * semantic
        + 0.20 * tool_fidelity
        + 0.20 * fork_penalty
    )
    overall = round(min(max(overall, 0.0), 100.0), 2)

    # ── Verdict ───────────────────────────────────────────────────────────────
    if overall >= pass_threshold:
        verdict = "PASS"
    elif overall >= warn_threshold:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return RegressionScore(
        overall=overall,
        structural=round(structural, 2),
        semantic=round(semantic, 2),
        tool_fidelity=round(tool_fidelity, 2),
        fork_penalty=round(fork_penalty, 2),
        verdict=verdict,
        threshold_pass=pass_threshold,
        threshold_warn=warn_threshold,
    )
