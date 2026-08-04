"""Closed-loop reader/gate for agentdelta (Non-Ornament L1).

Who reads the output?
  CI jobs, eagle-eyes ``dogfood_verify``, or any gate that must *fail loudly*
  when traces are empty, malformed, or behaviorally regress.

What outcome changes?
  Returns a structured ``GateOutcome`` with ``exit_code`` suitable for
  ``sys.exit`` — empty/wrong output is never a silent PASS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agentdelta.trace import AgentTrace

# Lazy imports for diff/score keep the empty-trace FAIL_LOUD path free of
# heavy deps (numpy / sentence-transformers) so eagle-eyes dogfood can call
# the gate without a full runtime install.


class ClosedLoopError(ValueError):
    """Raised when the gate refuses to score empty or unusable traces."""


@dataclass(frozen=True)
class GateOutcome:
    """Result of a closed-loop read of two agent traces.

    Attributes:
        ok: True only when the gate would let a pipeline continue (PASS/WARN).
        verdict: ``PASS``, ``WARN``, ``FAIL``, or ``FAIL_LOUD``.
        reason: Human-readable explanation (always non-empty).
        exit_code: 0 for PASS/WARN, 1 for FAIL, 2 for FAIL_LOUD (empty/wrong).
        score: Regression score when scoring ran; ``None`` on FAIL_LOUD.
        run_id_a / run_id_b: Trace identifiers when available.
        has_regression: Whether a behavioral fork was detected.
    """

    ok: bool
    verdict: str
    reason: str
    exit_code: int
    score: Any = None  # RegressionScore | None — typed loosely to avoid eager import
    run_id_a: str | None = None
    run_id_b: str | None = None
    has_regression: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON reports (eagle-eyes dogfood, CI artifacts)."""
        payload: dict[str, Any] = {
            "ok": self.ok,
            "verdict": self.verdict,
            "reason": self.reason,
            "exit_code": self.exit_code,
            "run_id_a": self.run_id_a,
            "run_id_b": self.run_id_b,
            "has_regression": self.has_regression,
            "score": None,
        }
        if self.score is not None:
            payload["score"] = asdict(self.score)
        return payload


def _fail_loud(reason: str, run_id_a: str | None = None, run_id_b: str | None = None) -> GateOutcome:
    return GateOutcome(
        ok=False,
        verdict="FAIL_LOUD",
        reason=reason,
        exit_code=2,
        score=None,
        run_id_a=run_id_a,
        run_id_b=run_id_b,
        has_regression=True,
    )


def _load_trace(source: AgentTrace | str | Path) -> AgentTrace:
    if isinstance(source, AgentTrace):
        return source
    path = Path(source)
    if not path.is_file():
        raise ClosedLoopError(f"trace file not found: {path}")
    return AgentTrace.load(path)


def gate_traces(
    baseline: AgentTrace | str | Path,
    candidate: AgentTrace | str | Path,
    *,
    pass_threshold: float = 80.0,
    warn_threshold: float = 60.0,
    fork_threshold: float = 0.70,
    match_threshold: float = 0.85,
    warn_is_ok: bool = True,
) -> GateOutcome:
    """Read two traces, score behavioral similarity, fail loudly on empty/wrong.

    This is the load-bearing closed-loop entry point for CI and eagle-eyes:

    * Empty baseline or candidate → ``FAIL_LOUD`` (exit 2), never silent PASS.
    * Diff with no aligned steps when either side had nodes → ``FAIL_LOUD``.
    * Score verdict ``FAIL`` → exit 1; ``PASS``/``WARN`` → exit 0 (WARN optional).

    Args:
        baseline: Baseline :class:`AgentTrace` or path to JSONL.
        candidate: Candidate :class:`AgentTrace` or path to JSONL.
        pass_threshold: Score >= this is ``PASS``.
        warn_threshold: Score >= this (but < pass) is ``WARN``.
        fork_threshold: Passed to :func:`diff_traces`.
        match_threshold: Passed to :func:`diff_traces`.
        warn_is_ok: If False, ``WARN`` is treated as not-ok (exit 1).

    Returns:
        :class:`GateOutcome` — callers should ``sys.exit(outcome.exit_code)``.
    """
    try:
        trace_a = _load_trace(baseline)
        trace_b = _load_trace(candidate)
    except ClosedLoopError as exc:
        return _fail_loud(str(exc))
    except Exception as exc:  # noqa: BLE001 — surface load errors as FAIL_LOUD
        return _fail_loud(f"trace load failed: {exc.__class__.__name__}: {exc}")

    run_a, run_b = trace_a.run_id, trace_b.run_id

    if len(trace_a) == 0:
        return _fail_loud("empty baseline trace — nothing to gate against", run_a, run_b)
    if len(trace_b) == 0:
        return _fail_loud("empty candidate trace — empty output is not a pass", run_a, run_b)

    try:
        from agentdelta.diff import diff_traces
        from agentdelta.score import compute_score
    except Exception as exc:  # noqa: BLE001
        return _fail_loud(
            f"scoring stack unavailable: {exc.__class__.__name__}: {exc}",
            run_a,
            run_b,
        )

    try:
        diff = diff_traces(
            trace_a,
            trace_b,
            fork_threshold=fork_threshold,
            match_threshold=match_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail_loud(
            f"diff failed: {exc.__class__.__name__}: {exc}",
            run_a,
            run_b,
        )

    if not diff.steps:
        return _fail_loud(
            "diff produced zero aligned steps despite non-empty traces",
            run_a,
            run_b,
        )

    required = {"total_steps", "matched", "changed", "added", "removed", "similarity_pct"}
    if not required.issubset(diff.summary.keys()):
        missing = sorted(required - set(diff.summary.keys()))
        return _fail_loud(f"diff summary missing keys: {missing}", run_a, run_b)

    score = compute_score(
        diff,
        pass_threshold=pass_threshold,
        warn_threshold=warn_threshold,
    )

    if score.verdict == "PASS":
        ok, exit_code = True, 0
    elif score.verdict == "WARN":
        ok, exit_code = (True, 0) if warn_is_ok else (False, 1)
    else:
        ok, exit_code = False, 1

    reason = (
        f"verdict={score.verdict} overall={score.overall:.1f} "
        f"has_regression={diff.has_regression}"
    )
    if diff.fork_point is not None:
        reason += f" fork_at=a{diff.fork_point.step_a}/b{diff.fork_point.step_b}"

    return GateOutcome(
        ok=ok,
        verdict=score.verdict,
        reason=reason,
        exit_code=exit_code,
        score=score,
        run_id_a=run_a,
        run_id_b=run_b,
        has_regression=diff.has_regression,
    )


def assert_no_regression(
    baseline: AgentTrace | str | Path,
    candidate: AgentTrace | str | Path,
    **kwargs: Any,
) -> GateOutcome:
    """Gate traces and raise :class:`ClosedLoopError` unless outcome is ok.

    Useful in unit tests and scripts that prefer exceptions over exit codes.
    """
    outcome = gate_traces(baseline, candidate, **kwargs)
    if not outcome.ok:
        raise ClosedLoopError(f"{outcome.verdict}: {outcome.reason}")
    return outcome
