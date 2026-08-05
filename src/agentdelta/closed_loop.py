"""Closed-loop reader/gate for agentdelta (Non-Ornament L1).

Who reads the output?
  CI jobs, eagle-eyes ``dogfood_verify``, or any gate that must *fail loudly*
  when traces are empty, malformed, or behaviorally regress.

What outcome changes?
  Returns a structured ``GateOutcome`` with ``exit_code`` suitable for
  ``sys.exit`` — empty/wrong output is never a silent PASS.

WRITER-NOT-READER (farm lesson):
  Fixes that only touch the *writer* (save path, cache key, in-memory object)
  while leaving *readers* on a collapsed key (final-answer-only, stale path,
  never reloading disk) are incomplete. This module exposes:

  * :func:`path_fingerprint` / :func:`answer_fingerprint` — full path vs END-only
  * :func:`gate_from_disk` — reader always reloads JSONL from disk
  * :func:`e2e_reader_after_write` — write then gate via disk only
  * answer-only collapse refusal inside :func:`gate_traces`
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agentdelta.trace import AgentTrace, NodeType

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
        has_regression: Whether a behavioral fork / path divergence was detected.
        path_fingerprint_a / path_fingerprint_b: Full-path ids when scored.
        answer_fingerprint_a / answer_fingerprint_b: END-only ids when scored.
    """

    ok: bool
    verdict: str
    reason: str
    exit_code: int
    score: Any = None  # RegressionScore | None — typed loosely to avoid eager import
    run_id_a: str | None = None
    run_id_b: str | None = None
    has_regression: bool = False
    path_fingerprint_a: str | None = None
    path_fingerprint_b: str | None = None
    answer_fingerprint_a: str | None = None
    answer_fingerprint_b: str | None = None

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
            "path_fingerprint_a": self.path_fingerprint_a,
            "path_fingerprint_b": self.path_fingerprint_b,
            "answer_fingerprint_a": self.answer_fingerprint_a,
            "answer_fingerprint_b": self.answer_fingerprint_b,
            "score": None,
        }
        if self.score is not None:
            payload["score"] = asdict(self.score)
        return payload


def path_fingerprint(trace: AgentTrace) -> str:
    """Content-addressed id of the full behavioral path (every node).

    This is the load-bearing identity for WRITER-NOT-READER. Readers that only
    compare :func:`answer_fingerprint` (END node) collapse intermediate tool
    and reasoning changes into a false match — the farm "cache key" lesson.
    """
    h = hashlib.sha256()
    for node in trace.nodes:
        h.update(node.node_type.value.encode())
        h.update(b"\0")
        h.update(node.content.encode())
        h.update(b"\0")
    return h.hexdigest()[:32]


def answer_fingerprint(trace: AgentTrace) -> str:
    """END-node-only fingerprint — *insufficient* for behavioral gates.

    Exposed so tests can prove the collapse trap: same answer, different path.
    Never use this alone to decide PASS.
    """
    h = hashlib.sha256()
    ends = [n for n in trace.nodes if n.node_type == NodeType.END]
    if not ends:
        return h.hexdigest()[:32]
    for node in ends:
        h.update(node.content.encode())
        h.update(b"\0")
    return h.hexdigest()[:32]


def _fail_loud(
    reason: str,
    run_id_a: str | None = None,
    run_id_b: str | None = None,
    *,
    path_a: str | None = None,
    path_b: str | None = None,
    answer_a: str | None = None,
    answer_b: str | None = None,
) -> GateOutcome:
    return GateOutcome(
        ok=False,
        verdict="FAIL_LOUD",
        reason=reason,
        exit_code=2,
        score=None,
        run_id_a=run_id_a,
        run_id_b=run_id_b,
        has_regression=True,
        path_fingerprint_a=path_a,
        path_fingerprint_b=path_b,
        answer_fingerprint_a=answer_a,
        answer_fingerprint_b=answer_b,
    )


def _load_trace(source: AgentTrace | str | Path) -> AgentTrace:
    if isinstance(source, AgentTrace):
        return source
    path = Path(source)
    if not path.is_file():
        raise ClosedLoopError(f"trace file not found: {path}")
    return AgentTrace.load(path)


def gate_from_disk(
    baseline_path: str | Path,
    candidate_path: str | Path,
    **kwargs: Any,
) -> GateOutcome:
    """Gate two traces by **always reloading from disk** (reader path).

    Refuses in-memory :class:`AgentTrace` objects so callers cannot accidentally
    unit-test only the writer. Paths must exist as files.
    """
    b_path = Path(baseline_path)
    c_path = Path(candidate_path)
    if not b_path.is_file():
        return _fail_loud(f"trace file not found: {b_path}")
    if not c_path.is_file():
        return _fail_loud(f"trace file not found: {c_path}")
    # Pass Path objects so gate_traces reloads via AgentTrace.load (reader).
    return gate_traces(b_path, c_path, **kwargs)


def e2e_reader_after_write(
    baseline: AgentTrace,
    candidate: AgentTrace,
    work_dir: str | Path,
    *,
    baseline_name: str = "baseline.jsonl",
    candidate_name: str = "candidate.jsonl",
    **kwargs: Any,
) -> GateOutcome:
    """Write both traces, then gate **only** via disk reload.

    WRITER-NOT-READER e2e control: a content-correct writer is not enough —
    the reader that CI uses must observe the bytes on disk. Returns the
    :class:`GateOutcome` of that reader path.
    """
    if not isinstance(baseline, AgentTrace) or not isinstance(candidate, AgentTrace):
        raise TypeError("e2e_reader_after_write requires in-memory AgentTrace writers")
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    b_path = root / baseline_name
    c_path = root / candidate_name
    baseline.save(b_path)
    candidate.save(c_path)
    return gate_from_disk(b_path, c_path, **kwargs)


def e2e_content_swap_rejudges(
    baseline: AgentTrace,
    candidate_before: AgentTrace,
    candidate_after: AgentTrace,
    work_dir: str | Path,
    **kwargs: Any,
) -> tuple[GateOutcome, GateOutcome]:
    """Content-swap e2e: overwrite candidate on disk; gate must re-judge.

    Farm lesson (footage content-id / cache keys): fixing the writer key without
    tracing readers fails when a later step collapses identity. A real content
    swap must change the *reader* outcome.

    Returns:
        ``(before_outcome, after_outcome)`` — both from :func:`gate_from_disk`.
    """
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    b_path = root / "baseline.jsonl"
    c_path = root / "candidate.jsonl"
    baseline.save(b_path)
    candidate_before.save(c_path)
    before = gate_from_disk(b_path, c_path, **kwargs)
    # Writer overwrites candidate file (same path / cache key).
    candidate_after.save(c_path)
    after = gate_from_disk(b_path, c_path, **kwargs)
    return before, after


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
    * **WRITER-NOT-READER:** path fingerprint diverges while final answer matches
      → never silent ``PASS`` (answer-only collapse refused).

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
    path_a = path_fingerprint(trace_a)
    path_b = path_fingerprint(trace_b)
    ans_a = answer_fingerprint(trace_a)
    ans_b = answer_fingerprint(trace_b)
    path_diverged = path_a != path_b
    answer_same = ans_a == ans_b

    if len(trace_a) == 0:
        return _fail_loud(
            "empty baseline trace — nothing to gate against",
            run_a,
            run_b,
            path_a=path_a,
            path_b=path_b,
            answer_a=ans_a,
            answer_b=ans_b,
        )
    if len(trace_b) == 0:
        return _fail_loud(
            "empty candidate trace — empty output is not a pass",
            run_a,
            run_b,
            path_a=path_a,
            path_b=path_b,
            answer_a=ans_a,
            answer_b=ans_b,
        )

    try:
        from agentdelta.diff import diff_traces
        from agentdelta.score import compute_score
    except Exception as exc:  # noqa: BLE001
        return _fail_loud(
            f"scoring stack unavailable: {exc.__class__.__name__}: {exc}",
            run_a,
            run_b,
            path_a=path_a,
            path_b=path_b,
            answer_a=ans_a,
            answer_b=ans_b,
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
            path_a=path_a,
            path_b=path_b,
            answer_a=ans_a,
            answer_b=ans_b,
        )

    if not diff.steps:
        return _fail_loud(
            "diff produced zero aligned steps despite non-empty traces",
            run_a,
            run_b,
            path_a=path_a,
            path_b=path_b,
            answer_a=ans_a,
            answer_b=ans_b,
        )

    required = {"total_steps", "matched", "changed", "added", "removed", "similarity_pct"}
    if not required.issubset(diff.summary.keys()):
        missing = sorted(required - set(diff.summary.keys()))
        return _fail_loud(
            f"diff summary missing keys: {missing}",
            run_a,
            run_b,
            path_a=path_a,
            path_b=path_b,
            answer_a=ans_a,
            answer_b=ans_b,
        )

    score = compute_score(
        diff,
        pass_threshold=pass_threshold,
        warn_threshold=warn_threshold,
    )

    # Path identity is the honest regression signal when aligner marks tools as
    # add/remove without a fork_point (tool swap, same final answer).
    has_reg = bool(diff.has_regression or path_diverged)

    if score.verdict == "PASS":
        ok, exit_code = True, 0
        verdict = "PASS"
    elif score.verdict == "WARN":
        ok, exit_code = (True, 0) if warn_is_ok else (False, 1)
        verdict = "WARN"
    else:
        ok, exit_code = False, 1
        verdict = score.verdict

    reason = (
        f"verdict={verdict} overall={score.overall:.1f} "
        f"has_regression={has_reg}"
    )
    if diff.fork_point is not None:
        reason += f" fork_at=a{diff.fork_point.step_a}/b{diff.fork_point.step_b}"

    # WRITER-NOT-READER: answer-only equality with path divergence is never PASS.
    if path_diverged and answer_same and verdict == "PASS":
        ok, exit_code = False, 1
        verdict = "FAIL"
        reason = (
            "WRITER-NOT-READER: path fingerprint diverged while final answer matches "
            f"— refusing answer-only PASS (path_a={path_a[:12]}… path_b={path_b[:12]}…)"
        )
        has_reg = True

    return GateOutcome(
        ok=ok,
        verdict=verdict,
        reason=reason,
        exit_code=exit_code,
        score=score,
        run_id_a=run_a,
        run_id_b=run_b,
        has_regression=has_reg,
        path_fingerprint_a=path_a,
        path_fingerprint_b=path_b,
        answer_fingerprint_a=ans_a,
        answer_fingerprint_b=ans_b,
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
