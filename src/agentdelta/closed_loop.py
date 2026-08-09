"""Closed-loop reader/gate for agentdelta (L1 + WRITER-NOT-READER + TRAJDEBUG).

Who reads the output?
  CI jobs, eagle-eyes ``dogfood_verify``, or any gate that must *fail loudly*
  when traces are empty, malformed, or behaviorally regress; trajectory
  scanners that must refuse claimed success after intermediate errors.

What outcome changes?
  Returns a structured ``GateOutcome`` with ``exit_code`` suitable for
  ``sys.exit`` - empty/wrong output is never a silent PASS.
  Intermediate tool/LLM failures with a clean END claim → FAIL (TRAJDEBUG).

WRITER-NOT-READER (farm lesson):
  Fixes that only touch the *writer* (save path, cache key, in-memory object)
  while leaving *readers* on a collapsed key (final-answer-only, stale path,
  never reloading disk) are incomplete. This module exposes:

  * :func:`path_fingerprint` / :func:`answer_fingerprint` - full path vs END-only
  * :func:`gate_from_disk` - reader always reloads JSONL from disk
  * :func:`e2e_reader_after_write` - write then gate via disk only
  * answer-only collapse refusal inside :func:`gate_traces`

TRAJDEBUG (public arXiv 2608.06346):
  Long-horizon agents hide critical intermediate failures when only the final
  answer is inspected. :func:`gate_error_lifecycle` walks the error lifecycle
  and refuses claimed success with unrecovered intermediate errors.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agentdelta.trace import AgentTrace, NodeType, TraceNode

# Lazy imports for diff/score keep the empty-trace FAIL_LOUD path free of
# heavy deps (numpy / sentence-transformers) so eagle-eyes dogfood can call
# the gate without a full runtime install.


class ClosedLoopError(ValueError):
    """Raised when the gate refuses to score empty or unusable traces."""


@dataclass(frozen=True)
class GateOutcome:
    """Result of a closed-loop read of agent traces or error lifecycle.

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
        error_step_count: Failed intermediate steps (TRAJDEBUG).
        critical_step: First unrecovered failure step number when present.
        human_required: True when intermediate errors need human review.
    """

    ok: bool
    verdict: str
    reason: str
    exit_code: int
    score: Any = None  # RegressionScore | None - typed loosely to avoid eager import
    run_id_a: str | None = None
    run_id_b: str | None = None
    has_regression: bool = False
    path_fingerprint_a: str | None = None
    path_fingerprint_b: str | None = None
    answer_fingerprint_a: str | None = None
    answer_fingerprint_b: str | None = None
    error_step_count: int = 0
    critical_step: int | None = None
    human_required: bool = False

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
            "error_step_count": self.error_step_count,
            "critical_step": self.critical_step,
            "human_required": self.human_required,
            "score": None,
        }
        if self.score is not None:
            payload["score"] = asdict(self.score)
        return payload


def path_fingerprint(trace: AgentTrace) -> str:
    """Content-addressed id of the full behavioral path (every node).

    This is the load-bearing identity for WRITER-NOT-READER. Readers that only
    compare :func:`answer_fingerprint` (END node) collapse intermediate tool
    and reasoning changes into a false match - the farm "cache key" lesson.
    """
    h = hashlib.sha256()
    for node in trace.nodes:
        h.update(node.node_type.value.encode())
        h.update(b"\0")
        h.update(node.content.encode())
        h.update(b"\0")
    return h.hexdigest()[:32]


def answer_fingerprint(trace: AgentTrace) -> str:
    """END-node-only fingerprint - *insufficient* for behavioral gates.

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

    WRITER-NOT-READER e2e control: a content-correct writer is not enough -
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
        ``(before_outcome, after_outcome)`` - both from :func:`gate_from_disk`.
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
        :class:`GateOutcome` - callers should ``sys.exit(outcome.exit_code)``.
    """
    try:
        trace_a = _load_trace(baseline)
        trace_b = _load_trace(candidate)
    except ClosedLoopError as exc:
        return _fail_loud(str(exc))
    except Exception as exc:
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
            "empty baseline trace - nothing to gate against",
            run_a,
            run_b,
            path_a=path_a,
            path_b=path_b,
            answer_a=ans_a,
            answer_b=ans_b,
        )
    if len(trace_b) == 0:
        return _fail_loud(
            "empty candidate trace - empty output is not a pass",
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
    except Exception as exc:
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
    except Exception as exc:
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

    reason = f"verdict={verdict} overall={score.overall:.1f} has_regression={has_reg}"
    if diff.fork_point is not None:
        reason += f" fork_at=a{diff.fork_point.step_a}/b{diff.fork_point.step_b}"

    # WRITER-NOT-READER: answer-only equality with path divergence is never PASS.
    if path_diverged and answer_same and verdict == "PASS":
        ok, exit_code = False, 1
        verdict = "FAIL"
        reason = (
            "WRITER-NOT-READER: path fingerprint diverged while final answer matches "
            f"- refusing answer-only PASS (path_a={path_a[:12]}… path_b={path_b[:12]}…)"
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


# ---------------------------------------------------------------------------
# TRAJDEBUG - error lifecycle on long-horizon trajectories
# ---------------------------------------------------------------------------

_ERROR_CONTENT_RE = re.compile(
    r"(?i)\b(error|exception|traceback|failed|failure|timeout|refused)\b"
)
_FAIL_STATUSES = frozenset(
    {"error", "fail", "failed", "failure", "exception", "timeout", "critical"}
)
_OK_STATUSES = frozenset({"ok", "success", "pass", "passed", "done", "completed"})


@dataclass(frozen=True)
class TrajectoryStep:
    """Minimal step record for TRAJDEBUG without a full :class:`AgentTrace`."""

    step: int
    name: str
    status: str = "ok"  # ok | error | fail | skip | …
    message: str = ""
    recovered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "recovered": self.recovered,
        }


@dataclass(frozen=True)
class ErrorLifecycle:
    """Summary of intermediate failures along a trajectory (TRAJDEBUG)."""

    step_count: int
    error_steps: tuple[int, ...]
    critical_step: int | None
    claimed_success: bool
    unrecovered_count: int
    error_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "error_steps": list(self.error_steps),
            "critical_step": self.critical_step,
            "claimed_success": self.claimed_success,
            "unrecovered_count": self.unrecovered_count,
            "error_names": list(self.error_names),
        }


def node_is_failed(node: TraceNode) -> bool:
    """True if a trace node represents an intermediate failure.

    Checks ``metadata['status'|'ok'|'success'|'error']`` and content markers
    (ERROR/Exception/failed). END nodes with empty failure metadata are not
    automatically failures.
    """
    md = node.metadata or {}
    status = str(md.get("status", "")).strip().lower()
    if status in _FAIL_STATUSES:
        return True
    if status in _OK_STATUSES:
        return False
    if "ok" in md and md["ok"] is False:
        return True
    if "success" in md and md["success"] is False:
        return True
    err = md.get("error")
    if err not in (None, "", False, 0):
        return True
    if node.node_type == NodeType.END:
        # Only fail END if explicitly marked
        return False
    content = node.content or ""
    if content.strip().lower().startswith("error"):
        return True
    return bool(
        _ERROR_CONTENT_RE.search(content)
        and (
            "traceback" in content.lower()
            or "exception" in content.lower()
            or md.get("failed") is True
        )
    )


def _steps_from_trace(trace: AgentTrace) -> list[TrajectoryStep]:
    out: list[TrajectoryStep] = []
    for node in trace.nodes:
        failed = node_is_failed(node)
        name = f"{node.node_type.value}:{node.content[:40]}"
        out.append(
            TrajectoryStep(
                step=node.step,
                name=name,
                status="error" if failed else "ok",
                message=(node.content or "")[:200],
                recovered=False,
            )
        )
    return out


def _normalize_steps(
    source: AgentTrace | Sequence[TrajectoryStep] | Sequence[dict[str, Any]],
) -> list[TrajectoryStep]:
    if isinstance(source, AgentTrace):
        return _steps_from_trace(source)
    steps: list[TrajectoryStep] = []
    for i, item in enumerate(source):
        if isinstance(item, TrajectoryStep):
            steps.append(item)
            continue
        if isinstance(item, dict):
            st = int(item.get("step", i + 1))
            name = str(item.get("name") or item.get("action") or f"step_{st}")
            status = str(item.get("status", "ok")).strip().lower()
            msg = str(item.get("message") or item.get("error") or "")
            recovered = bool(item.get("recovered", False))
            if item.get("ok") is False or item.get("success") is False:
                status = "error"
            steps.append(
                TrajectoryStep(
                    step=st,
                    name=name,
                    status=status,
                    message=msg,
                    recovered=recovered,
                )
            )
            continue
        raise TypeError(f"unsupported trajectory step type: {type(item)!r}")
    return steps


def analyze_error_lifecycle(
    source: AgentTrace | Sequence[TrajectoryStep] | Sequence[dict[str, Any]],
    *,
    claimed_success: bool | None = None,
) -> ErrorLifecycle:
    """Walk a trajectory and locate unrecovered intermediate failures.

    * If ``claimed_success`` is None and source is :class:`AgentTrace`, success
      is inferred from presence of an END node without failure metadata.
    * A step is unrecovered when status is fail/error and ``recovered`` is False
      (dict/TrajectoryStep) or, for traces, when no later OK TOOL_RETURN follows
      a failed TOOL_CALL with the same tool name hint.
    """
    steps = _normalize_steps(source)
    if not steps:
        return ErrorLifecycle(
            step_count=0,
            error_steps=(),
            critical_step=None,
            claimed_success=False,
            unrecovered_count=0,
        )

    if claimed_success is None:
        if isinstance(source, AgentTrace):
            ends = [n for n in source.nodes if n.node_type == NodeType.END]
            claimed_success = bool(ends) and not any(node_is_failed(n) for n in ends)
        else:
            # last step ok and no explicit failure claim
            last = steps[-1]
            claimed_success = last.status in _OK_STATUSES

    error_steps: list[int] = []
    error_names: list[str] = []
    unrecovered: list[int] = []
    for s in steps:
        if s.status in _FAIL_STATUSES or (
            s.message and s.status not in _OK_STATUSES and s.status == "error"
        ):
            error_steps.append(s.step)
            error_names.append(s.name)
            if not s.recovered:
                unrecovered.append(s.step)
        elif s.status not in _OK_STATUSES and s.status not in {"skip", "skipped", "info"}:
            # unknown non-ok → treat as error
            if s.status:
                error_steps.append(s.step)
                error_names.append(s.name)
                if not s.recovered:
                    unrecovered.append(s.step)

    critical = unrecovered[0] if unrecovered else (error_steps[0] if error_steps else None)
    return ErrorLifecycle(
        step_count=len(steps),
        error_steps=tuple(error_steps),
        critical_step=critical,
        claimed_success=bool(claimed_success),
        unrecovered_count=len(unrecovered),
        error_names=tuple(error_names[:20]),
    )


def gate_error_lifecycle(
    source: AgentTrace | Sequence[TrajectoryStep] | Sequence[dict[str, Any]],
    *,
    claimed_success: bool | None = None,
    refuse_claimed_success_with_errors: bool = True,
    max_unrecovered: int = 0,
) -> GateOutcome:
    """Refuse trajectories that hide critical intermediate failures (TRAJDEBUG).

    Public case: arXiv 2608.06346 *TRAJDEBUG: Tracing Error Lifecycle to
    Identify Critical Failures in Long-Horizon LLM Agents*. Scoring only the
    final answer misses where the run first went wrong.

    Rules:

    * Empty trajectory → **FAIL_LOUD**
    * ``unrecovered_count > max_unrecovered`` → **FAIL**
    * Claimed success + any unrecovered error (default) → **FAIL**
    * Clean path → **PASS**

    Args:
        source: :class:`AgentTrace` or sequence of step dicts / TrajectoryStep.
        claimed_success: Override success claim (None = infer).
        refuse_claimed_success_with_errors: Silent-success trap (default True).
        max_unrecovered: Allowed unrecovered errors (default 0).
    """
    steps = _normalize_steps(source)
    if len(steps) == 0:
        return GateOutcome(
            ok=False,
            verdict="FAIL_LOUD",
            reason=(
                "TRAJDEBUG: empty trajectory - no steps to scan for error lifecycle "
                "(write-only ornament / no intermediate evidence)"
            ),
            exit_code=2,
            error_step_count=0,
            critical_step=None,
            human_required=True,
        )

    life = analyze_error_lifecycle(source, claimed_success=claimed_success)

    if life.unrecovered_count > max_unrecovered:
        return GateOutcome(
            ok=False,
            verdict="FAIL",
            reason=(
                f"TRAJDEBUG: {life.unrecovered_count} unrecovered error step(s) "
                f"(max={max_unrecovered}) critical_step={life.critical_step} "
                f"errors={list(life.error_names)[:5]} - refuse long-horizon continue "
                f"(arXiv 2608.06346 error lifecycle)"
            ),
            exit_code=1,
            error_step_count=len(life.error_steps),
            critical_step=life.critical_step,
            human_required=True,
            run_id_a=getattr(source, "run_id", None) if isinstance(source, AgentTrace) else None,
        )

    if (
        refuse_claimed_success_with_errors
        and life.claimed_success
        and life.error_steps
        and life.unrecovered_count > 0
    ):
        return GateOutcome(
            ok=False,
            verdict="FAIL",
            reason=(
                f"TRAJDEBUG: claimed success with unrecovered intermediate errors "
                f"at steps={list(life.error_steps)[:8]} critical={life.critical_step} "
                f"- final-answer-only scoring hides lifecycle failures"
            ),
            exit_code=1,
            error_step_count=len(life.error_steps),
            critical_step=life.critical_step,
            human_required=True,
        )

    return GateOutcome(
        ok=True,
        verdict="PASS",
        reason=(
            f"TRAJDEBUG ok: steps={life.step_count} errors={len(life.error_steps)} "
            f"unrecovered={life.unrecovered_count} claimed_success={life.claimed_success}"
        ),
        exit_code=0,
        error_step_count=len(life.error_steps),
        critical_step=life.critical_step,
        human_required=False,
    )


def assert_error_lifecycle_ok(
    source: AgentTrace | Sequence[TrajectoryStep] | Sequence[dict[str, Any]],
    **kwargs: Any,
) -> GateOutcome:
    """Raise :class:`ClosedLoopError` unless :func:`gate_error_lifecycle` is ok."""
    outcome = gate_error_lifecycle(source, **kwargs)
    if not outcome.ok:
        raise ClosedLoopError(f"{outcome.verdict}: {outcome.reason}")
    return outcome
