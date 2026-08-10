"""SKILLPROX — closed-loop skill evolution gate (arXiv 2608.07449).

Public case: *SkillProx: Self-Evolving Agent Skills via Proximal Textual
Gradient Descent*. Agents accumulate procedural skills as textual artifacts
and refine them via diagnosis-driven edits. Existing frameworks lack explicit
**diagnosis→outcome feedback** and treat **deletion** as a generic edit instead
of utility-aware consolidation. SkillProx couples forward diagnostic evolution
with rollback on regression and backward utility-aware refinement.

Product role in agentdelta (BITTER-TOOL / TRAJDEBUG twin):
  Gate skill-library mutations so CI/runtimes refuse "evolved" skills that
  never measured post-edit outcomes, never diagnosed failures, or regressed
  without rollback.

Non-Ornament:
  Call ``gate_skill_evolution`` before accepting skill-library updates as
  production context. Pair with ``gate_traces`` for trajectory regression.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agentdelta.closed_loop import ClosedLoopError, GateOutcome

# Edit kinds that change skill content.
MUTATING_KINDS: frozenset[str] = frozenset(
    {
        "refine",
        "edit",
        "update",
        "create",
        "delete",
        "remove",
        "prune",
        "consolidate",
    }
)

DELETE_KINDS: frozenset[str] = frozenset({"delete", "remove", "prune", "consolidate"})


@dataclass(frozen=True)
class SkillEdit:
    """One diagnosis-driven skill library mutation.

    Attributes:
        edit_id: Stable id for this edit.
        skill_id: Skill artifact being mutated.
        edit_kind: ``create`` / ``refine`` / ``delete`` / …
        diagnosis: Failure diagnosis that motivated the edit (forward stage).
        baseline_metric: Task success (or loss inverse) before the edit.
        outcome_metric: Measured metric after re-execution on same batch.
        rolled_back: True if regression triggered proximal rollback.
        utility_score: Optional utility for delete/consolidate (backward stage).
        from_version / to_version: Optional version markers.
        meta: Extra fields.
    """

    edit_id: str
    skill_id: str
    edit_kind: str = "refine"
    diagnosis: str = ""
    baseline_metric: float | None = None
    outcome_metric: float | None = None
    rolled_back: bool = False
    utility_score: float | None = None
    from_version: int | None = None
    to_version: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "skill_id": self.skill_id,
            "edit_kind": self.edit_kind,
            "diagnosis": self.diagnosis,
            "baseline_metric": self.baseline_metric,
            "outcome_metric": self.outcome_metric,
            "rolled_back": self.rolled_back,
            "utility_score": self.utility_score,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class SkillEvolutionReport:
    """Summary of skill-evolution closed-loop health."""

    edit_count: int
    skill_ids: tuple[str, ...]
    missing_diagnosis: tuple[str, ...]
    missing_outcome: tuple[str, ...]
    unreverted_regressions: tuple[str, ...]
    unjustified_deletes: tuple[str, ...]
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return not (
            self.missing_diagnosis
            or self.missing_outcome
            or self.unreverted_regressions
            or self.unjustified_deletes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_count": self.edit_count,
            "skill_ids": list(self.skill_ids),
            "missing_diagnosis": list(self.missing_diagnosis),
            "missing_outcome": list(self.missing_outcome),
            "unreverted_regressions": list(self.unreverted_regressions),
            "unjustified_deletes": list(self.unjustified_deletes),
            "healthy": self.healthy,
            "details": dict(self.details),
        }


def _canon_kind(kind: str) -> str:
    return (kind or "").strip().lower().replace(" ", "_").replace("-", "_")


def _as_edit(item: Any, index: int = 0) -> SkillEdit:
    if isinstance(item, SkillEdit):
        return item
    if not isinstance(item, dict):
        raise TypeError(f"edit must be SkillEdit or dict, got {type(item)!r}")
    eid = str(item.get("edit_id") or item.get("id") or f"edit_{index}").strip()
    sid = str(item.get("skill_id") or item.get("skill") or item.get("name") or "").strip()
    if not sid:
        raise ValueError(f"edit {eid!r} missing skill_id")
    kind = _canon_kind(str(item.get("edit_kind") or item.get("kind") or "refine"))
    base = item.get("baseline_metric", item.get("before", item.get("baseline")))
    out = item.get("outcome_metric", item.get("after", item.get("outcome")))
    util = item.get("utility_score", item.get("utility"))
    return SkillEdit(
        edit_id=eid,
        skill_id=sid,
        edit_kind=kind,
        diagnosis=str(item.get("diagnosis") or item.get("reason") or item.get("rationale") or ""),
        baseline_metric=float(base) if base is not None else None,
        outcome_metric=float(out) if out is not None else None,
        rolled_back=bool(item.get("rolled_back") or item.get("rollback") or False),
        utility_score=float(util) if util is not None else None,
        from_version=item.get("from_version"),
        to_version=item.get("to_version"),
        meta=dict(item.get("meta") or {}) if isinstance(item.get("meta"), dict) else {},
    )


def _is_regression(baseline: float | None, outcome: float | None) -> bool:
    """Higher metric = better (success rate). Regression when outcome < baseline."""
    if baseline is None or outcome is None:
        return False
    return float(outcome) < float(baseline) - 1e-12


def analyze_skill_evolution(
    edits: Sequence[Any] | None,
    *,
    min_utility_for_delete: float = 0.0,
    require_diagnosis_for: Sequence[str] | None = None,
) -> SkillEvolutionReport:
    """Analyze skill edits for SkillProx closed-loop properties (no gate)."""
    parsed = [_as_edit(e, i) for i, e in enumerate(edits or [])]
    need_diag = {
        _canon_kind(k)
        for k in (require_diagnosis_for or sorted(MUTATING_KINDS - {"create"}))
    }

    missing_diag: list[str] = []
    missing_out: list[str] = []
    regressions: list[str] = []
    bad_del: list[str] = []
    skills: list[str] = []

    for e in parsed:
        skills.append(e.skill_id)
        kind = _canon_kind(e.edit_kind)
        if kind in need_diag and not (e.diagnosis or "").strip():
            missing_diag.append(e.edit_id)
        if kind in DELETE_KINDS:
            # utility-aware deletion: need utility_score or measured outcome
            if e.utility_score is None and e.outcome_metric is None:
                bad_del.append(e.edit_id)
        elif kind in MUTATING_KINDS and kind != "create" and e.outcome_metric is None:
            # forward re-exec outcome required for refine/update (not delete)
            missing_out.append(e.edit_id)
        if _is_regression(e.baseline_metric, e.outcome_metric) and not e.rolled_back:
            regressions.append(e.edit_id)
        _ = min_utility_for_delete  # reserved for future utility thresholds

    return SkillEvolutionReport(
        edit_count=len(parsed),
        skill_ids=tuple(dict.fromkeys(skills)),
        missing_diagnosis=tuple(missing_diag),
        missing_outcome=tuple(missing_out),
        unreverted_regressions=tuple(regressions),
        unjustified_deletes=tuple(bad_del),
        details={"min_utility_for_delete": min_utility_for_delete},
    )


def gate_skill_evolution(
    edits: Sequence[Any] | None,
    *,
    claim_evolved: bool = False,
    require_edits: bool = True,
    require_diagnosis: bool = True,
    require_outcome_feedback: bool = True,
    refuse_regression_without_rollback: bool = True,
    refuse_unjustified_delete: bool = True,
    min_utility_for_delete: float = 0.0,
) -> GateOutcome:
    """Refuse skill-library evolution without SkillProx closed-loop properties.

    Public case: arXiv 2608.07449 SkillProx — diagnosis-driven forward edits
    must re-measure outcomes and roll back regressions; deletion is
    utility-aware consolidation, not a silent generic edit.

    Rules:

    1. ``claim_evolved`` with zero edits → **FAIL_LOUD**
    2. Empty inventory when required → **FAIL_LOUD**
    3. Mutating edit without diagnosis → **FAIL**
    4. Mutating edit without post-edit outcome metric → **FAIL**
    5. Outcome worse than baseline without ``rolled_back`` → **FAIL**
    6. Delete/prune without utility or outcome signal → **FAIL**
    7. Closed-loop healthy edits → **PASS**
    """
    if not edits:
        if claim_evolved or require_edits:
            return GateOutcome(
                ok=False,
                verdict="FAIL_LOUD",
                reason=(
                    "SKILLPROX: empty skill-edit inventory — cannot claim "
                    "self-evolving skills without diagnosis-driven mutations "
                    f"(claim_evolved={claim_evolved}; arXiv 2608.07449)"
                ),
                exit_code=2,
                human_required=True,
            )
        return GateOutcome(
            ok=True,
            verdict="PASS",
            reason="SKILLPROX: no edits required",
            exit_code=0,
        )

    try:
        report = analyze_skill_evolution(
            edits,
            min_utility_for_delete=min_utility_for_delete,
        )
    except (TypeError, ValueError) as exc:
        return GateOutcome(
            ok=False,
            verdict="FAIL_LOUD",
            reason=f"SKILLPROX: invalid skill edits: {exc}",
            exit_code=2,
            human_required=True,
        )

    n = report.edit_count

    if require_diagnosis and report.missing_diagnosis:
        return GateOutcome(
            ok=False,
            verdict="FAIL",
            reason=(
                f"SKILLPROX: {len(report.missing_diagnosis)} edit(s) lack "
                f"failure diagnosis {list(report.missing_diagnosis)[:8]} — "
                f"refuse undiagnosed skill mutation (arXiv 2608.07449 forward stage)"
            ),
            exit_code=1,
            human_required=True,
            error_step_count=len(report.missing_diagnosis),
            has_regression=False,
        )

    if require_outcome_feedback and report.missing_outcome:
        return GateOutcome(
            ok=False,
            verdict="FAIL",
            reason=(
                f"SKILLPROX: {len(report.missing_outcome)} edit(s) lack "
                f"post-edit outcome_metric {list(report.missing_outcome)[:8]} — "
                f"refuse evolution without diagnosis→outcome feedback loop"
            ),
            exit_code=1,
            human_required=True,
            error_step_count=len(report.missing_outcome),
        )

    if refuse_regression_without_rollback and report.unreverted_regressions:
        return GateOutcome(
            ok=False,
            verdict="FAIL",
            reason=(
                f"SKILLPROX: {len(report.unreverted_regressions)} edit(s) "
                f"regressed vs baseline without rollback "
                f"{list(report.unreverted_regressions)[:8]} — refuse "
                f"non-proximal skill update (SkillProx forward rollback)"
            ),
            exit_code=1,
            human_required=True,
            has_regression=True,
            error_step_count=len(report.unreverted_regressions),
        )

    if refuse_unjustified_delete and report.unjustified_deletes:
        return GateOutcome(
            ok=False,
            verdict="FAIL",
            reason=(
                f"SKILLPROX: {len(report.unjustified_deletes)} delete/prune "
                f"edit(s) without utility_score or outcome "
                f"{list(report.unjustified_deletes)[:8]} — refuse generic "
                f"deletion; SkillProx treats delete as utility-aware consolidation"
            ),
            exit_code=1,
            human_required=True,
            error_step_count=len(report.unjustified_deletes),
        )

    return GateOutcome(
        ok=True,
        verdict="PASS",
        reason=(
            f"SKILLPROX ok: edits={n} skills={len(report.skill_ids)} "
            f"diagnosis+outcome closed-loop healthy claim_evolved={claim_evolved}"
        ),
        exit_code=0,
        error_step_count=0,
        has_regression=False,
        human_required=False,
    )


def assert_skill_evolution_ok(
    edits: Sequence[Any] | None,
    **kwargs: Any,
) -> GateOutcome:
    """Raise :class:`ClosedLoopError` unless :func:`gate_skill_evolution` is ok."""
    outcome = gate_skill_evolution(edits, **kwargs)
    if not outcome.ok:
        raise ClosedLoopError(f"{outcome.verdict}: {outcome.reason}")
    return outcome
