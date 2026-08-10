"""SKILLPROX — closed-loop skill evolution (arXiv 2608.07449).

Public case (Track B 20260810T121232Z):
  SkillProx: Self-Evolving Agent Skills via Proximal Textual Gradient Descent.
  Skill edits must carry diagnosis→outcome feedback and roll back regressions;
  deletion is utility-aware consolidation, not a silent generic edit.
"""

from __future__ import annotations

import pytest

from agentdelta.closed_loop import ClosedLoopError
from agentdelta.skill_evolution import (
    SkillEdit,
    analyze_skill_evolution,
    assert_skill_evolution_ok,
    gate_skill_evolution,
)


def test_empty_claim_evolved_fails_loud() -> None:
    out = gate_skill_evolution([], claim_evolved=True)
    assert out.verdict == "FAIL_LOUD"
    assert out.exit_code == 2
    assert out.human_required is True
    assert "SKILLPROX" in out.reason


def test_refine_without_diagnosis_fails() -> None:
    edits = [
        SkillEdit(
            edit_id="e1",
            skill_id="web_nav",
            edit_kind="refine",
            diagnosis="",
            baseline_metric=0.5,
            outcome_metric=0.7,
        )
    ]
    out = gate_skill_evolution(edits, claim_evolved=True)
    assert out.ok is False
    assert out.verdict == "FAIL"
    assert "diagnosis" in out.reason.lower()


def test_refine_without_outcome_fails() -> None:
    edits = [
        {
            "edit_id": "e2",
            "skill_id": "web_nav",
            "kind": "refine",
            "diagnosis": "missed confirm button",
            "baseline_metric": 0.4,
        }
    ]
    out = gate_skill_evolution(edits)
    assert out.ok is False
    assert out.verdict == "FAIL"
    assert "outcome" in out.reason.lower()


def test_regression_without_rollback_fails() -> None:
    edits = [
        SkillEdit(
            edit_id="e3",
            skill_id="checkout",
            edit_kind="refine",
            diagnosis="timeout on pay",
            baseline_metric=0.8,
            outcome_metric=0.3,
            rolled_back=False,
        )
    ]
    out = gate_skill_evolution(edits, claim_evolved=True)
    assert out.ok is False
    assert out.verdict == "FAIL"
    assert out.has_regression is True
    assert "rollback" in out.reason.lower() or "regress" in out.reason.lower()


def test_regression_with_rollback_passes() -> None:
    edits = [
        SkillEdit(
            edit_id="e4",
            skill_id="checkout",
            edit_kind="refine",
            diagnosis="timeout on pay",
            baseline_metric=0.8,
            outcome_metric=0.3,
            rolled_back=True,
        )
    ]
    out = gate_skill_evolution(edits, claim_evolved=True)
    assert out.ok is True
    assert out.verdict == "PASS"


def test_delete_without_utility_fails() -> None:
    edits = [
        SkillEdit(
            edit_id="e5",
            skill_id="legacy_skill",
            edit_kind="delete",
            diagnosis="unused after consolidation",
            # no utility_score, no outcome_metric
        )
    ]
    out = gate_skill_evolution(edits)
    assert out.ok is False
    assert out.verdict == "FAIL"
    assert "delete" in out.reason.lower() or "utility" in out.reason.lower()


def test_healthy_forward_backward_passes() -> None:
    edits = [
        SkillEdit(
            edit_id="e6",
            skill_id="form_fill",
            edit_kind="refine",
            diagnosis="missed required field validation",
            baseline_metric=0.55,
            outcome_metric=0.82,
            rolled_back=False,
        ),
        SkillEdit(
            edit_id="e7",
            skill_id="old_helper",
            edit_kind="delete",
            diagnosis="superseded by form_fill v2",
            utility_score=0.05,
            outcome_metric=0.82,
        ),
    ]
    out = gate_skill_evolution(edits, claim_evolved=True)
    assert out.ok is True
    assert out.verdict == "PASS"
    assert out.exit_code == 0
    payload = out.to_dict()
    assert payload["ok"] is True


def test_analyze_skill_evolution_report() -> None:
    report = analyze_skill_evolution(
        [
            {
                "edit_id": "a",
                "skill_id": "s",
                "edit_kind": "refine",
                "diagnosis": "",
                "outcome_metric": 0.1,
            }
        ]
    )
    assert "a" in report.missing_diagnosis
    assert report.healthy is False
    assert report.to_dict()["edit_count"] == 1


def test_assert_raises_and_passes() -> None:
    with pytest.raises(ClosedLoopError):
        assert_skill_evolution_ok([], claim_evolved=True)
    out = assert_skill_evolution_ok(
        [
            SkillEdit(
                edit_id="ok",
                skill_id="s",
                edit_kind="create",
                diagnosis="bootstrap",
            )
        ],
        claim_evolved=True,
    )
    assert out.ok is True


def test_invalid_payload_fails_loud() -> None:
    out = gate_skill_evolution([{"edit_id": "x"}])  # missing skill_id
    assert out.verdict == "FAIL_LOUD"
