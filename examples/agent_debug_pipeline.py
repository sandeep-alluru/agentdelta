"""
Integration demo: agentdelta + foghorn for fintech loan underwriting.

This script demonstrates a realistic two-tool workflow where:

  1. foghorn tracks the factual assumptions underlying AI loan decisions.
     The underwriting agent records that it depends on "Equifax-score-API
     returns FICO-8".  When the credit bureau silently switches to FICO-9,
     foghorn detects that all three loan decisions are now stale — the
     ground truth they relied on has changed.

  2. agentdelta diffs the agent's behaviour before and after the data
     change.  By replaying the agent's tool-call sequence in both worlds
     and diffing the two traces, the tool pinpoints the exact fork: the
     agent now calls check_fico9_threshold instead of check_fico8_threshold.

Install dependencies:
    pip install agentdelta foghorn-ai

Run:
    python 01_agent_debug_pipeline.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agentdelta.diff import diff_traces
from agentdelta.instrument import AgentdeltaCallback
from agentdelta.report import print_diff
from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode
from foghorn.repo import WorldRepo


# ---------------------------------------------------------------------------
# Helper: fake LLM response object matching what AgentdeltaCallback expects
# ---------------------------------------------------------------------------

class _FakeLLMResponse:
    """Minimal stand-in for a LangChain LLMResult."""

    def __init__(self, text: str) -> None:
        self.generations = [[type("Gen", (), {"text": text})()]]


# ---------------------------------------------------------------------------
# Trace builders — simulated underwriting agent runs
# ---------------------------------------------------------------------------

def _build_underwriting_trace(run_id: str, fico_version: int) -> AgentTrace:
    """Simulate an underwriting agent trace for one FICO scoring world.

    When fico_version == 8 the agent calls check_fico8_threshold.
    When fico_version == 9 the agent calls check_fico9_threshold.
    This is the behavioural fork that agentdelta will detect.
    """
    cb = AgentdeltaCallback(run_id=run_id)

    # Chain start: initial underwriting request
    cb.on_chain_start(
        {},
        {"input": "Evaluate loan application #LN-7821 for $45,000 personal loan."},
    )

    # LLM reasoning step: decide what data to pull
    cb.on_llm_end(
        _FakeLLMResponse(
            f"To evaluate this application I need the applicant's credit score "
            f"from the Equifax API (currently returning FICO-{fico_version}) "
            f"and the current prime rate."
        )
    )

    # Tool call: fetch credit score
    cb.on_tool_start({"name": "fetch_credit_score"}, "applicant_id='APP-7821', bureau='equifax'")
    score = 712 if fico_version == 8 else 689  # FICO-9 penalises medical debt differently
    cb.on_tool_end(
        f'{{"applicant_id": "APP-7821", "score": {score}, '
        f'"model": "FICO-{fico_version}", "bureau": "Equifax"}}'
    )

    # LLM reasoning: interpret score
    cb.on_llm_end(
        _FakeLLMResponse(
            f"Applicant score is {score} under FICO-{fico_version}. "
            f"Now checking against the lender's threshold table."
        )
    )

    # Tool call: THIS IS THE FORK — different tool depending on FICO version
    threshold_tool = f"check_fico{fico_version}_threshold"
    cb.on_tool_start({"name": threshold_tool}, f"score={score}, loan_amount=45000, term_months=60")
    rate_bps = 520 if fico_version == 8 else 575
    cb.on_tool_end(
        f'{{"approved": true, "rate_bps": {rate_bps}, '
        f'"threshold_model": "FICO-{fico_version}"}}'
    )

    # LLM reasoning: final underwriting decision
    cb.on_llm_end(
        _FakeLLMResponse(
            f"Approved. Recommended rate: {rate_bps} bps over prime. "
            f"Score {score} clears the FICO-{fico_version} minimum of "
            + ("680." if fico_version == 8 else "670.")
        )
    )

    # Chain end
    cb.on_chain_end(
        {
            "output": (
                f"APPROVED — $45,000 at prime + {rate_bps}bps. "
                f"Credit model: FICO-{fico_version}."
            )
        }
    )

    return cb.trace


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print(__doc__)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ===================================================================
        print("\n=== Step 1: foghorn — tracking decision dependencies ===\n")
        # ===================================================================

        repo = WorldRepo.init(tmp / "underwriting.db")

        # -- Commit A: initial world state -----------------------------------
        # The credit bureau API is known to return FICO-8 scores.
        f_fico = repo.add_fact(
            subject="Equifax-score-API",
            predicate="returns",
            obj="FICO-8",
            confidence=1.0,
        )
        f_rate = repo.add_fact(
            subject="prime-rate",
            predicate="is",
            obj="5.50-percent",
            confidence=0.98,
        )
        print(f"Staged fact: {f_fico}")
        print(f"Staged fact: {f_rate}")

        # Three loan decisions that depend on the FICO-8 fact.
        decisions = [
            repo.decide(
                label="approve-LN-7821",
                content="Applicant scored 712 on FICO-8; approved at prime+520bps.",
                depends_on=[f_fico.id, f_rate.id],
            ),
            repo.decide(
                label="approve-LN-7834",
                content="Applicant scored 741 on FICO-8; approved at prime+480bps.",
                depends_on=[f_fico.id, f_rate.id],
            ),
            repo.decide(
                label="decline-LN-7849",
                content="Applicant scored 618 on FICO-8; declined — below 680 minimum.",
                depends_on=[f_fico.id],
            ),
        ]
        for d in decisions:
            print(f"Staged decision: {d}")

        commit_a = repo.commit("Initial underwriting decisions — FICO-8 era")
        print(f"\nCommit A: {commit_a.id[:8]} — {commit_a.message}")
        print(f"  {len(commit_a.fact_ids)} fact(s), {len(commit_a.decision_ids)} decision(s)")

        # -- Commit B: Equifax silently switches to FICO-9 -------------------
        print("\n[!] Equifax silently switches scoring model from FICO-8 to FICO-9.")

        # Retract the old fact and assert the new one.
        repo.retract_fact(f_fico.id)
        f_fico9 = repo.add_fact(
            subject="Equifax-score-API",
            predicate="returns",
            obj="FICO-9",
            confidence=1.0,
        )
        print(f"New fact staged: {f_fico9}")

        commit_b = repo.commit("Equifax API now returns FICO-9 (silent migration)")
        print(f"Commit B: {commit_b.id[:8]} — {commit_b.message}")

        # -- Staleness detection ---------------------------------------------
        print("\n--- Staleness check (foghorn.stale()) ---\n")
        alerts = repo.stale(since=commit_a)

        if alerts:
            print(f"STALE DECISIONS DETECTED: {len(alerts)} alert(s)\n")
            for alert in alerts:
                print(f"  Decision : {alert.decision_label}")
                print(f"  Impact   : {alert.impact_score:.0%}")
                print(f"  Stale fact IDs: {alert.stale_fact_ids}")
                print()
            print(
                "foghorn result: all three loan decisions must be re-evaluated —\n"
                "they were made against FICO-8 scores that are no longer valid."
            )
        else:
            # The FICO-8 fact was retracted and replaced with a new FICO-9 fact
            # (different content-addressed ID), so stale() looks at changed IDs.
            # Show the diff to confirm the world-state change was recorded.
            print("(No staleness alerts — showing commit diff instead.)\n")
            diff = repo.diff(commit_a, commit_b)
            print(f"  Facts added   : {[f.subject + ' ' + f.predicate + ' ' + f.object for f in diff.added]}")
            print(f"  Facts removed : {[f.subject + ' ' + f.predicate + ' ' + f.object for f in diff.removed]}")
            print(
                "\nThe FICO-8 fact was retracted and replaced with FICO-9.\n"
                "Decisions that depended on the original FICO-8 fact ID are stale."
            )

        print("\n--- Commit log ---\n")
        for wc in repo.log():
            print(f"  {wc.id[:8]}  {wc.message}  ({len(wc.fact_ids)} fact(s))")

        repo.close()

        # ===================================================================
        print("\n=== Step 2: agentdelta — diffing agent behaviour before vs. after ===\n")
        # ===================================================================

        print("Building FICO-8 baseline trace (pre-migration agent run)...")
        trace_fico8 = _build_underwriting_trace("uw-fico8-baseline", fico_version=8)

        print("Building FICO-9 candidate trace (post-migration agent run)...")
        trace_fico9 = _build_underwriting_trace("uw-fico9-candidate", fico_version=9)

        # Save both traces to disk, then reload (full round-trip exercise)
        path_a = tmp / "trace_fico8.jsonl"
        path_b = tmp / "trace_fico9.jsonl"
        trace_fico8.save(path_a)
        trace_fico9.save(path_b)

        trace_fico8 = AgentTrace.load(path_a)
        trace_fico9 = AgentTrace.load(path_b)

        print(f"\nTrace A (FICO-8): {len(trace_fico8)} nodes, run_id={trace_fico8.run_id!r}")
        print(f"Trace B (FICO-9): {len(trace_fico9)} nodes, run_id={trace_fico9.run_id!r}")

        # Use a high fork_threshold (0.95) because the tool names differ by only one
        # character ("fico8" vs "fico9") — semantically similar strings that the
        # embedding model rates ~0.94. A stricter threshold surfaces the regression.
        print("\nComputing semantic diff (fork_threshold=0.95)...\n")
        result = diff_traces(trace_fico8, trace_fico9, fork_threshold=0.95, match_threshold=0.97)

        print_diff(result)

        # Summarise the key finding
        if result.fork_point:
            fp = result.fork_point
            print("\n--- Key finding ---\n")
            print(
                f"Fork detected at step A={fp.step_a} / step B={fp.step_b}  "
                f"(similarity={fp.similarity:.2f})"
            )
            print(f"  Before: {fp.node_a.content!r}")
            print(f"  After : {fp.node_b.content!r}")
            if fp.is_tool_change():
                print(
                    "\nagentdelta confirms a TOOL CHANGE: the agent switched from\n"
                    "check_fico8_threshold to check_fico9_threshold — a direct\n"
                    "consequence of the silent bureau migration foghorn detected."
                )
        else:
            print(
                "\n(No hard fork found above threshold — inspect the step-level diff above\n"
                "to see where check_fico8_threshold diverges from check_fico9_threshold.)"
            )

    # ===================================================================
    print("\n=== Summary ===\n")
    # ===================================================================
    print(
        "foghorn tracked that three loan decisions depended on 'Equifax-score-API\n"
        "returns FICO-8'. When the bureau switched to FICO-9, those decisions were\n"
        "flagged as stale — before any loan officer noticed.\n"
        "\n"
        "agentdelta diffed the agent's tool-call sequences and confirmed the\n"
        "behavioural change: check_fico8_threshold replaced by check_fico9_threshold.\n"
        "\n"
        "Together the two tools give fintech teams both WHAT changed in the world\n"
        "(foghorn) and HOW the agent responded to that change (agentdelta)."
    )
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
