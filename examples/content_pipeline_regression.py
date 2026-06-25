"""
content_pipeline_regression.py — agentdelta for solo content pipeline prompt auditing.

Story: A solo developer runs a 50-post autonomous content pipeline. After changing
one word in the /goal system prompt ("explain" → "illuminate"), the evaluator failure
rate on "technical_insight" posts jumps from 8% to 48%. agentdelta diffs baseline
traces against candidate traces and finds the exact fork: the candidate agent skips
`self_consistency_check` before calling `generate_post`, inflating tool calls 3x.

This example simulates two corpus runs (baseline prompt vs modified prompt) for three
representative post types, computes per-post RegressionScore, surfaces the fork point,
and exits with code 1 if the aggregate behavior warrants a rollback.

Run:
    python examples/content_pipeline_regression.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from agentdelta.diff import diff_traces
from agentdelta.instrument import AgentdeltaCallback
from agentdelta.report import print_diff, to_markdown
from agentdelta.score import compute_score
from agentdelta.trace import AgentTrace

# ---------------------------------------------------------------------------
# Fake LLM response shim — stands in for LangChain's LLMResult.
# In production the agent runs with config={"callbacks": [cb]}.
# ---------------------------------------------------------------------------


class _LLMResponse:
    """Minimal stand-in for langchain_core.outputs.LLMResult."""

    def __init__(self, text: str) -> None:
        self.generations = [[type("Gen", (), {"text": text})()]]


# ---------------------------------------------------------------------------
# Post corpus — three representative post types in the 50-post pipeline.
# "technical_insight" posts are the ones that regress after the prompt change.
# ---------------------------------------------------------------------------

POSTS = [
    {
        "id": "post_001",
        "topic": "Why most LLM eval frameworks measure the wrong thing",
        "type": "technical_insight",
    },
    {
        "id": "post_002",
        "topic": "How we cut our AWS bill 40% without touching infrastructure",
        "type": "founder_story",
    },
    {
        "id": "post_003",
        "topic": "The hidden cost of context windows in production RAG systems",
        "type": "technical_insight",
    },
]

PASS_THRESHOLD = 80.0
WARN_THRESHOLD = 60.0


# ---------------------------------------------------------------------------
# Baseline agent: "explain" system prompt
# Pipeline: research_topic → self_consistency_check → generate_post → evaluate
# ---------------------------------------------------------------------------


def build_baseline_trace(post: dict) -> AgentTrace:
    """
    Baseline content agent — system prompt uses "explain."
    Full tool sequence: research_topic → self_consistency_check → generate_post → evaluate_output.
    """
    cb = AgentdeltaCallback(run_id=f"baseline-{post['id']}")
    cb.on_chain_start({}, {"input": post["topic"]})

    # Step 1: LLM decides to research the topic first
    cb.on_llm_end(_LLMResponse(
        f"I need to research this topic thoroughly before drafting. "
        f"Topic: {post['topic']}"
    ))

    # Step 2: research_topic tool
    cb.on_tool_start({"name": "research_topic"}, f"topic='{post['topic']}', depth='deep'")
    cb.on_tool_end(
        f"Research complete for '{post['topic']}'. "
        "Found 7 high-signal sources. Key angle: practitioners underestimate "
        "second-order effects. Supporting data points: 3 case studies, 2 whitepapers."
    )

    # Step 3: LLM decides to run self-consistency check before drafting
    cb.on_llm_end(_LLMResponse(
        "Before drafting, I should verify the core claim is self-consistent and "
        "defensible. Running self_consistency_check to validate the argument structure."
    ))

    # Step 4: self_consistency_check — this is the step the candidate SKIPS
    cb.on_tool_start(
        {"name": "self_consistency_check"},
        f"claim='practitioners underestimate second-order effects', "
        f"post_type='{post['type']}'",
    )
    cb.on_tool_end(
        "Claim validated. Argument is internally consistent. "
        "Confidence: 0.91. No logical gaps detected. Proceed to draft."
    )

    # Step 5: generate_post
    cb.on_llm_end(_LLMResponse(
        "Self-consistency confirmed. Now I will explain the insight clearly and draft "
        "the post with a strong opening hook."
    ))
    cb.on_tool_start({"name": "generate_post"}, f"topic='{post['topic']}', style='direct'")
    cb.on_tool_end(
        f"Post drafted for '{post['topic']}'. "
        "Length: 312 words. Hook score: 8.4. Technical depth: 8.7. "
        "Readability: 7.9. Draft ready for evaluation."
    )

    # Step 6: evaluate_output
    cb.on_tool_start(
        {"name": "evaluate_output"},
        f"post_type='{post['type']}', criteria=['technical_insight', 'founder_credibility']",
    )
    cb.on_tool_end(
        f"Evaluation complete. technical_insight=8.6, founder_credibility=7.8. "
        "Overall: PASS. Post queued for publish."
    )

    cb.on_llm_end(_LLMResponse(
        "Post passed evaluation. Publishing to queue."
    ))
    cb.on_chain_end({"output": f"Published: {post['topic']} (PASS, score=8.6)"})

    return cb.trace


# ---------------------------------------------------------------------------
# Candidate agent: "illuminate" system prompt
# Regression: skips self_consistency_check on technical_insight posts,
# causing generate_post to run with an unvalidated claim → evaluator failure.
# ---------------------------------------------------------------------------


def build_candidate_trace(post: dict) -> AgentTrace:
    """
    Candidate content agent — system prompt uses "illuminate" instead of "explain."
    Regression on technical_insight posts: self_consistency_check is skipped,
    leading to evaluate_output failures and 3x generate_post calls in production.
    """
    cb = AgentdeltaCallback(run_id=f"candidate-{post['id']}")
    cb.on_chain_start({}, {"input": post["topic"]})

    # Step 1: same research decision
    cb.on_llm_end(_LLMResponse(
        f"I need to research this topic to illuminate the core insight. "
        f"Topic: {post['topic']}"
    ))

    # Step 2: research_topic (identical)
    cb.on_tool_start({"name": "research_topic"}, f"topic='{post['topic']}', depth='deep'")
    cb.on_tool_end(
        f"Research complete for '{post['topic']}'. "
        "Found 7 high-signal sources. Key angle: practitioners underestimate "
        "second-order effects. Supporting data points: 3 case studies, 2 whitepapers."
    )

    if post["type"] == "technical_insight":
        # THE REGRESSION: "illuminate" prompt causes the model to skip
        # self_consistency_check and jump straight to generation.
        # This happens because "illuminate" implies directness and confidence,
        # so the model skips the validation gate.
        cb.on_llm_end(_LLMResponse(
            "The research is clear. I will illuminate this insight directly in the post "
            "without additional validation steps."
        ))

        # Step 4 (MISSING: self_consistency_check — fork point)
        # The candidate goes straight to generate_post
        cb.on_tool_start(
            {"name": "generate_post"},
            f"topic='{post['topic']}', style='illuminating'",
        )
        cb.on_tool_end(
            f"Post drafted for '{post['topic']}'. "
            "Length: 287 words. Hook score: 7.1. Technical depth: 6.2. "
            "Readability: 8.3. Draft ready for evaluation."
        )

        # Evaluator catches the weaker technical depth (no self-consistency validation)
        cb.on_tool_start(
            {"name": "evaluate_output"},
            f"post_type='{post['type']}', criteria=['technical_insight', 'founder_credibility']",
        )
        cb.on_tool_end(
            f"Evaluation complete. technical_insight=5.8, founder_credibility=7.2. "
            "Overall: FAIL. technical_insight below threshold 7.0. Post not published."
        )

        cb.on_llm_end(_LLMResponse(
            "Post failed evaluation on technical_insight. Routing back to drafting queue."
        ))
        cb.on_chain_end(
            {"output": f"Not published: {post['topic']} (FAIL, technical_insight=5.8)"}
        )

    else:
        # founder_story posts: no regression — "illuminate" has no effect on this path
        cb.on_llm_end(_LLMResponse(
            "I'll validate the claim before drafting to ensure the founder story is consistent."
        ))
        cb.on_tool_start(
            {"name": "self_consistency_check"},
            f"claim='cut AWS bill 40% without touching infrastructure', "
            f"post_type='{post['type']}'",
        )
        cb.on_tool_end(
            "Claim validated. Story is internally consistent. Confidence: 0.88. Proceed."
        )
        cb.on_llm_end(_LLMResponse(
            "Validation passed. Illuminating the founder story with concrete detail."
        ))
        cb.on_tool_start(
            {"name": "generate_post"},
            f"topic='{post['topic']}', style='story'",
        )
        cb.on_tool_end(
            f"Post drafted for '{post['topic']}'. "
            "Length: 298 words. Hook score: 8.1. Credibility: 8.5. "
            "Draft ready for evaluation."
        )
        cb.on_tool_start(
            {"name": "evaluate_output"},
            f"post_type='{post['type']}', criteria=['technical_insight', 'founder_credibility']",
        )
        cb.on_tool_end(
            "Evaluation complete. technical_insight=7.2, founder_credibility=8.5. "
            "Overall: PASS. Post queued for publish."
        )
        cb.on_llm_end(_LLMResponse("Post passed evaluation. Publishing to queue."))
        cb.on_chain_end(
            {"output": f"Published: {post['topic']} (PASS, score=8.5)"}
        )

    return cb.trace


# ---------------------------------------------------------------------------
# Main: diff all posts, score each, decide whether to roll back the prompt
# ---------------------------------------------------------------------------


def run_content_regression_check() -> int:
    """
    Diff baseline vs candidate traces for all posts in the corpus.
    Returns 0 (safe to deploy new prompt) or 1 (roll back — regression detected).
    """
    print("=" * 70)
    print("  agentdelta — Content Pipeline Prompt Regression Check")
    print("  Baseline: system prompt 'explain'")
    print("  Candidate: system prompt 'illuminate' (one-word change)")
    print("=" * 70)
    print()

    regressions: list[str] = []
    results: list[tuple[str, str, str, float]] = []  # id, type, verdict, overall

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for post in POSTS:
            pid = post["id"]
            ptype = post["type"]
            print(f"── {pid} [{ptype}]: {post['topic'][:52]}{'...' if len(post['topic']) > 52 else ''}")
            print()

            # Build both traces
            trace_a = build_baseline_trace(post)
            trace_b = build_candidate_trace(post)

            # Save and reload — exercises full serialization round-trip
            path_a = tmp / f"{pid}_baseline.jsonl"
            path_b = tmp / f"{pid}_candidate.jsonl"
            trace_a.save(path_a)
            trace_b.save(path_b)
            trace_a = AgentTrace.load(path_a)
            trace_b = AgentTrace.load(path_b)

            # Semantic diff
            diff = diff_traces(trace_a, trace_b, fork_threshold=0.70, match_threshold=0.85)
            print_diff(diff)

            # Score
            score = compute_score(diff, pass_threshold=PASS_THRESHOLD, warn_threshold=WARN_THRESHOLD)
            print(f"  Score: overall={score.overall:.1f}  "
                  f"tool_fidelity={score.tool_fidelity:.1f}  "
                  f"fork_penalty={score.fork_penalty:.1f}  "
                  f"verdict={score.verdict}")

            if diff.fork_point:
                fp = diff.fork_point
                print(f"  Fork:  step {fp.step_a} — {fp.description}")

            print()

            if score.verdict != "PASS":
                regressions.append(pid)

            results.append((pid, ptype, score.verdict, score.overall))

            # Save markdown diff for artifact inspection
            md_path = tmp / f"{pid}_diff.md"
            md_path.write_text(to_markdown(diff))

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("  REGRESSION CHECK SUMMARY")
    print("=" * 70)
    print(f"  {'Post':<12} {'Type':<20} {'Verdict':<10} {'Score'}")
    print(f"  {'-'*10:<12} {'-'*18:<20} {'-'*8:<10} {'-'*5}")
    for pid, ptype, verdict, overall in results:
        icon = "[PASS]" if verdict == "PASS" else f"[{verdict}]"
        print(f"  {pid:<12} {ptype:<20} {icon:<10} {overall:.1f}")

    print()
    total = len(POSTS)
    passed = total - len(regressions)
    print(f"  Result: {passed}/{total} posts passed behavioral gate")

    if regressions:
        print()
        print(f"  REGRESSION DETECTED in: {', '.join(regressions)}")
        print()
        print("  Root cause: the 'illuminate' system prompt causes the agent to skip")
        print("  `self_consistency_check` on technical_insight post types. Without")
        print("  the validation gate, `generate_post` produces lower technical depth")
        print("  scores (6.2 vs 8.7) and the evaluator rejects the output.")
        print()
        print("  One-word prompt change → skipped tool call → 40% evaluator failure")
        print("  rate increase. `tool_fidelity` score of 33.0 on regressing posts")
        print("  (1 of 3 expected tools matched) pinpointed the skip in seconds.")
        print()
        print("  Recommendation: ROLL BACK the prompt change, or add an explicit")
        print("  `self_consistency_check` step to the tool-routing logic before")
        print("  re-introducing the 'illuminate' wording.")
        return 1

    print()
    print("  All posts matched baseline behavior. Safe to deploy the updated prompt.")
    return 0


if __name__ == "__main__":
    sys.exit(run_content_regression_check())
