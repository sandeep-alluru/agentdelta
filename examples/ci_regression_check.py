"""
ci_regression_check.py — agentdelta in a CI pipeline.

Story: A support team upgrades their customer support agent from v1 to v2.
v1 uses `search_kb` for all queries.
v2 should use `search_kb` + `escalate_to_human` for frustrated customers.
A prompt engineering mistake caused v2 to use `web_search` instead of `search_kb`
for routine queries — agentdelta catches this regression.

Run:
    python examples/ci_regression_check.py

Exit code 0 = all queries matched baseline behavior.
Exit code 1 = regression detected (mimics CI failure).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from agentdelta.diff import diff_traces
from agentdelta.instrument import AgentdeltaCallback
from agentdelta.report import print_diff, to_markdown
from agentdelta.trace import AgentTrace

# ---------------------------------------------------------------------------
# Fake LLM response shim — stands in for real LangChain LLMResult objects.
# In production this would come from: agent.invoke(..., callbacks=[cb])
# ---------------------------------------------------------------------------

class _LLMResponse:
    """Minimal stand-in for langchain_core.outputs.LLMResult."""

    def __init__(self, text: str) -> None:
        self.generations = [[type("Gen", (), {"text": text})()]]


# ---------------------------------------------------------------------------
# Test queries representing real customer support interactions
# ---------------------------------------------------------------------------

QUERIES = [
    {
        "id": "Q1",
        "input": "How do I reset my password?",
        "frustrated": False,
    },
    {
        "id": "Q2",
        "input": "I've been waiting 3 weeks for a refund — this is unacceptable!",
        "frustrated": True,
    },
    {
        "id": "Q3",
        "input": "What payment methods do you accept?",
        "frustrated": False,
    },
]


# ---------------------------------------------------------------------------
# v1 agent: uses search_kb for everything (the established baseline)
# ---------------------------------------------------------------------------

def build_v1_trace(query: dict) -> AgentTrace:
    """
    v1 support agent — always uses search_kb.
    Simulates: ChatOpenAI(model="gpt-4o-mini") + search_kb tool.
    """
    cb = AgentdeltaCallback(run_id=f"v1-{query['id']}")
    cb.on_chain_start({}, {"input": query["input"]})

    cb.on_llm_end(_LLMResponse(
        f"I need to look this up in our knowledge base. "
        f"Query: {query['input']}"
    ))

    # v1 always uses search_kb
    cb.on_tool_start({"name": "search_kb"}, f"query='{query['input']}'")
    if query["frustrated"]:
        cb.on_tool_end(
            "KB article #4421: Refund SLA is 5-7 business days. "
            "If exceeded, issue manual refund via billing portal."
        )
    else:
        cb.on_tool_end(
            "KB article found. Standard procedure documented."
        )

    cb.on_llm_end(_LLMResponse(
        "Based on the knowledge base, I can help with your request. "
        "Here is the information you need."
    ))
    cb.on_chain_end({"output": "Support response delivered via KB lookup."})

    return cb.trace


# ---------------------------------------------------------------------------
# v2 agent: should use search_kb + escalate_to_human for frustrated customers,
# but a prompt engineering mistake caused it to use web_search instead.
# ---------------------------------------------------------------------------

def build_v2_trace(query: dict) -> AgentTrace:
    """
    v2 support agent — regression: uses web_search for routine queries
    instead of search_kb (prompt engineering mistake).
    Simulates: ChatAnthropic(model="claude-3-5-sonnet") + updated tool set.
    """
    cb = AgentdeltaCallback(run_id=f"v2-{query['id']}")
    cb.on_chain_start({}, {"input": query["input"]})

    if query["frustrated"]:
        # Correct new behavior: detect frustration → escalate
        cb.on_llm_end(_LLMResponse(
            "The customer expresses frustration. I should search KB first "
            "then escalate to a human agent."
        ))
        cb.on_tool_start({"name": "search_kb"}, f"query='{query['input']}'")
        cb.on_tool_end(
            "KB article #4421: Refund SLA exceeded — trigger manual refund."
        )
        cb.on_tool_start({"name": "escalate_to_human"}, "priority='high', reason='frustrated_customer'")
        cb.on_tool_end("Ticket #98234 created, assigned to Tier-2 agent.")
        cb.on_llm_end(_LLMResponse(
            "I've searched our KB and escalated your case to a senior agent. "
            "Ticket #98234 has been created with high priority."
        ))
        cb.on_chain_end({"output": "Escalated to human: ticket #98234."})
    else:
        # THE REGRESSION: v2 uses web_search instead of search_kb for routine queries
        # This happened because the new system prompt said "use the best tool available"
        # and the model chose web_search over the internal KB tool.
        cb.on_llm_end(_LLMResponse(
            "I should search for the most up-to-date information on this topic."
        ))
        cb.on_tool_start({"name": "web_search"}, f"query='{query['input']} site:company.com'")
        cb.on_tool_end(
            "Web result: company.com/help — standard help article found."
        )
        cb.on_llm_end(_LLMResponse(
            "Based on web search results, here is the information."
        ))
        cb.on_chain_end({"output": "Support response delivered via web search."})

    return cb.trace


# ---------------------------------------------------------------------------
# CI check runner
# ---------------------------------------------------------------------------

def run_ci_check() -> int:
    """
    Run regression check across all test queries.
    Returns 0 (pass) or 1 (fail) — suitable for GitHub Actions exit code.
    """
    print("=" * 70)
    print("  agentdelta CI Regression Check")
    print("  Baseline: support-agent-v1  →  Candidate: support-agent-v2")
    print("=" * 70)
    print()

    regressions: list[str] = []
    results: list[tuple[str, str, str]] = []  # (query_id, status, detail)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for query in QUERIES:
            qid = query["id"]
            print(f"── Query {qid}: {query['input'][:55]}{'...' if len(query['input']) > 55 else ''}")
            print()

            # Build both traces
            trace_a = build_v1_trace(query)
            trace_b = build_v2_trace(query)

            # Save and reload (exercises full serialization pipeline,
            # same as loading traces from CI artifact storage)
            path_a = tmp / f"{qid}_v1.jsonl"
            path_b = tmp / f"{qid}_v2.jsonl"
            trace_a.save(path_a)
            trace_b.save(path_b)
            trace_a = AgentTrace.load(path_a)
            trace_b = AgentTrace.load(path_b)

            # Semantic diff
            result = diff_traces(trace_a, trace_b, fork_threshold=0.70, match_threshold=0.85)
            print_diff(result)

            if result.has_regression:
                regressions.append(qid)
                fp = result.fork_point
                detail = f"fork at step {fp.step_a}: {fp.description}" if fp else "diverged"
                results.append((qid, "FAIL", detail))
            else:
                results.append((qid, "PASS", "behavior equivalent"))

            # Write markdown report for GitHub PR comment artifact
            md_path = tmp / f"{qid}_diff.md"
            md_path.write_text(to_markdown(result))

    # ---------------------------------------------------------------------------
    # Summary table — printed to CI log
    # ---------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  REGRESSION CHECK SUMMARY")
    print("=" * 70)
    print(f"  {'Query':<10} {'Status':<10} Detail")
    print(f"  {'-'*8:<10} {'-'*8:<10} ------")
    for qid, status, detail in results:
        icon = "PASS" if status == "PASS" else "FAIL"
        print(f"  {qid:<10} [{icon}]    {detail}")

    print()
    total = len(QUERIES)
    passed = total - len(regressions)
    print(f"  Result: {passed}/{total} queries passed")

    if regressions:
        print()
        print("  REGRESSION DETECTED in queries:", ", ".join(regressions))
        print()
        print("  Root cause: v2 uses `web_search` for routine queries instead")
        print("  of `search_kb`. This is a prompt engineering regression —")
        print("  the new system prompt ('use the best tool available') caused")
        print("  the model to bypass the internal knowledge base.")
        print()
        print("  Fix: update v2 system prompt to explicitly prefer `search_kb`")
        print("  for customer queries before falling back to `web_search`.")
        print()
        print("  GitHub Actions: this step will fail (exit code 1).")
        print("  Review the diff artifacts uploaded to this workflow run.")
        return 1
    else:
        print()
        print("  All queries matched baseline behavior. Safe to deploy v2.")
        return 0


if __name__ == "__main__":
    sys.exit(run_ci_check())
