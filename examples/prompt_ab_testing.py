"""
prompt_ab_testing.py — agentdelta for A/B prompt variant comparison.

Story: A prompt engineer is testing 3 system prompt variants for a research
assistant agent:

  Variant A (baseline): concise, direct instructions
  Variant B (CoT):      "think step by step" chain-of-thought
  Variant C (tools):    "use tools liberally" — more aggressive tool use

For 3 test queries, agentdelta diffs B vs A and C vs A to build a behavioral
comparison matrix. This reveals which prompt changes actually alter behavior
vs which only change wording.

Run:
    python examples/prompt_ab_testing.py
"""

from __future__ import annotations

import sys
from typing import NamedTuple

from agentdelta.diff import diff_traces
from agentdelta.instrument import AgentdeltaCallback
from agentdelta.trace import AgentTrace


# ---------------------------------------------------------------------------
# Fake LLM response shim
# ---------------------------------------------------------------------------

class _LLMResponse:
    def __init__(self, text: str) -> None:
        self.generations = [[type("Gen", (), {"text": text})()]]


# ---------------------------------------------------------------------------
# Test queries for the research assistant
# ---------------------------------------------------------------------------

QUERIES = [
    {
        "id": "Q1",
        "topic": "What is the current state of quantum computing?",
        "category": "factual-overview",
    },
    {
        "id": "Q2",
        "topic": "Compare LangChain vs LlamaIndex for building RAG systems",
        "category": "comparative-analysis",
    },
    {
        "id": "Q3",
        "topic": "How does AlphaFold 3 improve on AlphaFold 2?",
        "category": "technical-deep-dive",
    },
]


# ---------------------------------------------------------------------------
# Variant A (baseline): concise, does one web_search then summarizes
# ---------------------------------------------------------------------------

def build_variant_a(query: dict) -> AgentTrace:
    """
    Variant A — baseline system prompt: concise, direct.
    System: "You are a research assistant. Answer questions concisely."
    Behavior: single web_search → LLM summary
    """
    cb = AgentdeltaCallback(run_id=f"variant-A-{query['id']}")
    cb.on_chain_start({}, {"input": query["topic"]})

    cb.on_llm_end(_LLMResponse(
        f"I'll search for information about this topic."
    ))

    cb.on_tool_start({"name": "web_search"}, f"query='{query['topic']}'")
    cb.on_tool_end(
        f"[Search results for: {query['topic']}] "
        "Found 3 relevant articles. Key findings summarized."
    )

    cb.on_llm_end(_LLMResponse(
        f"Based on search results: {query['topic']} — here is a concise answer."
    ))
    cb.on_chain_end({"output": "Concise research summary delivered."})

    return cb.trace


# ---------------------------------------------------------------------------
# Variant B (CoT): "think step by step" — adds reasoning steps but same tools
# ---------------------------------------------------------------------------

def build_variant_b(query: dict) -> AgentTrace:
    """
    Variant B — CoT system prompt: "think step by step before answering."
    System: "You are a research assistant. Think step by step before answering."
    Behavior: reasoning step → web_search → reasoning step → summary
    The extra LLM calls change wording but not tool selection.
    """
    cb = AgentdeltaCallback(run_id=f"variant-B-{query['id']}")
    cb.on_chain_start({}, {"input": query["topic"]})

    # Extra reasoning step (CoT effect)
    cb.on_llm_end(_LLMResponse(
        f"Let me think step by step. First, I need to understand what is being asked: "
        f"{query['topic']}. I should search for the most recent information."
    ))

    # Same tool as variant A
    cb.on_tool_start({"name": "web_search"}, f"query='{query['topic']}'")
    cb.on_tool_end(
        f"[Search results for: {query['topic']}] "
        "Found 3 relevant articles. Key findings summarized."
    )

    # Extra synthesis reasoning step (CoT effect)
    cb.on_llm_end(_LLMResponse(
        f"Now let me synthesize these findings step by step. "
        f"The search results address {query['topic']} from multiple angles."
    ))

    cb.on_llm_end(_LLMResponse(
        f"Final answer about {query['topic']}: comprehensive summary with step-by-step reasoning."
    ))
    cb.on_chain_end({"output": "Step-by-step research summary delivered."})

    return cb.trace


# ---------------------------------------------------------------------------
# Variant C (tools): "use tools liberally" — more tool calls, different tools
# ---------------------------------------------------------------------------

def build_variant_c(query: dict) -> AgentTrace:
    """
    Variant C — tool-heavy system prompt: "use tools liberally to maximize accuracy."
    System: "You are a research assistant. Use all available tools liberally."
    Behavior: web_search + arxiv_search + summarize_sources (more tool calls)
    This is a genuine behavioral change vs variant A.
    """
    cb = AgentdeltaCallback(run_id=f"variant-C-{query['id']}")
    cb.on_chain_start({}, {"input": query["topic"]})

    cb.on_llm_end(_LLMResponse(
        f"I'll use multiple tools to maximize accuracy on this topic."
    ))

    # Tool 1: web_search (same as A and B)
    cb.on_tool_start({"name": "web_search"}, f"query='{query['topic']}'")
    cb.on_tool_end(
        f"[Web search: {query['topic']}] General results from web."
    )

    # Tool 2: arxiv_search (ADDED — not in A or B)
    cb.on_llm_end(_LLMResponse(
        "I found web results. Let me also check academic sources for more depth."
    ))
    cb.on_tool_start({"name": "arxiv_search"}, f"query='{query['topic']}', max_results=5")
    cb.on_tool_end(
        f"[arXiv: {query['topic']}] 5 recent papers found. "
        "Most recent: 2025. Key citations identified."
    )

    # Tool 3: summarize_sources (ADDED — aggregates findings)
    cb.on_llm_end(_LLMResponse(
        "I have both web and academic results. Let me cross-reference them."
    ))
    cb.on_tool_start({"name": "summarize_sources"}, "sources=['web', 'arxiv'], format='structured'")
    cb.on_tool_end(
        "Cross-referenced summary: web and academic sources agree on core findings. "
        "2 discrepancies noted in recent developments."
    )

    cb.on_llm_end(_LLMResponse(
        f"Comprehensive answer using web + arxiv sources about {query['topic']}."
    ))
    cb.on_chain_end({"output": "Multi-source research summary delivered."})

    return cb.trace


# ---------------------------------------------------------------------------
# Cell renderer for the comparison matrix
# ---------------------------------------------------------------------------

class CellResult(NamedTuple):
    match_pct: float
    has_divergence: bool
    fork_description: str


def compare_traces(trace_baseline: AgentTrace, trace_variant: AgentTrace) -> CellResult:
    result = diff_traces(trace_baseline, trace_variant, fork_threshold=0.70, match_threshold=0.85)
    pct = result.summary.get("similarity_pct", 100.0)
    if result.fork_point:
        desc = result.fork_point.description[:35]
    elif result.added_steps or result.removed_steps:
        n = len(result.added_steps) + len(result.removed_steps)
        desc = f"{n} step(s) added/removed"
    else:
        desc = "equivalent behavior"
    return CellResult(
        match_pct=pct,
        has_divergence=result.has_regression,
        fork_description=desc,
    )


# ---------------------------------------------------------------------------
# Main: build comparison matrix and print
# ---------------------------------------------------------------------------

def run_ab_test() -> int:
    print("=" * 72)
    print("  agentdelta Prompt A/B Testing — Behavioral Comparison Matrix")
    print("  Research Assistant: 3 prompt variants x 3 test queries")
    print("=" * 72)
    print()
    print("  Variant A (baseline): concise prompt — 1 tool call")
    print("  Variant B (CoT):      'think step by step' — same tools, more reasoning")
    print("  Variant C (tools):    'use tools liberally' — adds arxiv + summarize")
    print()

    # Build all traces
    traces: dict[str, dict[str, AgentTrace]] = {"A": {}, "B": {}, "C": {}}
    for query in QUERIES:
        traces["A"][query["id"]] = build_variant_a(query)
        traces["B"][query["id"]] = build_variant_b(query)
        traces["C"][query["id"]] = build_variant_c(query)

    # Compute comparison matrix: B vs A and C vs A, for each query
    matrix: dict[str, dict[str, CellResult]] = {}
    for query in QUERIES:
        qid = query["id"]
        matrix[qid] = {
            "B_vs_A": compare_traces(traces["A"][qid], traces["B"][qid]),
            "C_vs_A": compare_traces(traces["A"][qid], traces["C"][qid]),
        }

    # ---------------------------------------------------------------------------
    # Print behavioral comparison matrix
    # ---------------------------------------------------------------------------
    print("  BEHAVIORAL COMPARISON MATRIX")
    print(f"  {'Query':<6} {'Category':<22} {'B vs A (CoT)':<30} {'C vs A (Tools)'}")
    print(f"  {'-'*4:<6} {'-'*20:<22} {'-'*28:<30} {'-'*28}")

    for query in QUERIES:
        qid = query["id"]
        bva = matrix[qid]["B_vs_A"]
        cva = matrix[qid]["C_vs_A"]

        def cell(r: CellResult) -> str:
            icon = "DIFF" if r.has_divergence else "SAME"
            return f"[{icon}] {r.match_pct:.0f}% — {r.fork_description[:20]}"

        print(f"  {qid:<6} {query['category'][:20]:<22} {cell(bva):<30} {cell(cva)}")

    # ---------------------------------------------------------------------------
    # Match score table
    # ---------------------------------------------------------------------------
    print()
    print("  MATCH SCORES (% of steps semantically equivalent to baseline A)")
    print()
    print(f"  {'Query':<6}", end="")
    for variant in ("B (CoT)", "C (Tools)"):
        print(f"  {variant:<16}", end="")
    print()
    print(f"  {'-'*4:<6}", end="")
    for _ in range(2):
        print(f"  {'-'*14:<16}", end="")
    print()

    for query in QUERIES:
        qid = query["id"]
        bva = matrix[qid]["B_vs_A"]
        cva = matrix[qid]["C_vs_A"]
        print(f"  {qid:<6}  {bva.match_pct:>5.1f}%          {cva.match_pct:>5.1f}%")

    avg_b = sum(matrix[q["id"]]["B_vs_A"].match_pct for q in QUERIES) / len(QUERIES)
    avg_c = sum(matrix[q["id"]]["C_vs_A"].match_pct for q in QUERIES) / len(QUERIES)
    print(f"  {'AVG':<6}  {avg_b:>5.1f}%          {avg_c:>5.1f}%")

    # ---------------------------------------------------------------------------
    # Interpretation and recommendation
    # ---------------------------------------------------------------------------
    print()
    print("  INTERPRETATION:")
    print()
    print(f"  Variant B (CoT, avg {avg_b:.0f}% match):")
    print("    The 'think step by step' prompt changes reasoning WORDING but")
    print("    not tool SELECTION. agentdelta shows the same tool sequence.")
    print("    Behavioral impact: LOW. Extra LLM calls add latency but don't")
    print("    change what the agent actually does. Deploy if latency is OK.")
    print()
    print(f"  Variant C (Tools, avg {avg_c:.0f}% match):")
    print("    The 'use tools liberally' prompt adds arxiv_search and")
    print("    summarize_sources tool calls — genuine behavioral change.")
    print("    Behavioral impact: HIGH. More accurate but significantly")
    print("    higher latency and cost. Validate quality uplift before deploy.")
    print()
    print("  RECOMMENDATION: Use variant B for latency-sensitive paths,")
    print("  variant C for deep research queries where accuracy > speed.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(run_ab_test())
