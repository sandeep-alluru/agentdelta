Diff two agentdelta trace files and report any behavioral regression.

Usage: /project:diff <baseline.jsonl> <candidate.jsonl> [--format rich|json|markdown] [--exit-code]

Steps:
1. Run: `agentdelta diff $ARGUMENTS`
2. Parse the output:
   - "REGRESSION DETECTED" → a ForkPoint was found; show the fork step, tool change, and similarity score
   - "No regression" → traces are equivalent; show match percentage
3. If a fork is detected, explain in plain English what changed:
   - Tool change (e.g. get_weather → web_search): mention latency/reliability implications
   - Reasoning divergence: mention that the LLM reasoning path changed even if the final answer is the same
   - Step count difference: mention added/removed tool calls
4. Suggest a remediation if asked: tighten the system prompt, pin the model version, or add a unit test fixture for this trace pair.

Context:
- Traces are JSONL files — one JSON object per line (trace_meta, node, edge records)
- Fork threshold default: 0.70 cosine similarity
- Match threshold default: 0.85 cosine similarity
- The embedding model is all-MiniLM-L6-v2, runs locally, no API key needed
- `has_regression` is True iff fork_point is not None
