Inspect an redline trace file and summarize the agent's execution path.

Usage: /project:inspect <trace.jsonl>

Steps:
1. Run: `redline inspect $ARGUMENTS`
2. Present a clean summary:
   - Total steps and run_id
   - Sequence of node types (start → llm → tool_call → tool_return → ... → end)
   - Each tool call: name and truncated arguments
   - Each LLM step: first 100 chars of reasoning
   - Any metadata fields present
3. Highlight anything unusual:
   - More than 10 steps (complex agent run)
   - Repeated tool calls (possible loop)
   - Missing start or end node
   - Very short LLM reasoning (< 20 chars — may be truncated or empty)

Context:
- Trace format: each line is JSON with "type": "node"|"edge"|"trace_meta"
- Node types: start, llm, tool_call, tool_return, end
- Node IDs are content-addressed (SHA-256[:16]) — same content = same ID across runs
