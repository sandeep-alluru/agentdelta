Generate boilerplate code to record an agent run with agentdelta.

Usage: /project:record [framework] [agent_variable_name]

Arguments: $ARGUMENTS (optional: framework name like "langchain", "langgraph", "custom")

Generate a self-contained Python snippet that:

For LangChain/LangGraph (default):
```python
from agentdelta import record

# Baseline (before your change)
with record("baseline.jsonl", run_id="v1.0") as cb:
    agent.invoke({"input": "..."}, config={"callbacks": [cb]})

# Candidate (after your change)
with record("candidate.jsonl", run_id="v1.1") as cb:
    agent.invoke({"input": "..."}, config={"callbacks": [cb]})
```

For custom/framework-agnostic:
```python
from agentdelta import AgentTrace
from agentdelta.trace import TraceNode, TraceEdge, NodeType, EdgeType

trace = AgentTrace(run_id="my_run")
trace.add_node(TraceNode(step=1, node_type=NodeType.START, content="user input here"))
trace.add_node(TraceNode(step=2, node_type=NodeType.LLM, content="reasoning text here"))
trace.add_node(TraceNode(step=3, node_type=NodeType.TOOL_CALL, content="tool_name(args)"))
trace.add_node(TraceNode(step=4, node_type=NodeType.TOOL_RETURN, content="tool result"))
trace.add_node(TraceNode(step=5, node_type=NodeType.END, content="final output"))
trace.save("my_run.jsonl")
```

Then show the diff command: `agentdelta diff baseline.jsonl candidate.jsonl`

Context:
- AgentDeltaCallback is LangChain BaseCallbackHandler-compatible (no import of BaseCallbackHandler needed)
- record() saves the trace on context exit, even if the agent raises an exception
- run_id appears in the diff report header — use something meaningful (version, git hash, etc.)
