Scaffold a new instrumentation adapter for a framework not yet supported by agentdelta.

Usage: /project:add-adapter <framework_name>

Framework name from $ARGUMENTS (e.g. "autogen", "crewai", "smolagents", "pydantic-ai").

Generate the following:

1. `src/agentdelta/instrument_<framework>.py` — adapter module:
   - A class `<Framework>AgentdeltaAdapter` that wraps the framework's event/callback API
   - Must emit `TraceNode` objects to an `AgentTrace` for each: agent start, LLM call, tool call, tool return, agent end
   - Use `NodeType.START`, `NodeType.LLM`, `NodeType.TOOL_CALL`, `NodeType.TOOL_RETURN`, `NodeType.END`
   - Include a `record_<framework>()` context manager mirroring the LangChain `record()` API

2. Export from `src/agentdelta/__init__.py`:
   - Add to `__all__` alphabetically

3. `tests/test_instrument_<framework>.py` — at least 3 tests:
   - Callback captures LLM output as a NodeType.LLM node
   - Callback captures tool calls as NodeType.TOOL_CALL nodes
   - record() context manager saves a valid JSONL file on exit

4. Update `README.md` Quick Start section with a snippet for the new framework

5. Update `pyproject.toml` optional dependencies:
   - Add `[<framework>]` extra with the framework package pinned to `>=` minimum supported version

Context:
- See `src/agentdelta/instrument.py` for the LangChain reference implementation
- AgentTrace.add_node() and add_edge() are the only write methods needed
- Keep content fields under 2000 chars for LLM nodes, 500 chars for tool nodes
- The adapter does NOT need to import agentdelta internals other than trace.py
