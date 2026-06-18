"""MCP server exposing redline as Claude tools.

Start with: python -m redline.mcp_server
Or via CLI: redline mcp

Add to Claude Desktop (~/.config/claude/claude_desktop_config.json):
    {
        "mcpServers": {
            "redline": {
                "command": "python",
                "args": ["-m", "redline.mcp_server"]
            }
        }
    }

Tools exposed:
    diff_traces     — compare two JSONL trace files, returns DiffResult as JSON
    inspect_trace   — summarise a single JSONL trace file
    record_snippet  — return copy-paste Python code to record a framework agent run
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _require_mcp() -> Any:
    try:
        import mcp.server.stdio
        import mcp.types as types
        from mcp.server import Server
        return mcp, types, Server
    except ImportError:
        print(
            "MCP server requires: pip install 'redline[mcp]'",
            file=sys.stderr,
        )
        sys.exit(1)


def _diff(
    path_a: str, path_b: str, fork_threshold: float, match_threshold: float
) -> dict[str, Any]:
    from redline import AgentTrace, diff_traces
    from redline.report import to_json

    trace_a = AgentTrace.load(path_a)
    trace_b = AgentTrace.load(path_b)
    result = diff_traces(
        trace_a, trace_b, fork_threshold=fork_threshold, match_threshold=match_threshold
    )
    return json.loads(to_json(result))


def _inspect(path: str) -> dict[str, Any]:
    from redline import AgentTrace

    trace = AgentTrace.load(path)
    steps = []
    for node in trace.nodes:
        steps.append({
            "step": node.step,
            "type": node.node_type.value,
            "content_preview": node.content[:120],
            "id": node.id,
        })
    return {
        "run_id": trace.run_id,
        "total_nodes": len(trace.nodes),
        "total_edges": len(trace.edges),
        "steps": steps,
        "metadata": trace.metadata,
    }


_RECORD_SNIPPETS = {
    "langchain": '''\
from redline import record

with record("baseline.jsonl", run_id="v1.0") as cb:
    agent.invoke({"input": "..."}, config={"callbacks": [cb]})

with record("candidate.jsonl", run_id="v1.1") as cb:
    agent.invoke({"input": "..."}, config={"callbacks": [cb]})
''',
    "custom": '''\
from redline import AgentTrace
from redline.trace import TraceNode, NodeType

trace = AgentTrace(run_id="my_run")
trace.add_node(TraceNode(step=1, node_type=NodeType.START, content="user input"))
trace.add_node(TraceNode(step=2, node_type=NodeType.LLM, content="reasoning..."))
trace.add_node(TraceNode(step=3, node_type=NodeType.TOOL_CALL, content="tool_name(args)"))
trace.add_node(TraceNode(step=4, node_type=NodeType.TOOL_RETURN, content="result"))
trace.add_node(TraceNode(step=5, node_type=NodeType.END, content="final answer"))
trace.save("my_run.jsonl")
''',
}


def run_server() -> None:
    """Start the MCP server on stdio."""
    mcp_mod, types, Server = _require_mcp()

    server = Server("redline")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="diff_traces",
                description=(
                    "Compare two redline JSONL trace files. Returns a DiffResult JSON with "
                    "fork_point (first divergent step), has_regression bool, and per-step details."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "trace_a": {"type": "string", "description": "Path to baseline JSONL"},
                        "trace_b": {"type": "string", "description": "Path to candidate JSONL"},
                        "fork_threshold": {
                            "type": "number", "default": 0.70,
                            "description": "Similarity below this marks a fork",
                        },
                        "match_threshold": {
                            "type": "number", "default": 0.85,
                            "description": "Similarity above this is a match",
                        },
                    },
                    "required": ["trace_a", "trace_b"],
                },
            ),
            types.Tool(
                name="inspect_trace",
                description=(
                    "Summarise a single redline JSONL trace file: run_id, node count, "
                    "step sequence with type and content preview."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "trace_path": {"type": "string", "description": "Path to JSONL trace file"},
                    },
                    "required": ["trace_path"],
                },
            ),
            types.Tool(
                name="record_snippet",
                description=(
                    "Return a copy-paste Python snippet to record an agent run with redline."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "framework": {
                            "type": "string",
                            "enum": ["langchain", "custom"],
                            "default": "langchain",
                            "description": "Agent framework to generate snippet for",
                        },
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        if name == "diff_traces":
            result = _diff(
                arguments["trace_a"],
                arguments["trace_b"],
                float(arguments.get("fork_threshold", 0.70)),
                float(arguments.get("match_threshold", 0.85)),
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        if name == "inspect_trace":
            result = _inspect(arguments["trace_path"])
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        if name == "record_snippet":
            framework = arguments.get("framework", "langchain")
            snippet = _RECORD_SNIPPETS.get(framework, _RECORD_SNIPPETS["custom"])
            return [types.TextContent(type="text", text=snippet)]

        raise ValueError(f"Unknown tool: {name}")

    import asyncio

    async def _main() -> None:
        async with mcp_mod.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(_main())


if __name__ == "__main__":
    run_server()
