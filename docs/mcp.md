# MCP / Claude Integration

redline ships an MCP (Model Context Protocol) server that exposes its core operations as native Claude tools.

## Install

```bash
pip install "redline[mcp]"
```

## Add to Claude Desktop

Edit `~/.config/claude/claude_desktop_config.json` (Linux) or  
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "redline": {
      "command": "redline-mcp"
    }
  }
}
```

Restart Claude Desktop. You now have three tools available in every conversation:

## Available tools

### `diff_traces`

Compare two JSONL trace files and return a `DiffResult` JSON.

```
diff_traces(trace_a="/path/to/baseline.jsonl", trace_b="/path/to/candidate.jsonl")
```

Returns: `has_regression`, `fork_point` (step, description, similarity), `summary` stats, and per-step alignment.

### `inspect_trace`

Summarise a single trace file's execution path.

```
inspect_trace(trace_path="/path/to/trace.jsonl")
```

Returns: `run_id`, node/edge count, step breakdown with type and content preview.

### `record_snippet`

Get a copy-paste Python snippet to record an agent run.

```
record_snippet(framework="langchain")
```

Returns: ready-to-run Python code for the specified framework.

## Claude Code slash commands

After cloning the repo, these project-level commands are available in Claude Code:

| Command | What it does |
|---|---|
| `/project:diff <baseline> <candidate>` | Diff two traces and explain the fork |
| `/project:inspect <trace>` | Summarise a trace's execution path |
| `/project:record [framework]` | Generate recording boilerplate |
| `/project:add-adapter <framework>` | Scaffold a new framework adapter |
| `/project:pr-prep` | Run lint + types + tests + CHANGELOG check |
| `/project:test` | Run test suite and report failures |

## Smithery

redline is also listed on [smithery.ai](https://smithery.ai) — an MCP server marketplace. Search for "redline" to install with one click from Claude Desktop.
