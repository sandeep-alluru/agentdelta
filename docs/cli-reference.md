# CLI Reference

## `agentdelta diff`

Compare two agent trace files and report any behavioral regression.

```bash
agentdelta diff TRACE_A TRACE_B [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--format` | `rich` | Output format: `rich` \| `json` \| `markdown` |
| `--fork-threshold` | `0.70` | Similarity below this marks a fork point |
| `--match-threshold` | `0.85` | Similarity above this is a match (no change) |
| `--show-matches` | `false` | Include unchanged steps in the output |
| `--exit-code` | `false` | Exit 1 if a regression is detected (for CI pipelines) |

### Examples

```bash
# Default rich terminal output
agentdelta diff baseline.jsonl candidate.jsonl

# JSON output for programmatic use
agentdelta diff baseline.jsonl candidate.jsonl --format json

# Markdown for a GitHub PR comment
agentdelta diff baseline.jsonl candidate.jsonl --format markdown > diff.md

# CI gate — fails the pipeline if traces diverged
agentdelta diff baseline.jsonl candidate.jsonl --exit-code

# Show all steps including unchanged ones
agentdelta diff baseline.jsonl candidate.jsonl --show-matches

# Tighter thresholds (stricter regression detection)
agentdelta diff baseline.jsonl candidate.jsonl \
  --fork-threshold 0.80 \
  --match-threshold 0.90
```

---

## `agentdelta inspect`

Print a step-by-step summary of a single trace file.

```bash
agentdelta inspect TRACE_FILE
```

No options. Outputs a table of steps with node type, content preview, and node ID.

### Example

```bash
agentdelta inspect baseline.jsonl
```

```
run_id: v1.0  ·  6 nodes  ·  5 edges

 Step   Type          Content
    1   start         What is the weather in Tokyo?
    2   llm           I should look up the current weather...
    3   tool_call     get_weather(location='Tokyo')
    4   tool_return   {"temp": 22, "condition": "sunny"}
    5   llm           The weather in Tokyo is 22°C and sunny.
    6   end           Tokyo: 22°C, sunny.
```

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success — no regression detected (or `--exit-code` not set) |
| `1` | Regression detected (only when `--exit-code` is set) |
| `2` | Invalid arguments or file not found |
