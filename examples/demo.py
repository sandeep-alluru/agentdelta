"""
Demo: create two synthetic agent traces and diff them.

Run from the repo root:
    python examples/demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agentdelta.diff import diff_traces
from agentdelta.instrument import AgentdeltaCallback
from agentdelta.report import print_diff, to_markdown
from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode


def _build_trace(run_id: str, use_weather_tool: bool) -> AgentTrace:
    """Build a synthetic weather-query trace.

    One variant uses `get_weather`, the other uses `web_search` — this
    is the fork that agentdelta should detect.
    """
    cb = AgentdeltaCallback(run_id=run_id)
    cb.on_chain_start({}, {"input": "What is the weather in Tokyo?"})
    cb.on_llm_end(
        _FakeLLMResponse("I should look up the current weather in Tokyo.")
    )

    if use_weather_tool:
        cb.on_tool_start({"name": "get_weather"}, "location='Tokyo'")
        cb.on_tool_end('{"temp": 22, "condition": "sunny", "humidity": 60}')
        cb.on_llm_end(
            _FakeLLMResponse("The weather in Tokyo is 22°C and sunny with 60% humidity.")
        )
        cb.on_chain_end({"output": "Tokyo: 22°C, sunny, humidity 60%."})
    else:
        cb.on_tool_start({"name": "web_search"}, "query='Tokyo weather today'")
        cb.on_tool_end("Tokyo: 22°C, mostly sunny — Japan Meteorological Agency")
        cb.on_llm_end(
            _FakeLLMResponse("According to JMA, Tokyo is 22°C and mostly sunny today.")
        )
        cb.on_chain_end({"output": "Tokyo is 22°C and mostly sunny (JMA)."})

    return cb.trace


class _FakeLLMResponse:
    def __init__(self, text: str):
        self.generations = [[type("G", (), {"text": text})()]]


def main() -> None:
    print("Building baseline trace (get_weather tool)…")
    trace_a = _build_trace("baseline-001", use_weather_tool=True)

    print("Building candidate trace (web_search tool)…")
    trace_b = _build_trace("candidate-002", use_weather_tool=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        path_a = Path(tmpdir) / "baseline.jsonl"
        path_b = Path(tmpdir) / "candidate.jsonl"
        trace_a.save(path_a)
        trace_b.save(path_b)

        # Reload from disk to exercise the full pipeline
        trace_a = AgentTrace.load(path_a)
        trace_b = AgentTrace.load(path_b)

    print("\nComputing semantic diff…\n")
    result = diff_traces(trace_a, trace_b, fork_threshold=0.70, match_threshold=0.85)

    print_diff(result)

    print("\n--- Markdown output (for GitHub PR comments) ---\n")
    print(to_markdown(result)[:800] + "\n…")


if __name__ == "__main__":
    main()
