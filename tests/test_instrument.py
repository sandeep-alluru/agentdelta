"""Tests for agentdelta.instrument."""

from __future__ import annotations

import tempfile
from pathlib import Path

from agentdelta.instrument import AgentdeltaCallback, record
from agentdelta.trace import AgentTrace, NodeType


class _FakeLLMResponse:
    def __init__(self, text: str):
        self.generations = [[_FakeGeneration(text)]]


class _FakeGeneration:
    def __init__(self, text: str):
        self.text = text


def test_callback_records_llm_node():
    cb = AgentdeltaCallback(run_id="test")
    cb.on_llm_end(_FakeLLMResponse("I should search for this."))
    assert len(cb.trace.nodes) == 1
    assert cb.trace.nodes[0].node_type == NodeType.LLM
    assert "search" in cb.trace.nodes[0].content


def test_callback_records_tool_nodes():
    cb = AgentdeltaCallback(run_id="test")
    cb.on_tool_start({"name": "web_search"}, "query='foo'")
    cb.on_tool_end("result: bar")
    nodes = cb.trace.nodes
    assert nodes[0].node_type == NodeType.TOOL_CALL
    assert nodes[1].node_type == NodeType.TOOL_RETURN
    assert "web_search" in nodes[0].content


def test_callback_creates_start_on_chain_start():
    cb = AgentdeltaCallback(run_id="test")
    cb.on_chain_start({}, {"input": "hello"})
    assert len(cb.trace.nodes) == 1
    assert cb.trace.nodes[0].node_type == NodeType.START


def test_callback_no_duplicate_start():
    cb = AgentdeltaCallback(run_id="test")
    cb.on_chain_start({}, {"input": "hello"})
    cb.on_chain_start({}, {"input": "hello again"})
    starts = [n for n in cb.trace.nodes if n.node_type == NodeType.START]
    assert len(starts) == 1


def test_callback_records_end_on_chain_end():
    cb = AgentdeltaCallback(run_id="test")
    cb.on_chain_start({}, {"input": "hello"})
    cb.on_chain_end({"output": "done"})
    ends = [n for n in cb.trace.nodes if n.node_type == NodeType.END]
    assert len(ends) == 1


def test_callback_edges_connect_steps():
    cb = AgentdeltaCallback(run_id="test")
    cb.on_chain_start({}, {"input": "hello"})
    cb.on_llm_end(_FakeLLMResponse("thinking"))
    cb.on_tool_start({"name": "search"}, "query='x'")
    assert len(cb.trace.edges) == 2


def test_record_context_manager_saves_file():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)

    try:
        with record(path, run_id="ctx_test") as cb:
            cb.on_chain_start({}, {"input": "test"})
            cb.on_llm_end(_FakeLLMResponse("I will search."))

        trace = AgentTrace.load(path)
        assert trace.run_id == "ctx_test"
        assert len(trace.nodes) >= 2
    finally:
        path.unlink(missing_ok=True)
