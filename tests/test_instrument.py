"""Tests for agentdelta.instrument."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import ClassVar

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
    """record() context manager should save the trace to the specified path on exit."""
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


def test_on_llm_end_fallback_when_response_malformed():
    """on_llm_end should fall back to str(response) when generations is missing."""
    cb = AgentdeltaCallback(run_id="fb")

    class _BadResponse:
        pass  # no .generations attribute

    cb.on_llm_end(_BadResponse())
    assert len(cb.trace.nodes) == 1
    assert cb.trace.nodes[0].node_type == NodeType.LLM
    # content should be the str() of the object, not crash
    assert isinstance(cb.trace.nodes[0].content, str)


def test_on_llm_end_fallback_when_generations_empty():
    """on_llm_end should fall back to str(response) when generations list is empty."""
    cb = AgentdeltaCallback(run_id="empty_gen")

    class _EmptyGens:
        generations: ClassVar[list] = []

    cb.on_llm_end(_EmptyGens())
    assert len(cb.trace.nodes) == 1
    assert cb.trace.nodes[0].node_type == NodeType.LLM


def test_on_tool_start_without_name():
    """on_tool_start should default tool name to 'unknown_tool' when 'name' key is absent."""
    cb = AgentdeltaCallback(run_id="noname")
    cb.on_tool_start({}, "some_input")
    assert len(cb.trace.nodes) == 1
    assert "unknown_tool" in cb.trace.nodes[0].content


def test_on_chain_end_without_output_key():
    """on_chain_end should serialize the full outputs dict when 'output' key is absent."""
    cb = AgentdeltaCallback(run_id="noout")
    cb.on_chain_end({"result": "42"})
    ends = [n for n in cb.trace.nodes if n.node_type == NodeType.END]
    assert len(ends) == 1
    assert "42" in ends[0].content or "result" in ends[0].content


def test_error_shims_are_no_ops():
    """LangGraph error shim methods must not raise and must not add nodes."""
    cb = AgentdeltaCallback(run_id="errs")
    cb.on_llm_error(RuntimeError("llm fail"))
    cb.on_tool_error(ValueError("tool fail"))
    cb.on_chain_error(Exception("chain fail"))
    assert len(cb.trace.nodes) == 0


def test_on_agent_action_and_finish_are_no_ops():
    """on_agent_action and on_agent_finish must not add nodes."""
    cb = AgentdeltaCallback(run_id="noop")
    cb.on_agent_action(object())
    cb.on_agent_finish(object())
    assert len(cb.trace.nodes) == 0


def test_run_id_auto_generated_when_none():
    """AgentdeltaCallback with run_id=None should auto-generate a non-empty run_id."""
    cb = AgentdeltaCallback()
    assert isinstance(cb.run_id, str)
    assert len(cb.run_id) > 0


def test_on_llm_start_is_no_op():
    """on_llm_start must not add any nodes (prompts are not captured)."""
    cb = AgentdeltaCallback(run_id="llm_start")
    cb.on_llm_start({"name": "gpt"}, ["prompt text"])
    assert len(cb.trace.nodes) == 0
