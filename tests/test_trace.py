"""Tests for agentdelta.trace."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode


def test_node_id_is_deterministic():
    n1 = TraceNode(step=1, node_type=NodeType.LLM, content="hello")
    n2 = TraceNode(step=2, node_type=NodeType.LLM, content="hello")
    assert n1.id == n2.id  # same type+content → same ID


def test_node_id_differs_on_content():
    n1 = TraceNode(step=1, node_type=NodeType.LLM, content="hello")
    n2 = TraceNode(step=1, node_type=NodeType.LLM, content="world")
    assert n1.id != n2.id


def test_node_roundtrip():
    node = TraceNode(step=3, node_type=NodeType.TOOL_CALL, content="search('foo')")
    d = node.to_dict()
    restored = TraceNode.from_dict(d)
    assert restored.step == node.step
    assert restored.node_type == node.node_type
    assert restored.content == node.content


def test_edge_roundtrip():
    edge = TraceEdge(1, 2, EdgeType.TOOL_CALL, "search")
    d = edge.to_dict()
    restored = TraceEdge.from_dict(d)
    assert restored.source_step == 1
    assert restored.target_step == 2
    assert restored.edge_type == EdgeType.TOOL_CALL


def test_trace_save_and_load(simple_trace_a):
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)

    try:
        simple_trace_a.save(path)
        restored = AgentTrace.load(path)

        assert restored.run_id == simple_trace_a.run_id
        assert len(restored.nodes) == len(simple_trace_a.nodes)
        assert len(restored.edges) == len(simple_trace_a.edges)

        for orig, rest in zip(simple_trace_a.nodes, restored.nodes, strict=True):
            assert orig.step == rest.step
            assert orig.node_type == rest.node_type
            assert orig.content == rest.content
    finally:
        path.unlink(missing_ok=True)


def test_trace_jsonl_format(simple_trace_a):
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = Path(f.name)

    try:
        simple_trace_a.save(path)
        raw = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        types = [rec["type"] for rec in raw]
        assert types[0] == "trace_meta"
        assert "node" in types
        assert "edge" in types
    finally:
        path.unlink(missing_ok=True)


def test_trace_len(simple_trace_a):
    assert len(simple_trace_a) == 6


def test_node_to_dict_has_all_keys():
    """TraceNode.to_dict() must include all required keys."""
    node = TraceNode(step=2, node_type=NodeType.LLM, content="thinking")
    d = node.to_dict()
    assert "id" in d
    assert d["step"] == 2
    assert d["node_type"] == "llm"
    assert d["content"] == "thinking"
    assert "metadata" in d


def test_edge_to_dict_has_all_keys():
    """TraceEdge.to_dict() must include all required keys."""
    edge = TraceEdge(3, 4, EdgeType.LLM_DECISION, "llm_output")
    d = edge.to_dict()
    assert d["source_step"] == 3
    assert d["target_step"] == 4
    assert d["edge_type"] == "llm_decision"
    assert d["label"] == "llm_output"


def test_node_from_dict_with_metadata():
    """TraceNode.from_dict() must preserve metadata."""
    d = {"step": 5, "node_type": "tool_call", "content": "run(x)", "metadata": {"key": "val"}}
    node = TraceNode.from_dict(d)
    assert node.metadata == {"key": "val"}


def test_node_from_dict_missing_metadata():
    """TraceNode.from_dict() must default metadata to empty dict when absent."""
    d = {"step": 1, "node_type": "start", "content": "begin"}
    node = TraceNode.from_dict(d)
    assert node.metadata == {}


def test_edge_from_dict_missing_label():
    """TraceEdge.from_dict() must default label to empty string when absent."""
    d = {"source_step": 1, "target_step": 2, "edge_type": "sequence"}
    edge = TraceEdge.from_dict(d)
    assert edge.label == ""


def test_trace_load_with_metadata(simple_trace_a):
    """AgentTrace.load() should preserve metadata from the trace_meta line."""
    import tempfile
    trace = AgentTrace(run_id="meta_test", metadata={"version": "1.0", "env": "ci"})
    trace.add_node(TraceNode(step=1, node_type=NodeType.START, content="start"))
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        trace.save(path)
        restored = AgentTrace.load(path)
        assert restored.run_id == "meta_test"
        assert restored.metadata.get("version") == "1.0"
        assert restored.metadata.get("env") == "ci"
    finally:
        path.unlink(missing_ok=True)


def test_trace_load_skips_blank_lines():
    """AgentTrace.load() should not crash on blank lines in the JSONL file."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = Path(f.name)
        f.write('{"type": "trace_meta", "run_id": "blank_test"}\n')
        f.write('\n')  # blank line
        f.write('{"type": "node", "step": 1, "node_type": "start", "content": "hi"}\n')
    try:
        restored = AgentTrace.load(path)
        assert restored.run_id == "blank_test"
        assert len(restored.nodes) == 1
    finally:
        path.unlink(missing_ok=True)


def test_all_node_types_roundtrip():
    """Every NodeType value should survive a to_dict/from_dict roundtrip."""
    for ntype in NodeType:
        node = TraceNode(step=1, node_type=ntype, content=f"content for {ntype.value}")
        restored = TraceNode.from_dict(node.to_dict())
        assert restored.node_type == ntype


def test_all_edge_types_roundtrip():
    """Every EdgeType value should survive a to_dict/from_dict roundtrip."""
    for etype in EdgeType:
        edge = TraceEdge(1, 2, etype, "label")
        restored = TraceEdge.from_dict(edge.to_dict())
        assert restored.edge_type == etype


def test_trace_add_node_and_edge():
    """add_node and add_edge should append to nodes and edges lists respectively."""
    trace = AgentTrace(run_id="t")
    node = TraceNode(step=1, node_type=NodeType.START, content="go")
    edge = TraceEdge(1, 2, EdgeType.SEQUENCE, "")
    trace.add_node(node)
    trace.add_edge(edge)
    assert len(trace.nodes) == 1
    assert len(trace.edges) == 1
