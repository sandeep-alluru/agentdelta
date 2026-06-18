"""Tests for redline.trace."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from redline.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode


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
