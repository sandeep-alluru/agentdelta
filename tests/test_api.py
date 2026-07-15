"""Tests for the agentdelta FastAPI REST server."""

from __future__ import annotations

from fastapi.testclient import TestClient

from agentdelta.api import app
from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode

client = TestClient(app)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_trace_jsonl(run_id: str, steps: list[tuple[NodeType, str]]) -> str:
    """Build an in-memory JSONL trace string."""
    trace = AgentTrace(run_id=run_id)
    for i, (ntype, content) in enumerate(steps, start=1):
        trace.add_node(TraceNode(step=i, node_type=ntype, content=content))
        if i > 1:
            trace.add_edge(TraceEdge(i - 1, i, EdgeType.SEQUENCE, ""))

    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        trace.save(f.name)
        tmp = f.name
    content_str = Path(tmp).read_text()
    Path(tmp).unlink(missing_ok=True)
    return content_str


_SIMPLE_STEPS: list[tuple[NodeType, str]] = [
    (NodeType.START, "What is 2+2?"),
    (NodeType.LLM, "The answer is four."),
    (NodeType.END, "four"),
]

_TRACE_A_JSONL = _make_trace_jsonl("api_a", _SIMPLE_STEPS)
_TRACE_B_JSONL = _make_trace_jsonl("api_b", _SIMPLE_STEPS)


# ── /health ───────────────────────────────────────────────────────────────────


def test_health_returns_200() -> None:
    """GET /health should return HTTP 200."""
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_response_shape() -> None:
    """GET /health response must contain 'status' and 'version'."""
    resp = client.get("/health")
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


# ── /diff ─────────────────────────────────────────────────────────────────────


def test_diff_identical_traces_no_regression() -> None:
    """POST /diff with identical traces should return has_regression=False."""
    resp = client.post("/diff", json={"trace_a": _TRACE_A_JSONL, "trace_b": _TRACE_B_JSONL})
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert data["summary"]["has_regression"] is False


def test_diff_response_has_steps() -> None:
    """POST /diff response must contain a 'steps' list."""
    resp = client.post("/diff", json={"trace_a": _TRACE_A_JSONL, "trace_b": _TRACE_B_JSONL})
    assert resp.status_code == 200
    data = resp.json()
    assert "steps" in data
    assert isinstance(data["steps"], list)


def test_diff_invalid_trace_a() -> None:
    """POST /diff with invalid trace_a should return 422."""
    resp = client.post("/diff", json={"trace_a": "not valid jsonl!!!!", "trace_b": _TRACE_B_JSONL})
    assert resp.status_code == 422


def test_diff_invalid_trace_b() -> None:
    """POST /diff with invalid trace_b should return 422."""
    resp = client.post("/diff", json={"trace_a": _TRACE_A_JSONL, "trace_b": "garbage"})
    assert resp.status_code == 422


def test_diff_accepts_custom_thresholds() -> None:
    """POST /diff should accept custom fork_threshold and match_threshold."""
    resp = client.post(
        "/diff",
        json={
            "trace_a": _TRACE_A_JSONL,
            "trace_b": _TRACE_B_JSONL,
            "fork_threshold": 0.5,
            "match_threshold": 0.9,
        },
    )
    assert resp.status_code == 200


# ── /inspect ──────────────────────────────────────────────────────────────────


def test_inspect_returns_200() -> None:
    """POST /inspect with a valid trace should return HTTP 200."""
    resp = client.post("/inspect", json={"trace": _TRACE_A_JSONL})
    assert resp.status_code == 200


def test_inspect_response_shape() -> None:
    """POST /inspect response must contain run_id, total_nodes, and steps."""
    resp = client.post("/inspect", json={"trace": _TRACE_A_JSONL})
    data = resp.json()
    assert "run_id" in data
    assert "total_nodes" in data
    assert "steps" in data
    assert isinstance(data["steps"], list)


def test_inspect_node_type_counts() -> None:
    """POST /inspect response must include node_type_counts dict."""
    resp = client.post("/inspect", json={"trace": _TRACE_A_JSONL})
    data = resp.json()
    assert "node_type_counts" in data
    assert isinstance(data["node_type_counts"], dict)


def test_inspect_invalid_trace() -> None:
    """POST /inspect with invalid trace content should return 422."""
    resp = client.post("/inspect", json={"trace": "this is not jsonl at all"})
    assert resp.status_code == 422


def test_inspect_has_tool_calls_field() -> None:
    """POST /inspect response must include has_tool_calls boolean."""
    resp = client.post("/inspect", json={"trace": _TRACE_A_JSONL})
    data = resp.json()
    assert "has_tool_calls" in data
    assert isinstance(data["has_tool_calls"], bool)
