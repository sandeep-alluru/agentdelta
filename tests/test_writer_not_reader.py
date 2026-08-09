"""WRITER-NOT-READER - reader e2e that fails if only the writer path is fixed.

Farm lesson (Qdrant): cache fixes often update the writer key and never trace
readers. A content swap must re-judge via the disk reader, not in-memory only.

Also maps public DiagChain (arXiv 2608.03591): intermediate stages matter;
final-answer-only evaluation hides path divergence.
"""

from __future__ import annotations

from pathlib import Path

from agentdelta.closed_loop import (
    answer_fingerprint,
    e2e_content_swap_rejudges,
    e2e_reader_after_write,
    gate_from_disk,
    gate_traces,
    path_fingerprint,
)
from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode


def _make_trace(
    run_id: str,
    tool: str,
    end: str = "18C cloudy",
    llm: str = "I will call a tool",
) -> AgentTrace:
    trace = AgentTrace(run_id=run_id)
    steps: list[tuple[NodeType, str]] = [
        (NodeType.START, "Check weather in London"),
        (NodeType.LLM, llm),
        (NodeType.TOOL_CALL, tool),
        (NodeType.TOOL_RETURN, '{"temp": 18}'),
        (NodeType.END, end),
    ]
    for i, (ntype, content) in enumerate(steps, start=1):
        trace.add_node(TraceNode(step=i, node_type=ntype, content=content))
        if i > 1:
            trace.add_edge(TraceEdge(i - 1, i, EdgeType.SEQUENCE, ""))
    return trace


def test_path_fingerprint_differs_when_tool_changes_answer_same() -> None:
    """Collapse trap: END matches, full path must not."""
    a = _make_trace("a", "get_weather(location='London')")
    b = _make_trace("b", "web_search(query='London weather')")
    assert answer_fingerprint(a) == answer_fingerprint(b)
    assert path_fingerprint(a) != path_fingerprint(b)


def test_path_fingerprint_stable_for_identical_path() -> None:
    a = _make_trace("a", "get_weather(location='London')")
    b = _make_trace("b", "get_weather(location='London')")
    assert path_fingerprint(a) == path_fingerprint(b)
    assert answer_fingerprint(a) == answer_fingerprint(b)


def test_gate_marks_regression_when_path_diverges_same_answer() -> None:
    """Aligner may not set fork_point on tool add/remove; path id must."""
    a = _make_trace("baseline", "get_weather(location='London')")
    b = _make_trace("candidate", "web_search(query='London weather')")
    out = gate_traces(a, b, pass_threshold=95.0, warn_threshold=90.0, warn_is_ok=False)
    assert out.has_regression is True
    assert out.path_fingerprint_a != out.path_fingerprint_b
    assert out.answer_fingerprint_a == out.answer_fingerprint_b
    assert out.ok is False
    assert out.exit_code in {1, 2}
    assert out.verdict in {"FAIL", "WARN", "FAIL_LOUD"}


def test_e2e_reader_after_write_observes_disk(tmp_path: Path) -> None:
    """Writer alone is insufficient - gate must go through disk reload."""
    a = _make_trace("baseline", "get_weather(location='London')")
    b = _make_trace("candidate", "get_weather(location='London')")
    out = e2e_reader_after_write(a, b, tmp_path)
    assert (tmp_path / "baseline.jsonl").is_file()
    assert (tmp_path / "candidate.jsonl").is_file()
    assert out.ok is True
    assert out.verdict == "PASS"
    assert out.path_fingerprint_a == out.path_fingerprint_b


def test_e2e_reader_after_write_fails_on_tool_swap(tmp_path: Path) -> None:
    a = _make_trace("baseline", "get_weather(location='London')")
    b = _make_trace("candidate", "web_search(query='London weather')")
    out = e2e_reader_after_write(
        a, b, tmp_path, pass_threshold=95.0, warn_threshold=90.0, warn_is_ok=False
    )
    assert out.ok is False
    assert out.has_regression is True


def test_content_swap_rejudges_via_reader(tmp_path: Path) -> None:
    """Overwriting the candidate file (same cache key) must change gate outcome.

    PRE-FIX class of bug: writer updates bytes; reader still uses old in-memory
    or collapsed (bid,name) key → no re-judge. POST-FIX: gate_from_disk reloads.
    """
    baseline = _make_trace("baseline", "get_weather(location='London')")
    same = _make_trace("cand_same", "get_weather(location='London')")
    swapped = _make_trace("cand_swap", "web_search(query='London weather')")

    before, after = e2e_content_swap_rejudges(
        baseline,
        same,
        swapped,
        tmp_path,
        pass_threshold=95.0,
        warn_threshold=90.0,
        warn_is_ok=False,
    )
    assert before.ok is True
    assert before.verdict == "PASS"
    assert after.ok is False
    assert after.has_regression is True
    assert after.path_fingerprint_a != after.path_fingerprint_b
    # Same final answer - the collapse trap that unit-level answer checks miss.
    assert after.answer_fingerprint_a == after.answer_fingerprint_b


def test_gate_from_disk_missing_file_fails_loud(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    other = tmp_path / "ok.jsonl"
    _make_trace("ok", "get_weather(location='London')").save(other)
    out = gate_from_disk(missing, other)
    assert out.verdict == "FAIL_LOUD"
    assert out.exit_code == 2
    assert "not found" in out.reason.lower()


def test_answer_only_would_wrongly_pass() -> None:
    """Document the trap: answer_fingerprint alone is not a gate."""
    a = _make_trace("a", "get_weather(location='London')")
    b = _make_trace("b", "web_search(query='London weather')")
    # Broken "writer-only / answer-only" check would PASS here:
    assert answer_fingerprint(a) == answer_fingerprint(b)
    # Load-bearing product check must not:
    out = gate_traces(a, b, pass_threshold=99.0, warn_threshold=98.0, warn_is_ok=False)
    assert out.ok is False
    assert path_fingerprint(a) != path_fingerprint(b)


def test_to_dict_includes_fingerprints() -> None:
    a = _make_trace("a", "get_weather(location='London')")
    b = _make_trace("b", "get_weather(location='London')")
    payload = gate_traces(a, b).to_dict()
    assert payload["path_fingerprint_a"]
    assert payload["path_fingerprint_a"] == payload["path_fingerprint_b"]
    assert payload["answer_fingerprint_a"] == payload["answer_fingerprint_b"]
