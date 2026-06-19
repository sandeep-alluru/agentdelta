"""Core diff algorithm: align two agent traces and find the first semantic fork."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentdelta.embed import align_traces, embed_trace
from agentdelta.trace import AgentTrace, NodeType, TraceNode


@dataclass
class ForkPoint:
    """The first step where two traces take meaningfully different paths.

    Attributes:
        step_a: Step number in trace A where the fork occurred.
        step_b: Step number in trace B where the fork occurred.
        node_a: The divergent node from trace A.
        node_b: The divergent node from trace B.
        similarity: Cosine similarity between the two nodes at the fork (< fork_threshold).
        description: Human-readable explanation of why this step diverged.
    """

    step_a: int
    step_b: int
    node_a: TraceNode
    node_b: TraceNode
    similarity: float
    description: str

    def is_tool_change(self) -> bool:
        """Return True if the fork is a tool-selection or tool-return change."""
        return (
            self.node_a.node_type in (NodeType.TOOL_CALL, NodeType.TOOL_RETURN)
            and self.node_b.node_type in (NodeType.TOOL_CALL, NodeType.TOOL_RETURN)
        )

    def is_reasoning_change(self) -> bool:
        """Return True if the fork is an LLM reasoning divergence."""
        return (
            self.node_a.node_type == NodeType.LLM
            and self.node_b.node_type == NodeType.LLM
        )


@dataclass
class StepDiff:
    """A single aligned step pair with its comparison result.

    Attributes:
        step_a: Node from trace A, or ``None`` if this step was added in B.
        step_b: Node from trace B, or ``None`` if this step was removed in A.
        similarity: Cosine similarity between the two nodes (0.0 for added/removed).
        status: One of ``"match"``, ``"changed"``, ``"added"``, or ``"removed"``.
        summary: Human-readable one-line description of this diff entry.
    """

    step_a: TraceNode | None
    step_b: TraceNode | None
    similarity: float
    status: str
    summary: str = ""


@dataclass
class DiffResult:
    """Full diff result between two agent traces.

    Attributes:
        run_id_a: Run identifier of the baseline trace.
        run_id_b: Run identifier of the candidate trace.
        steps: All aligned step pairs, in order.
        fork_point: The first divergent step, or ``None`` if the traces are equivalent.
        summary: Pre-computed aggregate statistics (total, matched, changed, etc.).
    """

    run_id_a: str
    run_id_b: str
    steps: list[StepDiff] = field(default_factory=list)
    fork_point: ForkPoint | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def has_regression(self) -> bool:
        """True if the traces diverged (a fork point was detected)."""
        fp = self.fork_point is not None
        counts = self.summary
        return fp or (counts.get("total", 0) > 0 and counts.get("matched", 0) == 0)

    @property
    def changed_steps(self) -> list[StepDiff]:
        """Steps where both traces have a node but they diverged semantically."""
        return [s for s in self.steps if s.status == "changed"]

    @property
    def added_steps(self) -> list[StepDiff]:
        """Steps present only in trace B (inserted relative to baseline)."""
        return [s for s in self.steps if s.status == "added"]

    @property
    def removed_steps(self) -> list[StepDiff]:
        """Steps present only in trace A (removed relative to baseline)."""
        return [s for s in self.steps if s.status == "removed"]


def _describe_fork(na: TraceNode, nb: TraceNode, similarity: float) -> str:
    if na.node_type == NodeType.TOOL_CALL and nb.node_type == NodeType.TOOL_CALL:
        tool_a = na.content.split("(")[0].strip()
        tool_b = nb.content.split("(")[0].strip()
        if tool_a != tool_b:
            return f"Tool selection changed: '{tool_a}' → '{tool_b}'"
        return f"Tool call arguments diverged (similarity: {similarity:.2f})"
    if na.node_type == NodeType.LLM and nb.node_type == NodeType.LLM:
        return f"Reasoning path diverged (similarity: {similarity:.2f})"
    return (
        f"Step type changed: {na.node_type.value} → {nb.node_type.value} "
        f"(similarity: {similarity:.2f})"
    )


def diff_traces(
    trace_a: AgentTrace,
    trace_b: AgentTrace,
    fork_threshold: float = 0.70,
    match_threshold: float = 0.85,
) -> DiffResult:
    """
    Compute a semantic diff between two agent traces.

    Args:
        trace_a: Baseline trace.
        trace_b: Comparison trace.
        fork_threshold: Similarity below this triggers a fork point.
        match_threshold: Similarity above this is considered a match.

    Returns:
        DiffResult with aligned steps and the first fork point if found.
    """
    # Ensure both traces are embedded
    embed_trace(trace_a)
    embed_trace(trace_b)

    alignment = align_traces(trace_a, trace_b, threshold=fork_threshold)

    steps: list[StepDiff] = []
    fork_point: ForkPoint | None = None

    for na, nb, score in alignment:
        if na is None:
            if nb is None:
                continue
            summary = f"+ [{nb.node_type.value}] {nb.content[:80]}"
            steps.append(StepDiff(None, nb, 0.0, "added", summary))
        elif nb is None:
            summary = f"- [{na.node_type.value}] {na.content[:80]}"
            steps.append(StepDiff(na, None, 0.0, "removed", summary))
        elif score >= match_threshold:
            steps.append(StepDiff(na, nb, score, "match"))
        else:
            desc = _describe_fork(na, nb, score)
            step = StepDiff(na, nb, score, "changed", desc)
            steps.append(step)
            # Record the first fork point
            if fork_point is None:
                fork_point = ForkPoint(
                    step_a=na.step,
                    step_b=nb.step,
                    node_a=na,
                    node_b=nb,
                    similarity=score,
                    description=desc,
                )

    total = len(alignment)
    matched = sum(1 for s in steps if s.status == "match")
    changed = sum(1 for s in steps if s.status == "changed")

    result = DiffResult(
        run_id_a=trace_a.run_id,
        run_id_b=trace_b.run_id,
        steps=steps,
        fork_point=fork_point,
        summary={
            "total_steps": total,
            "matched": matched,
            "changed": changed,
            "added": len([s for s in steps if s.status == "added"]),
            "removed": len([s for s in steps if s.status == "removed"]),
            "similarity_pct": round(matched / total * 100, 1) if total else 100.0,
            "has_regression": fork_point is not None or (total > 0 and matched == 0),
            "fork_step": fork_point.step_a if fork_point else None,
        },
    )
    return result
