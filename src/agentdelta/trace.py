"""Core trace data model for agentdelta."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class NodeType(str, Enum):
    """Classification of a single step in an agent trace."""

    START = "start"
    LLM = "llm"
    TOOL_CALL = "tool_call"
    TOOL_RETURN = "tool_return"
    END = "end"


class EdgeType(str, Enum):
    """Classification of a directed edge between two trace nodes."""

    LLM_DECISION = "llm_decision"
    TOOL_CALL = "tool_call"
    TOOL_RETURN = "tool_return"
    SEQUENCE = "sequence"


@dataclass
class TraceNode:
    """A single step in an agent execution trace.

    Attributes:
        step: 1-based sequential position in the trace.
        node_type: Category of this step (LLM reasoning, tool call, etc.).
        content: Human-readable text — reasoning output, ``tool(args)``, or tool return value.
        metadata: Arbitrary key/value pairs for framework-specific data.
        embedding: Floating-point sentence embedding, populated by ``embed_trace()``.
    """

    step: int
    node_type: NodeType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = field(default=None, repr=False)

    @property
    def id(self) -> str:
        """Content-addressed ID — same content always produces the same ID."""
        payload = f"{self.node_type}:{self.content}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "id": self.id,
            "step": self.step,
            "node_type": self.node_type.value,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceNode:
        """Deserialise from a plain dict (as produced by ``to_dict``)."""
        return cls(
            step=d["step"],
            node_type=NodeType(d["node_type"]),
            content=d["content"],
            metadata=d.get("metadata", {}),
        )


@dataclass
class TraceEdge:
    """A directed connection between two steps in a trace.

    Attributes:
        source_step: Step number of the originating node.
        target_step: Step number of the destination node.
        edge_type: Semantic category of this transition.
        label: Optional human-readable label (tool name, phase, etc.).
    """

    source_step: int
    target_step: int
    edge_type: EdgeType
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "source_step": self.source_step,
            "target_step": self.target_step,
            "edge_type": self.edge_type.value,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceEdge:
        """Deserialise from a plain dict (as produced by ``to_dict``)."""
        return cls(
            source_step=d["source_step"],
            target_step=d["target_step"],
            edge_type=EdgeType(d["edge_type"]),
            label=d.get("label", ""),
        )


@dataclass
class AgentTrace:
    """Complete execution trace for a single agent run.

    Attributes:
        run_id: Unique identifier for this run (e.g. ``"v1.0"``).
        nodes: Ordered list of trace steps.
        edges: Directed edges connecting steps.
        metadata: Arbitrary key/value pairs stored in the trace header.
    """

    run_id: str
    nodes: list[TraceNode] = field(default_factory=list)
    edges: list[TraceEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: TraceNode) -> None:
        """Append a node to the trace."""
        self.nodes.append(node)

    def add_edge(self, edge: TraceEdge) -> None:
        """Append an edge to the trace."""
        self.edges.append(edge)

    def save(self, path: str | Path) -> None:
        """Write the trace to a JSONL file at *path* (one record per line)."""
        path = Path(path)
        with path.open("w") as f:
            meta = {"type": "trace_meta", "run_id": self.run_id, **self.metadata}
            f.write(json.dumps(meta) + "\n")
            for node in self.nodes:
                f.write(json.dumps({"type": "node", **node.to_dict()}) + "\n")
            for edge in self.edges:
                f.write(json.dumps({"type": "edge", **edge.to_dict()}) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> AgentTrace:
        """Load a trace from a JSONL file previously written by ``save``."""
        path = Path(path)
        nodes: list[TraceNode] = []
        edges: list[TraceEdge] = []
        run_id = path.stem
        metadata: dict[str, Any] = {}

        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rtype = record.pop("type")
                if rtype == "trace_meta":
                    run_id = record.pop("run_id", run_id)
                    metadata = record
                elif rtype == "node":
                    nodes.append(TraceNode.from_dict(record))
                elif rtype == "edge":
                    edges.append(TraceEdge.from_dict(record))

        return cls(run_id=run_id, nodes=nodes, edges=edges, metadata=metadata)

    def __len__(self) -> int:
        return len(self.nodes)
