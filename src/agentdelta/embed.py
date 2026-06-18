"""Embedding and semantic alignment for trace nodes."""

from __future__ import annotations

import threading

import numpy as np

from agentdelta.trace import AgentTrace, TraceNode

_lock = threading.Lock()
_model = None


def _get_model():
    """Return the sentence-transformer singleton, initialising it on first call (thread-safe)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_trace(trace: AgentTrace, batch_size: int = 64) -> AgentTrace:
    """Compute embeddings for all nodes in a trace (in-place) and return the trace."""
    model = _get_model()
    contents = [node.content for node in trace.nodes]
    if not contents:
        return trace
    embeddings = model.encode(contents, batch_size=batch_size, show_progress_bar=False)
    for node, emb in zip(trace.nodes, embeddings, strict=False):
        node.embedding = emb.tolist()
    return trace


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity between two embedding vectors. Returns 0.0 for zero vectors."""
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def find_best_match(
    node: TraceNode,
    candidates: list[TraceNode],
    threshold: float = 0.75,
) -> tuple[TraceNode | None, float]:
    """Find the candidate most semantically similar to *node*.

    Returns ``(best_node, score)``. If the best score is below *threshold*,
    returns ``(None, best_score)`` rather than a low-confidence match.
    """
    if node.embedding is None or not candidates:
        return None, 0.0

    best_node, best_score = None, -1.0
    for candidate in candidates:
        if candidate.embedding is None:
            continue
        score = cosine_similarity(node.embedding, candidate.embedding)
        if score > best_score:
            best_score = score
            best_node = candidate

    if best_score < threshold:
        return None, best_score
    return best_node, best_score


def align_traces(
    trace_a: AgentTrace,
    trace_b: AgentTrace,
    window: int = 5,
    threshold: float = 0.75,
) -> list[tuple[TraceNode | None, TraceNode | None, float]]:
    """Align nodes from two traces by semantic similarity within a sliding window.

    Uses greedy 1:1 matching: each node in *trace_a* is paired with the
    closest unmatched node in *trace_b* within ±*window* positions.

    Returns:
        List of ``(node_a, node_b, similarity)`` triples.
        Unmatched nodes appear as ``(node, None, 0.0)`` or ``(None, node, 0.0)``.
    """
    nodes_a = trace_a.nodes
    nodes_b = trace_b.nodes

    alignment: list[tuple[TraceNode | None, TraceNode | None, float]] = []
    used_b: set[int] = set()
    node_to_idx: dict[int, int] = {id(nb): j for j, nb in enumerate(nodes_b)}

    for i, na in enumerate(nodes_a):
        start = max(0, i - window)
        end = min(len(nodes_b), i + window + 1)
        candidates = [nodes_b[j] for j in range(start, end) if j not in used_b]

        match, score = find_best_match(na, candidates, threshold)
        if match is not None:
            used_b.add(node_to_idx[id(match)])
            alignment.append((na, match, score))
        else:
            alignment.append((na, None, 0.0))

    for j, nb in enumerate(nodes_b):
        if j not in used_b:
            alignment.append((None, nb, 0.0))

    return alignment
