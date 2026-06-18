"""Tests for redline.embed."""

from __future__ import annotations

import pytest

from redline.embed import align_traces, cosine_similarity, embed_trace, find_best_match


def test_cosine_similarity_identical():
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_embed_trace_fills_embeddings(simple_trace_a):
    result = embed_trace(simple_trace_a)
    for node in result.nodes:
        assert node.embedding is not None
        assert len(node.embedding) > 0


def test_embed_trace_idempotent(simple_trace_a):
    embed_trace(simple_trace_a)
    first_emb = list(simple_trace_a.nodes[0].embedding)
    embed_trace(simple_trace_a)
    assert simple_trace_a.nodes[0].embedding == first_emb


def test_find_best_match_returns_none_below_threshold(simple_trace_a):
    embed_trace(simple_trace_a)
    node = simple_trace_a.nodes[0]
    # threshold=1.1 is impossible — nothing matches
    match, _score = find_best_match(node, simple_trace_a.nodes[1:], threshold=1.1)
    assert match is None


def test_align_traces_same_length(simple_trace_a, simple_trace_b_match):
    embed_trace(simple_trace_a)
    embed_trace(simple_trace_b_match)
    alignment = align_traces(simple_trace_a, simple_trace_b_match)
    assert len(alignment) >= len(simple_trace_a.nodes)


def test_align_traces_all_scored(simple_trace_a, simple_trace_b_match):
    embed_trace(simple_trace_a)
    embed_trace(simple_trace_b_match)
    alignment = align_traces(simple_trace_a, simple_trace_b_match)
    for _na, _nb, score in alignment:
        assert score >= 0.0
        assert score <= 1.0 + 1e-6  # numerical tolerance
