# Python API Reference

## Top-level exports

```python
from redline import AgentTrace, diff_traces, record
```

::: redline.trace.AgentTrace

::: redline.trace.TraceNode

::: redline.trace.TraceEdge

::: redline.trace.NodeType

::: redline.trace.EdgeType

---

## Diff

::: redline.diff.diff_traces

::: redline.diff.DiffResult

::: redline.diff.ForkPoint

::: redline.diff.StepDiff

---

## Embeddings

::: redline.embed.embed_trace

::: redline.embed.align_traces

::: redline.embed.cosine_similarity

::: redline.embed.find_best_match

---

## Instrumentation

::: redline.instrument.record

::: redline.instrument.RedlineCallback

---

## Report

::: redline.report.print_diff

::: redline.report.to_json

::: redline.report.to_markdown
