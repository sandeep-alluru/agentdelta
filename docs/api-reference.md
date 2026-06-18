# Python API Reference

## Top-level exports

```python
from agentdelta import AgentTrace, diff_traces, record
```

::: agentdelta.trace.AgentTrace

::: agentdelta.trace.TraceNode

::: agentdelta.trace.TraceEdge

::: agentdelta.trace.NodeType

::: agentdelta.trace.EdgeType

---

## Diff

::: agentdelta.diff.diff_traces

::: agentdelta.diff.DiffResult

::: agentdelta.diff.ForkPoint

::: agentdelta.diff.StepDiff

---

## Embeddings

::: agentdelta.embed.embed_trace

::: agentdelta.embed.align_traces

::: agentdelta.embed.cosine_similarity

::: agentdelta.embed.find_best_match

---

## Instrumentation

::: agentdelta.instrument.record

::: agentdelta.instrument.AgentdeltaCallback

---

## Report

::: agentdelta.report.print_diff

::: agentdelta.report.to_json

::: agentdelta.report.to_markdown
