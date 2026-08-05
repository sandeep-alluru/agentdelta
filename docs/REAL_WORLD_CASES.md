# Real-world cases driving agentdelta

Mined from farm_memory (Qdrant) and public research (eagle-eyes Track B).

## Case WRITER-NOT-READER (farm) — CRITICAL

**Source:** Qdrant `farm_memory` lesson on cache keys (Pioneer Content Foundry /
footage content-id); eagle-eyes `REAL_WORK_QUEUE` P0; related to dual-path
memory and “writer fixed, readers not.”

**What failed (class):**

Integrators fixed the **writer** (correct cache key, correct bytes written to
disk / store) and never traced **readers**. Downstream gates still collapsed
identity to a coarse key — e.g. `(bid, name)` for footage, or **final answer
only** for agent runs — so a real content swap did not re-judge.

Measured pattern (Foundry): re-sweep of footage content-id was defeated because
the gate and a shared name-collapser ignored the new content identity.
Lesson text: *Verify END-TO-END (a content swap must re-judge AND re-select),
not unit-level.*

**What fails for agent traces specifically:**

1. **Answer-only equality:** same `END` content, different `TOOL_CALL` path →
   silent “looks fine” if the gate only checks final output.
2. **In-memory writer unit tests:** `trace.save` tested; CI never reloads JSONL.
3. **Stale reader path:** overwrite candidate file; gate still uses old object.

**Product fix in this repo:**

| Control | API |
|---------|-----|
| Full path identity | `path_fingerprint(trace)` |
| Collapse trap (END only) | `answer_fingerprint(trace)` — documented insufficient |
| Disk reader | `gate_from_disk(baseline_path, candidate_path)` |
| Writer→reader e2e | `e2e_reader_after_write(baseline, candidate, work_dir)` |
| Content-swap re-judge | `e2e_content_swap_rejudges(...)` |
| Answer-only refuse | `gate_traces` → never PASS when path diverges & answer matches |

**Tests:** `tests/test_writer_not_reader.py`

**Non-Ornament:** CI must call `gate_traces` / `gate_from_disk` on **files** (or
e2e after write), not only assert that a callback populated an in-memory
`AgentTrace`.

---

## Case DiagChain (public) — intermediate stages

**Source:** arXiv [2608.03591](https://arxiv.org/abs/2608.03591v1) — *DiagChain:
A Diagnostic Benchmark for Evaluating LLM Agents on Evidence-Grounded Attack
Chain Reconstruction* (eagle-eyes research session 20260805T041217Z).

**Gap:** Benchmarks that score only final outputs miss how errors arise and
propagate across intermediate reasoning / tool stages.

**Product mapping:** agentdelta step-level `diff_traces` + `gate_traces` with
path fingerprints (WRITER-NOT-READER) — behavioral CI, not answer equality.

---

## Related queue IDs

- **WRITER-NOT-READER** — this case (P0)
- **SILENT-SUCCESS** — degraded exit-0 (notarize / groundcrew)
- D-FOGHORN (foghorn) — oldest-as-current reader bug (sibling reader discipline)
