# Closed loop — `agentdelta`

**Status:** reader wired (eagle-eyes / 2026-08-04)  
**Owner loop:** L4

## Load-bearing job

Behavioral / path diff between agent runs (CI regression)

## Who reads the output?

- Library API: `agentdelta.gate_traces` / `gate_from_disk` / `e2e_reader_after_write` / `assert_no_regression` (`closed_loop.py`)
- CLI: `agentdelta score …` / `agentdelta diff --exit-code`
- CI job or eagle-eyes `dogfood_verify` compares two traces and acts on `exit_code`

## What outcome changes?

PR / gate fails if tool/reasoning path regresses; empty traces → `FAIL_LOUD` (exit 2), never silent pass.
**WRITER-NOT-READER:** path fingerprint diverges + same final answer → never silent PASS; content-swap e2e must re-judge via disk reader.

## When NOT to use (anti-ornament)

Do not call from product agents as free MCP decoration; do not treat as final-answer equality check only

## Non-Ornament checklist

- [x] Reader implemented in CI, gate, or eagle-eyes script (`gate_traces` + tests)
- [x] Empty/wrong output fails loudly (`FAIL_LOUD`, exit 2)
- [x] Not exposed as free MCP in product agents (import/CI gate only)
- [ ] Linked gap IDs in mem0 when improving

## Related failures (farm memory)

- 2026-07-22 MCP buffet trim: write-only tools removed from Foundry framework
- D-FOGHORN: misuse of append-only fact log as current state
- Dual-path mem0: never rely on MCP-only for critical memory

## Daily rotation note

Prefer small daily commits that move remaining checkboxes or raise scorer pillars (G/H/T).

## Auto-run 2026-08-04
- pytest_rc: 1
- node: clawer-samurai-2

## Auto-run 2026-08-04
- pytest_rc: 0
- node: clawer-samurai-2

## Auto-run 2026-08-04
- pytest_rc: 0
- node: clawer-samurai-2

## Auto-run 2026-08-04
- pytest_rc: 0
- node: clawer-samurai-2

## Auto-run 2026-08-04
- pytest_rc: 0
- node: clawer-samurai-2

## Auto-run 2026-08-04
- pytest_rc: 0
- node: clawer-samurai-2

## Auto-run 2026-08-04
- pytest_rc: 0
- node: clawer-samurai-2

## Auto-run 2026-08-04
- pytest_rc: 0
- node: clawer-samurai-2

## Auto-run 2026-08-05
- pytest_rc: 0
- node: clawer-samurai-2

## Auto-run 2026-08-05
- pytest_rc: 0
- node: clawer-samurai-2

## Auto-run 2026-08-05
- pytest_rc: 0
- node: clawer-samurai-2

## Auto-run 2026-08-05
- pytest_rc: 0
- node: clawer-samurai-2

## Auto-run 2026-08-05
- pytest_rc: 0
- node: clawer-samurai-2
