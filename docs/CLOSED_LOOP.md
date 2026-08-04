# Closed loop — `agentdelta`

**Status:** stub (eagle-eyes Phase 0 / 2026-08-04)  
**Owner loop:** L4

## Load-bearing job

Behavioral / path diff between agent runs (CI regression)

## Who reads the output?

CI job or eagle-eyes verify step compares two traces

## What outcome changes?

PR fails or alert if tool/reasoning path regresses

## When NOT to use (anti-ornament)

Do not call from product agents as free MCP decoration; do not treat as final-answer equality check only

## Non-Ornament checklist

- [ ] Reader implemented in CI, gate, or eagle-eyes script
- [ ] Empty/wrong output fails loudly
- [ ] Not exposed as free MCP in product agents
- [ ] Linked gap IDs in mem0 when improving

## Related failures (farm memory)

- 2026-07-22 MCP buffet trim: write-only tools removed from Foundry framework
- D-FOGHORN: misuse of append-only fact log as current state
- Dual-path mem0: never rely on MCP-only for critical memory

## Daily rotation note

This file exists so pillar **C (closed loop)** can rise with real wiring over time. Prefer small daily commits that move a checkbox toward done.

## Auto-run 2026-08-04
- pytest_rc: 1
- node: clawer-samurai-2
