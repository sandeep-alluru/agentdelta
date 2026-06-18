# GitHub Action

Use redline in your GitHub Actions workflow to automatically detect behavioral regressions on every PR.

## Setup

### Option A — Use the prebuilt action

```yaml
# .github/workflows/agent-regression.yml
name: Agent behavioral diff

on: [pull_request]

jobs:
  diff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run your agent (baseline)
        run: python scripts/run_agent.py --output traces/baseline.jsonl

      - name: Run your agent (candidate)
        run: python scripts/run_agent.py --output traces/candidate.jsonl

      - name: Behavioral diff
        uses: sandeep-alluru/redline@v0.1.0
        with:
          baseline: traces/baseline.jsonl
          candidate: traces/candidate.jsonl
          fail-on-regression: "true"

      - name: Post diff as PR comment
        if: always()
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          path: redline-diff.md
```

### Option B — Use the CLI directly

```yaml
- name: Install redline
  run: pip install redline

- name: Behavioral diff
  run: |
    redline diff traces/baseline.jsonl traces/candidate.jsonl \
      --format markdown --exit-code > diff.md

- name: Post comment
  uses: marocchino/sticky-pull-request-comment@v2
  with:
    path: diff.md
```

## Action inputs

| Input | Required | Default | Description |
|---|---|---|---|
| `baseline` | ✅ | — | Path to the baseline `.jsonl` trace |
| `candidate` | ✅ | — | Path to the candidate `.jsonl` trace |
| `fail-on-regression` | ❌ | `"false"` | Set to `"true"` to fail the workflow if regression detected |
| `fork-threshold` | ❌ | `"0.70"` | Similarity below this marks a fork point |
| `match-threshold` | ❌ | `"0.85"` | Similarity above this is a match |
| `format` | ❌ | `"markdown"` | Output format: `rich`, `json`, or `markdown` |

## Action outputs

| Output | Description |
|---|---|
| `has-regression` | `"true"` or `"false"` |
| `fork-step` | Step number of the first fork (or empty) |
| `similarity-pct` | Percentage of steps that matched |
| `diff-file` | Path to the generated `redline-diff.md` |

## Storing baseline traces

A typical pattern: store baseline traces in your repo and update them manually when you intentionally change behavior.

```bash
# After a verified good run, commit the baseline
cp traces/current.jsonl traces/baseline.jsonl
git add traces/baseline.jsonl
git commit -m "chore: update behavioral baseline for v1.2"
```

The CI always compares the current run against the committed baseline.
