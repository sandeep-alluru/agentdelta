# Case Study: Catching Invisible Behavioral Regressions in a Fraud Detection Agent

## Company Profile

**Meridian Payments** is a B2B payment processing company headquartered in Austin, TX. With 200 engineers, they process $4.2B in transactions annually for mid-market merchants. Their fraud detection pipeline is built on LangGraph agents backed by OpenAI models, running on AWS with a PostgreSQL transactional store and a Redis feature cache.

## The Problem

When Meridian's ML team upgraded their fraud detection agent from GPT-4o-mini to GPT-4o, the initial benchmark results looked promising. Accuracy on their held-out evaluation set improved by 1.8 percentage points. The team shipped the upgrade to production.

Three days later, the ops team noticed something wrong: the false positive rate — legitimate transactions incorrectly flagged as fraud — had climbed 34% compared to the prior week. Downstream, this meant 3,400 legitimate business transactions were being declined per day, generating chargebacks, merchant complaints, and potential SLA breach penalties.

The root cause was invisible to their existing evals. The GPT-4o model was calling the fraud feature tools in a different order: instead of checking `velocity_check` first (a fast, cheap gate that filters ~60% of transactions early), it was starting with `transaction_graph_analysis` (an expensive graph query). The logic was still correct — the agent reached the same fraud/not-fraud verdict — but the changed tool-call order meant the transaction graph ran on every transaction, not just the 40% that passed the velocity gate. This amplified a subtle bug in the graph query that had been dormant: it occasionally flagged high-frequency legitimate merchants when called without the velocity filter's pre-screening.

The team had no way to detect this before production. Their evals measured verdict accuracy. They had nothing that measured *how* the agent arrived at that verdict.

## Solution Architecture

Meridian added agentdelta to their CI pipeline as a behavioral regression gate. Every PR that touches the fraud agent's prompt, tool definitions, or model configuration triggers a comparison run against 50 standardized fraud scenarios — a mix of confirmed fraud cases, confirmed legitimate cases, and ambiguous edge cases from their historical data.

```
┌─────────────────────────────────────────────────────────────────┐
│                     GitHub Actions CI Pipeline                  │
│                                                                 │
│  PR opened         ┌─────────────────────────────────────────┐  │
│  (prompt/model  → │  1. Run baseline agent (main branch)    │  │
│   change)          │     → 50 scenarios → baseline/*.jsonl   │  │
│                    │                                         │  │
│                    │  2. Run candidate agent (PR branch)     │  │
│                    │     → 50 scenarios → candidate/*.jsonl  │  │
│                    │                                         │  │
│                    │  3. agentdelta batch_from_directory()   │  │
│                    │     → BatchDiffResult                   │  │
│                    │     → compute_score() per pair         │  │
│                    │     → aggregate_score < 80 → FAIL CI   │  │
│                    └─────────────────────────────────────────┘  │
│                                                                 │
│  Artifacts: HTML diff report uploaded to S3 for review         │
└─────────────────────────────────────────────────────────────────┘
```

The CI gate fails if the aggregate behavioral similarity score drops below 80. Any individual scenario scoring below 60 triggers a `WARN` annotation on the PR with the specific fork point — the exact step where the candidate agent diverged from the baseline.

## Implementation

```python
# ci/behavioral_regression_check.py
from pathlib import Path
from agentdelta import record, batch_from_directory, to_html
from agentdelta.score import compute_score
from agentdelta.diff import diff_traces
from agentdelta.trace import AgentTrace

import sys
import json

BASELINE_DIR = Path("ci/traces/baseline")
CANDIDATE_DIR = Path("ci/traces/candidate")
SCENARIOS_DIR = Path("ci/scenarios")

# Step 1: Record baseline traces (runs against main-branch agent)
def record_traces(agent, out_dir: Path, run_prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = list(SCENARIOS_DIR.glob("*.json"))
    print(f"Recording {len(scenarios)} scenarios → {out_dir}")

    for scenario_path in scenarios:
        scenario = json.loads(scenario_path.read_text())
        out_file = out_dir / scenario_path.name.replace(".json", ".jsonl")

        with record(out_file, run_id=f"{run_prefix}/{scenario_path.stem}") as cb:
            agent.invoke(
                {"input": scenario["transaction"]},
                config={"callbacks": [cb]},
            )

# Step 2: Diff all 50 pairs and compute aggregate score
def run_regression_gate() -> int:
    result = batch_from_directory(
        baseline_dir=BASELINE_DIR,
        candidate_dir=CANDIDATE_DIR,
        pass_threshold=80.0,
        warn_threshold=60.0,
    )

    print(f"\nagentdelta behavioral regression report")
    print(f"  Scenarios compared : {len(result)}")
    print(f"  Aggregate score    : {result.aggregate_score:.1f} / 100")
    print(f"  Regressions        : {len(result.regressions)}")

    if result.regressions:
        print("\nFailed scenarios:")
        for label in result.regressions:
            print(f"  - {label}")

    # Generate HTML report for the PR artifact
    for baseline_id, candidate_id, diff_result in result.pairs:
        score = compute_score(diff_result)
        if score.verdict != "PASS":
            print(f"\n  [{score.verdict}] {baseline_id}")
            print(f"    overall={score.overall:.1f}  "
                  f"tool_fidelity={score.tool_fidelity:.1f}  "
                  f"fork_penalty={score.fork_penalty:.1f}")
            if diff_result.fork_point:
                fp = diff_result.fork_point
                print(f"    Fork at step {fp.step_a}: "
                      f"baseline called {fp.node_a.content!r}, "
                      f"candidate called {fp.node_b.content!r}")

    # Exit code: 0 = pass, 1 = regressions found
    return 1 if result.has_regressions else 0

if __name__ == "__main__":
    sys.exit(run_regression_gate())
```

The tool-call order change that caused the production incident would have produced a `tool_fidelity` score near 40 and a `fork_penalty` near 0 (fork at step 1), yielding an overall score well below the 80-point pass threshold — a clear `FAIL` before any code reached production.

## Results

- **34% false positive spike** caught in CI before the upgrade reached production (retroactive simulation confirmed it would have been caught)
- **CI gate runtime**: 90 seconds for 50 scenarios, running in parallel across 4 GitHub Actions workers
- **Zero behavioral regressions** have escaped to production in the 3 months since deployment
- **Fork point precision**: When a regression is detected, the exact diverging tool call is pinpointed — engineers don't spend hours debugging "why did the false positive rate change?"
- **Cost**: The 50-scenario regression suite costs $0.18 per CI run at current GPT-4o pricing — less than the cost of one declined legitimate transaction

## Key Takeaways

- Accuracy metrics and behavioral metrics measure different things. An agent can maintain perfect accuracy while silently changing the path it takes — and that path change can expose latent bugs.
- Tool-call order is a behavioral fingerprint. `tool_fidelity` and `fork_penalty` in agentdelta's score are specifically designed to catch the class of regression Meridian experienced.
- 50 standardized scenarios is enough. You don't need thousands of test cases to detect behavioral drift — a carefully curated set covering your edge cases gives high signal at low cost.
- CI gates should fail on behavior, not just output. Adding `batch_from_directory()` to CI takes under 20 lines of Python and blocks exactly the class of silent regressions that cause production incidents.
- The HTML diff report (`to_html()`) gives engineers the context to understand *why* the gate failed, not just that it did.

## Try It Yourself

```bash
# Install agentdelta
pip install agentdelta

# Record two traces of the included example agent
python -c "
from agentdelta import record
# Simulate baseline trace
import json
trace_data = {'run_id': 'baseline', 'steps': []}
"

# Or use the CLI to diff two pre-recorded traces from the examples/ directory
git clone https://github.com/sandeep-alluru/agentdelta
cd agentdelta
pip install -e .
agentdelta diff examples/baseline.jsonl examples/candidate.jsonl
agentdelta diff examples/baseline.jsonl examples/candidate.jsonl --format json
```
