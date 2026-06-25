# Case Study: Detecting Prompt Regression in a 50-Post Autonomous Content Pipeline

## Company Profile

**IndieSignal** is a one-person media company building AI-powered thought leadership content for B2B SaaS founders. The pipeline runs fully autonomously: every morning it researches a topic, drafts a LinkedIn post, self-evaluates the output for "technical insight" and "founder credibility," then publishes or queues for review. The stack is LangGraph on a single DigitalOcean droplet, with posts stored in a Supabase table and evaluation scores written to a Postgres log.

## The Problem

After three days of smooth operation — 15 posts published, evaluation pass rate above 90%, average technical insight score of 8.2 / 10 — the developer made a one-word change to the `/goal` definition in the system prompt: swapped "explain" for "illuminate." The intent was stylistic. The next morning, the evaluator failure rate on posts tagged `technical_insight` had jumped from 8% to 48%.

The first instinct was to blame the evaluator. But the evaluator prompt had not changed. The raw post text looked fine to a casual reader. The developer spent four hours adding logging, re-running ablations, and comparing post length distributions before giving up and reverting the change. The regression was real and measurable, but there was no way to answer the question that mattered: *at which decision step did the agent's behavior actually diverge?*

The problem is structural. When you change a system prompt and an eval score drops, you get a verdict but no evidence. Did the agent use a different tool to gather research? Did it skip the self-consistency check? Did it reach the evaluation step with a shorter reasoning chain? Traditional logging captures inputs and outputs. It does not capture the behavioral path between them.

A week later the developer reached for agentdelta. A full batch re-run of the 50-post corpus — baseline traces captured before the prompt change, candidate traces re-recorded with the "illuminate" variant — took 11 minutes and produced an exact answer: the fork happened at step 4, where the candidate agent called `generate_post` without first calling `self_consistency_check`. The single-word change had caused the model to skip a tool. Three times more often on average across the `technical_insight` post type.

## Solution Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                  IndieSignal Content Pipeline                         │
│                                                                       │
│  Morning run (baseline, "explain" prompt)                             │
│       │                                                               │
│       ↓                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  LangGraph agent                                                │ │
│  │  research_topic → self_consistency_check → generate_post       │ │
│  │       │                    │                       │            │ │
│  │       └────────────────────┴───────────────────────┘            │ │
│  │                 agentdelta record() callback                    │ │
│  │                 → baseline/post_{id}.jsonl                      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  Candidate run (modified, "illuminate" prompt)                        │
│       │                                                               │
│       ↓  same pipeline, same topics, new system prompt               │
│       → candidate/post_{id}.jsonl                                     │
│                                                                       │
│  agentdelta batch diff                                                │
│       │                                                               │
│       ↓                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  batch_from_directory(baseline/, candidate/)                    │ │
│  │  → aggregate_score, regressions list                            │ │
│  │  → per-pair compute_score() → PASS / WARN / FAIL verdict       │ │
│  │  → fork_point: step 4 — candidate skips self_consistency_check │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

The batch diff compares every baseline `.jsonl` file in `baseline/` against a filename-matched candidate in `candidate/`. `pass_threshold=80` and `warn_threshold=60` mirror the defaults — appropriate for a content pipeline where behavioral drift matters but the stakes are lower than regulated domains. Any post scoring below 60 surfaces its fork point for manual review.

The key metric is `tool_fidelity`: the fraction of tool calls in the baseline that were reproduced in the candidate. A score near 33 on `technical_insight` posts (one of three expected tool calls matched) immediately identified the skipped `self_consistency_check` as the regression root cause — not the evaluator, not the post quality, not a data issue.

## Implementation

```python
# audit/prompt_regression_check.py
import sys
from pathlib import Path

from agentdelta import batch_from_directory, to_html
from agentdelta.score import compute_score

BASELINE_DIR = Path("traces/baseline")
CANDIDATE_DIR = Path("traces/candidate")
REPORT_DIR   = Path("reports")

PASS_THRESHOLD = 80.0
WARN_THRESHOLD = 60.0


def run_regression_check() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    batch = batch_from_directory(
        baseline_dir=BASELINE_DIR,
        candidate_dir=CANDIDATE_DIR,
        pass_threshold=PASS_THRESHOLD,
        warn_threshold=WARN_THRESHOLD,
    )

    print(f"Prompt regression audit — 50-post corpus")
    print(f"  Posts compared    : {len(list(batch.pairs))}")
    print(f"  Aggregate score   : {batch.aggregate_score:.1f} / 100")
    print(f"  Regressions       : {len(batch.regressions)}")
    print()

    for baseline_id, candidate_id, diff_result in batch.pairs:
        score = compute_score(diff_result, pass_threshold=PASS_THRESHOLD)
        if score.verdict != "PASS":
            print(f"  [{score.verdict}] {baseline_id}")
            print(f"    tool_fidelity={score.tool_fidelity:.1f}  "
                  f"fork_penalty={score.fork_penalty:.1f}  "
                  f"overall={score.overall:.1f}")
            if diff_result.fork_point:
                fp = diff_result.fork_point
                print(f"    Fork at step {fp.step_a}: {fp.description}")
            # Write per-post HTML diff for manual review
            report_path = REPORT_DIR / f"{baseline_id}.html"
            report_path.write_text(to_html(diff_result))
            print(f"    Report → {report_path}")
            print()

    return 1 if batch.has_regressions else 0


if __name__ == "__main__":
    sys.exit(run_regression_check())
```

## Results

- **Regression root cause identified in 11 minutes** — versus 4 hours of manual log analysis that produced no actionable answer
- **Evaluator failure rate**: 8% → 48% on `technical_insight` posts (the original symptom); agentdelta traced this to a tool-skip, not content quality
- **Tool call inflation**: 1x → 3x `generate_post` calls on average for the affected post type, driven by missing `self_consistency_check` gating
- **`tool_fidelity` score on regressing posts**: 33.0 (one of three expected tool calls matched) — a clear signal in the diff report
- **Aggregate behavioral score**: 74.1 / 100 across all 50 posts — above the WARN threshold but below PASS, correctly flagging the change as risky before publish

The developer reverted the one-word change, verified the aggregate score returned to 94.7, and then re-introduced it with an explicit `self_consistency_check` call forced in the tool-routing logic. The second version of the "illuminate" prompt passed the regression gate.

## Key Takeaways

- A one-word system prompt change can silently remove a tool call — and tool omission is invisible to output-quality metrics.
- `tool_fidelity` is the right metric to watch first: a score of 33 tells you immediately that the candidate is skipping two-thirds of the baseline's tool calls.
- Fork point attribution converts a symptom ("evaluator failure rate jumped") into a root cause ("fork at step 4: candidate skips `self_consistency_check`").
- Batch diff over a full corpus is faster than one manual ablation run and produces quantified, reproducible evidence rather than subjective impression.
- Prompt regression gates are not just for teams — a solo developer with a 50-post pipeline gets the same value from `batch_from_directory()` that a 200-engineer company gets from a CI suite.

## Try It Yourself

```bash
pip install agentdelta

git clone https://github.com/sandeep-alluru/agentdelta
cd agentdelta
pip install -e .

python examples/content_pipeline_regression.py
```
