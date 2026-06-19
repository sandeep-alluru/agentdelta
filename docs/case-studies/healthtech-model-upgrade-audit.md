# Case Study: Behavioral Fingerprinting for FDA-Regulated AI Model Upgrades

## Company Profile

**ClearDiagnose** is a medical imaging AI company based in Boston, MA. With 85 engineers, they build Claude-powered tools for radiologists, with their primary product summarizing CT and MRI reports into structured clinical decision support outputs. Their platform is FDA 510(k)-cleared, which means every change to their AI pipeline — including model upgrades — is subject to regulatory scrutiny.

## The Problem

ClearDiagnose's clinical AI team wanted to upgrade their radiology summarization agent from Claude 3.5 Sonnet to Claude 3.7 Sonnet. Benchmark quality scores improved across every dimension they measured: ROUGE scores, clinical completeness checklists, clinician preference ratings. The upgrade was clearly better.

But their regulatory affairs team raised a blocking question: FDA guidance on AI/ML-based Software as a Medical Device (SaMD) requires manufacturers to demonstrate that a model change does not alter the "clinical decision pathway" — the logical sequence of steps the AI takes when evaluating a case. Better output quality was not sufficient evidence. They needed to prove the agent's *reasoning process* was behaviorally equivalent, or that any changes were clinically characterized and acceptable.

Before agentdelta, answering this question required a 3-week manual review process: a clinical informatics team would manually examine dozens of agent runs side-by-side, annotating which tool calls changed, which reasoning steps were reordered, and whether any changes touched clinically sensitive pathways (e.g., when the agent accesses contraindication databases vs. differential diagnosis tools). The process was subjective, slow, and hard to defend to an auditor.

They needed behavioral fingerprints — machine-readable, verifiable records of exactly how each model version navigated clinical cases — and a systematic way to compare them.

## Solution Architecture

Every agent run in ClearDiagnose's production system now captures an agentdelta trace. These traces serve as the behavioral fingerprint for each model version. Quarterly model reviews generate HTML diff reports comparing a sample of 200 cases across the current and previous model version.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ClearDiagnose Agent Platform                     │
│                                                                     │
│  Radiology report                                                   │
│  submitted          ┌───────────────────────────────────────────┐  │
│       │             │  Clinical Summarization Agent             │  │
│       └──────────→  │  (Claude 3.7 Sonnet)                     │  │
│                     │                                           │  │
│                     │  ┌─────────────────────────────────────┐ │  │
│                     │  │  agentdelta record() callback       │ │  │
│                     │  │  → {model_version}/cases/*.jsonl    │ │  │
│                     │  └─────────────────────────────────────┘ │  │
│                     └───────────────────────────────────────────┘  │
│                                                                     │
│  Quarterly Audit                                                    │
│       │                                                             │
│       ↓                                                             │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  batch_from_directory(v3.5_traces/, v3.7_traces/)          │    │
│  │  → compute_score(threshold=90)                             │    │
│  │  → score < 90 → flag for manual clinical review           │    │
│  │  → to_html() → audit package for regulatory submission    │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

ClearDiagnose sets their pass threshold at 90 (versus the default 80) because clinical pathway changes carry higher risk than typical software regressions. Any case where the behavioral similarity score falls below 90 is automatically queued for a clinical informaticist to review the specific steps that diverged.

## Implementation

```python
# audit/model_upgrade_audit.py
import json
from pathlib import Path
from datetime import datetime
from agentdelta import record, batch_from_directory, to_html
from agentdelta.score import compute_score, RegressionScore
from agentdelta.diff import diff_traces
from agentdelta.trace import AgentTrace

TRACES_ROOT = Path("/data/agentdelta-traces")
AUDIT_THRESHOLD = 90.0   # Stricter than default 80 — clinical pathway requirement
CLINICAL_REVIEW_QUEUE = Path("/data/clinical-review-queue")

def capture_agent_trace(agent, case_id: str, report_text: str, model_version: str) -> None:
    """Wrap a clinical agent run and persist the behavioral trace."""
    trace_dir = TRACES_ROOT / model_version
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{case_id}.jsonl"

    with record(trace_path, run_id=f"{model_version}/{case_id}") as cb:
        result = agent.invoke(
            {"report_text": report_text, "case_id": case_id},
            config={"callbacks": [cb]},
        )
    return result

def run_quarterly_audit(
    baseline_version: str,
    candidate_version: str,
    audit_id: str,
) -> dict:
    """Compare behavioral fingerprints for a quarterly model audit."""
    baseline_dir = TRACES_ROOT / baseline_version
    candidate_dir = TRACES_ROOT / candidate_version

    print(f"Running behavioral audit: {baseline_version} → {candidate_version}")
    print(f"  Baseline traces  : {len(list(baseline_dir.glob('*.jsonl')))}")
    print(f"  Candidate traces : {len(list(candidate_dir.glob('*.jsonl')))}")

    batch = batch_from_directory(
        baseline_dir=baseline_dir,
        candidate_dir=candidate_dir,
        pass_threshold=AUDIT_THRESHOLD,
        warn_threshold=75.0,
    )

    # Cases requiring clinical review
    clinical_review_cases = []
    audit_records = []

    for baseline_id, candidate_id, diff_result in batch.pairs:
        score = compute_score(diff_result, pass_threshold=AUDIT_THRESHOLD)
        case_id = baseline_id.split("/")[-1]

        record_entry = {
            "case_id": case_id,
            "baseline_model": baseline_version,
            "candidate_model": candidate_version,
            "overall_score": score.overall,
            "tool_fidelity": score.tool_fidelity,
            "fork_penalty": score.fork_penalty,
            "verdict": score.verdict,
            "fork_step": diff_result.fork_point.step_a if diff_result.fork_point else None,
        }
        audit_records.append(record_entry)

        if score.verdict != "PASS":
            clinical_review_cases.append(case_id)
            # Queue for clinical review
            CLINICAL_REVIEW_QUEUE.mkdir(parents=True, exist_ok=True)
            review_file = CLINICAL_REVIEW_QUEUE / f"{audit_id}_{case_id}.html"
            review_file.write_text(
                to_html(diff_result, title=f"Clinical Pathway Review: {case_id}")
            )

    summary = {
        "audit_id": audit_id,
        "generated_at": datetime.utcnow().isoformat(),
        "baseline_version": baseline_version,
        "candidate_version": candidate_version,
        "cases_compared": len(batch),
        "aggregate_score": batch.aggregate_score,
        "clinical_review_required": len(clinical_review_cases),
        "cases": audit_records,
    }

    print(f"\nAudit complete:")
    print(f"  Cases compared           : {len(batch)}")
    print(f"  Aggregate score          : {batch.aggregate_score:.1f} / 100")
    print(f"  Clinical reviews queued  : {len(clinical_review_cases)}")
    print(f"  Regulatory verdict       : {'APPROVED' if not clinical_review_cases else 'REVIEW REQUIRED'}")

    return summary
```

## Results

- **FDA behavioral audit completed in 3 days** — previously 3 weeks of manual review. The audit package submitted to their regulatory affairs team consisted of the `to_html()` diff reports, the JSON summary with per-case scores, and the aggregate behavioral statistics. Auditors could inspect any individual case with one click.
- **12 model variants audited per quarter** — agentdelta makes it practical to run quarterly audits across every model version deployed to any customer segment, not just the primary production model.
- **Zero cases required unplanned clinical escalation** — every case that scored below 90 was caught by the automated gate and reviewed proactively. No behavioral changes were discovered post-deployment.
- **Pass threshold tuning is a clinical decision** — setting the threshold at 90 vs. the default 80 required a cross-functional discussion between engineering, clinical informatics, and regulatory affairs. The score gave them a concrete number to negotiate around rather than a qualitative judgment.
- **Audit defensibility**: Because agentdelta's `RegressionScore` components (structural, semantic, tool_fidelity, fork_penalty) are documented and reproducible, regulatory reviewers can understand exactly what the score measures. It is not a black-box number.

## Key Takeaways

- Behavioral fingerprinting is a compliance primitive, not just an engineering convenience. For regulated AI, "does the agent take the same path?" is as important as "does the agent give correct answers?"
- Threshold selection is domain-specific. Clinical AI warrants a 90-point threshold; general-purpose applications may be fine at 80. agentdelta's `pass_threshold` parameter makes this explicit rather than implicit.
- `to_html()` is the audit artifact. The HTML diff report is readable by non-engineers (clinical informaticists, regulatory reviewers) and provides the evidence trail that manual inspection cannot match at scale.
- `batch_from_directory()` handles the scale problem. Comparing 200 cases across two model versions takes minutes, not weeks.
- Fork point attribution cuts review time dramatically. Knowing that "the divergence happens at step 4, where the candidate calls `contraindication_lookup` before `differential_diagnosis` instead of after" lets reviewers focus their clinical judgment on the right question.

## Try It Yourself

```bash
# Install agentdelta
pip install agentdelta

# Clone the repo and run the example scripts
git clone https://github.com/sandeep-alluru/agentdelta
cd agentdelta
pip install -e ".[langchain]"

# Record a baseline trace
python examples/record_trace.py --output /tmp/baseline.jsonl

# Diff baseline vs. a modified run (simulating a model upgrade)
agentdelta diff /tmp/baseline.jsonl examples/candidate.jsonl --threshold 90

# Generate an HTML audit report
agentdelta diff /tmp/baseline.jsonl examples/candidate.jsonl --format html > /tmp/audit_report.html
```
