"""
model_upgrade_audit.py — agentdelta for model upgrade auditing.

Story: A team is upgrading their code review agent from gpt-4o-mini to
claude-3-5-sonnet. They want to know: does the more capable model change
behavior on the same code review tasks?

v1 (mini): check_syntax → check_style → LLM summary
v2 (sonnet): check_syntax → (if issues found) suggest_fix → check_style → LLM summary

The diff reveals that sonnet is more thorough but changes the tool call
sequence — critical info before rolling out a model upgrade to production.

Run:
    python examples/model_upgrade_audit.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from agentdelta.diff import diff_traces
from agentdelta.instrument import AgentdeltaCallback
from agentdelta.report import print_diff
from agentdelta.trace import AgentTrace


# ---------------------------------------------------------------------------
# Fake LLM response shim
# ---------------------------------------------------------------------------

class _LLMResponse:
    def __init__(self, text: str) -> None:
        self.generations = [[type("Gen", (), {"text": text})()]]


# ---------------------------------------------------------------------------
# Code review tasks (simulated code snippets)
# ---------------------------------------------------------------------------

@dataclass
class ReviewTask:
    id: str
    file: str
    snippet: str
    has_syntax_issue: bool
    description: str


REVIEW_TASKS = [
    ReviewTask(
        id="T1",
        file="auth/login.py",
        snippet="def login(user, pwd):\n    if user == 'admin' and pwd == 'admin': return True",
        has_syntax_issue=False,
        description="Hardcoded credentials check",
    ),
    ReviewTask(
        id="T2",
        file="api/routes.py",
        snippet="@app.route('/users')\ndef get_users():\n    return db.execute('SELECT * FROM users WHERE id=' + request.args.get('id'))",
        has_syntax_issue=True,
        description="SQL injection vulnerability",
    ),
    ReviewTask(
        id="T3",
        file="utils/cache.py",
        snippet="import redis\ncache = redis.Redis(host='localhost')\ndef get(key): return cache.get(key)",
        has_syntax_issue=False,
        description="Redis cache helper",
    ),
    ReviewTask(
        id="T4",
        file="models/user.py",
        snippet="class User:\n    def __init__(self, name, email, password):\n        self.password = password  # stored in plaintext",
        has_syntax_issue=True,
        description="Plaintext password storage",
    ),
]


# ---------------------------------------------------------------------------
# v1 agent: gpt-4o-mini style — fast, linear, no branching
# Always: check_syntax → check_style → final LLM summary
# ---------------------------------------------------------------------------

def build_v1_trace(task: ReviewTask) -> AgentTrace:
    """
    v1 code review agent (gpt-4o-mini).
    Tool sequence: check_syntax → check_style → LLM summary.
    Fast and linear — mini doesn't add extra steps even when issues exist.
    Simulates: ChatOpenAI(model="gpt-4o-mini") with code_review_tools.
    """
    cb = AgentdeltaCallback(run_id=f"mini-{task.id}")
    cb.on_chain_start({}, {"input": f"Review {task.file}: {task.snippet[:60]}"})

    cb.on_llm_end(_LLMResponse(
        f"I will review {task.file} by checking syntax then style."
    ))

    # Step 1: syntax check (always runs)
    cb.on_tool_start({"name": "check_syntax"}, f"file='{task.file}', code='{task.snippet[:80]}'")
    if task.has_syntax_issue:
        cb.on_tool_end(f"WARNING: Potential issue detected in {task.file}. Line 2 flagged.")
    else:
        cb.on_tool_end(f"OK: {task.file} passes syntax validation.")

    # Step 2: style check (always runs, regardless of syntax result)
    cb.on_tool_start({"name": "check_style"}, f"file='{task.file}', standard='PEP8'")
    cb.on_tool_end("Style: 2 minor warnings (line length, missing docstring).")

    # Step 3: LLM summary
    if task.has_syntax_issue:
        summary = (
            f"Code review for {task.file}: found potential issues. "
            "Recommend manual inspection before merging."
        )
    else:
        summary = f"Code review for {task.file}: no critical issues. Minor style fixes needed."

    cb.on_llm_end(_LLMResponse(summary))
    cb.on_chain_end({"output": summary})

    return cb.trace


# ---------------------------------------------------------------------------
# v2 agent: claude-3-5-sonnet style — more thorough, branches on issues
# check_syntax → (if issues) suggest_fix → check_style → LLM summary
# ---------------------------------------------------------------------------

def build_v2_trace(task: ReviewTask) -> AgentTrace:
    """
    v2 code review agent (claude-3-5-sonnet).
    More thorough: adds suggest_fix step when syntax issues are detected.
    Simulates: ChatAnthropic(model="claude-3-5-sonnet-20241022") with same tools.
    """
    cb = AgentdeltaCallback(run_id=f"sonnet-{task.id}")
    cb.on_chain_start({}, {"input": f"Review {task.file}: {task.snippet[:60]}"})

    cb.on_llm_end(_LLMResponse(
        f"I will thoroughly review {task.file}. Starting with syntax analysis."
    ))

    # Step 1: syntax check
    cb.on_tool_start({"name": "check_syntax"}, f"file='{task.file}', code='{task.snippet[:80]}'")
    if task.has_syntax_issue:
        cb.on_tool_end(
            f"CRITICAL: Security/correctness issue detected in {task.file}. "
            "Immediate fix required before style review."
        )

        # Step 2 (ADDED by sonnet): suggest_fix — sonnet proactively fixes before styling
        cb.on_llm_end(_LLMResponse(
            "A critical issue was found. I should suggest a fix before proceeding to style."
        ))
        cb.on_tool_start({"name": "suggest_fix"}, f"file='{task.file}', issue='critical'")
        cb.on_tool_end(
            f"Suggested fix for {task.file}: use parameterized queries / "
            "hash passwords with bcrypt. See OWASP guidelines."
        )
    else:
        cb.on_tool_end(f"OK: {task.file} passes syntax validation.")

    # Step 3: style check (sonnet always runs this, even after fixes)
    cb.on_tool_start({"name": "check_style"}, f"file='{task.file}', standard='PEP8'")
    cb.on_tool_end("Style: 2 minor warnings (line length, missing docstring).")

    # Step 4: LLM summary (more detailed from sonnet)
    if task.has_syntax_issue:
        summary = (
            f"CRITICAL issues in {task.file}: {task.description}. "
            "Fix suggestions provided. Do not merge until resolved. "
            "Also: 2 minor style warnings."
        )
    else:
        summary = (
            f"Code review for {task.file}: no critical issues. "
            "Minor style fixes recommended. Safe to merge after style fixes."
        )

    cb.on_llm_end(_LLMResponse(summary))
    cb.on_chain_end({"output": summary})

    return cb.trace


# ---------------------------------------------------------------------------
# Audit runner
# ---------------------------------------------------------------------------

def run_model_upgrade_audit() -> int:
    print("=" * 70)
    print("  agentdelta Model Upgrade Audit")
    print("  gpt-4o-mini  →  claude-3-5-sonnet")
    print("  4 code review tasks")
    print("=" * 70)
    print()

    # Track results for summary table
    table_rows: list[tuple[str, str, str, str]] = []

    for task in REVIEW_TASKS:
        print(f"── Task {task.id}: {task.description} ({task.file})")
        print()

        trace_a = build_v1_trace(task)
        trace_b = build_v2_trace(task)

        result = diff_traces(trace_a, trace_b, fork_threshold=0.70, match_threshold=0.85)
        print_diff(result)

        if result.has_regression:
            fp = result.fork_point
            # Format fork step info
            fork_info = f"step {fp.step_a}" if fp else "unknown step"
            added_count = len(result.added_steps)
            status = "DIVERGED"
            detail = f"{fork_info}, +{added_count} tool call(s)"
        else:
            status = "MATCHED"
            pct = result.summary.get("similarity_pct", 100.0)
            detail = f"{pct}% similarity"

        table_rows.append((task.id, task.description, status, detail))

    # ---------------------------------------------------------------------------
    # Summary table — the output a team reviews before approving the model upgrade
    # ---------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  MODEL UPGRADE AUDIT SUMMARY")
    print("  Comparing: gpt-4o-mini (baseline) vs claude-3-5-sonnet (candidate)")
    print("=" * 70)
    print()
    print(f"  {'Task':<6} {'Description':<35} {'Result':<10} {'Detail'}")
    print(f"  {'-'*4:<6} {'-'*33:<35} {'-'*8:<10} {'------'}")
    diverged = []
    for task_id, desc, status, detail in table_rows:
        marker = ">>>" if status == "DIVERGED" else "   "
        print(f"  {marker} {task_id:<4} {desc[:33]:<35} {status:<10} {detail}")
        if status == "DIVERGED":
            diverged.append(task_id)

    print()
    print("  BEHAVIORAL ANALYSIS:")
    print()
    print("  MATCHED tasks: sonnet produces functionally equivalent output.")
    print("  Tool call sequence is the same; only reasoning wording differs.")
    print()
    print("  DIVERGED tasks: sonnet adds a `suggest_fix` tool call when")
    print("  critical issues are detected. This is MORE thorough behavior,")
    print("  not a regression — but it changes the tool call sequence.")
    print()
    print("  ROLLOUT RECOMMENDATION:")
    print("  - The extra `suggest_fix` calls from sonnet are intentional.")
    print("  - Update downstream consumers to handle variable tool sequences.")
    print("  - Verify `suggest_fix` latency is acceptable (+~300ms per call).")
    print("  - Enable model upgrade for code review pipeline.")
    print()

    if diverged:
        print(f"  Tasks with behavioral changes: {', '.join(diverged)}")
        print("  (These are improvements, not regressions — safe to deploy.)")

    return 0


if __name__ == "__main__":
    sys.exit(run_model_upgrade_audit())
