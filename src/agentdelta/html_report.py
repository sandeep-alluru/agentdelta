"""Self-contained HTML diff report generator for agentdelta."""

from __future__ import annotations

import html
from typing import Any

from agentdelta.diff import DiffResult

_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
    margin: 0;
    padding: 24px;
    background: #0d1117;
    color: #c9d1d9;
}
h1 { color: #58a6ff; margin-bottom: 4px; }
.subtitle { color: #8b949e; margin-bottom: 24px; font-size: 0.9em; }
.summary-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 24px;
    margin-bottom: 24px;
    display: flex;
    gap: 32px;
    flex-wrap: wrap;
}
.stat { text-align: center; }
.stat-value { font-size: 2em; font-weight: bold; }
.stat-label { font-size: 0.75em; color: #8b949e; text-transform: uppercase; }
.pass { color: #3fb950; }
.warn { color: #d29922; }
.fail { color: #f85149; }
.fork-box {
    background: #1c1a12;
    border: 1px solid #d29922;
    border-radius: 8px;
    padding: 16px 24px;
    margin-bottom: 24px;
}
.fork-box h3 { color: #d29922; margin-top: 0; }
.fork-content {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 12px;
}
.fork-side {
    background: #0d1117;
    border-radius: 4px;
    padding: 12px;
    font-size: 0.85em;
    word-break: break-word;
}
.fork-side-label { font-size: 0.75em; color: #8b949e; margin-bottom: 4px; }
table {
    border-collapse: collapse;
    width: 100%;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    overflow: hidden;
}
th {
    background: #21262d;
    color: #8b949e;
    text-align: left;
    padding: 10px 14px;
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
td { padding: 10px 14px; border-top: 1px solid #21262d; font-size: 0.85em; vertical-align: top; }
tr.match td { opacity: 0.5; }
tr.changed td { background: #1f1b06; }
tr.added td { background: #0d1f0d; }
tr.removed td { background: #1f0d0d; }
tr.fork-here td { outline: 2px solid #d29922; outline-offset: -2px; }
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75em;
    font-weight: bold;
}
.badge-match { background: #21262d; color: #8b949e; }
.badge-changed { background: #3d2e00; color: #d29922; }
.badge-added { background: #0d2c0d; color: #3fb950; }
.badge-removed { background: #2c0d0d; color: #f85149; }
.type-tag {
    font-size: 0.75em;
    background: #21262d;
    padding: 2px 6px;
    border-radius: 3px;
    margin-right: 4px;
}
"""


def _esc(text: str, max_len: int = 120) -> str:
    """HTML-escape and truncate text."""
    truncated = text[:max_len] + ("…" if len(text) > max_len else "")
    return html.escape(truncated)


def _stat_html(value: Any, label: str, css_class: str = "") -> str:
    cls = f' class="stat-value {css_class}"' if css_class else ' class="stat-value"'
    return (
        f'<div class="stat">'
        f'<div{cls}>{value}</div>'
        f'<div class="stat-label">{label}</div>'
        f"</div>"
    )


def to_html(diff_result: DiffResult, title: str = "agentdelta diff") -> str:
    """Generate a self-contained HTML behavioral diff report.

    Args:
        diff_result: The result of :func:`~agentdelta.diff.diff_traces`.
        title: Browser tab title for the report page.

    Returns:
        A complete HTML string with embedded CSS — no external dependencies.
    """
    s = diff_result.summary
    has_regression = s.get("has_regression", False)
    verdict_class = "fail" if has_regression else "pass"
    verdict_text = "REGRESSION DETECTED" if has_regression else "NO REGRESSION"
    sim_pct = s.get("similarity_pct", 0.0)

    # ── Summary box ───────────────────────────────────────────────────────────
    summary_html = (
        '<div class="summary-box">'
        + _stat_html(
            f'<span class="{verdict_class}">{verdict_text}</span>',
            "verdict",
        )
        + _stat_html(
            f'<span class="{verdict_class}">{sim_pct}%</span>',
            "similarity",
        )
        + _stat_html(f"{s.get('matched', 0)}/{s.get('total_steps', 0)}", "steps matched")
        + _stat_html(f"+{s.get('added', 0)}", "added", "pass")
        + _stat_html(f"-{s.get('removed', 0)}", "removed", "fail")
        + _stat_html(str(s.get("changed", 0)), "changed", "warn")
        + "</div>"
    )

    # ── Fork point ────────────────────────────────────────────────────────────
    fork_html = ""
    if diff_result.fork_point:
        fp = diff_result.fork_point
        fork_html = (
            '<div class="fork-box">'
            f'<h3>⚡ First Fork at Step {fp.step_a}</h3>'
            f"<p>{_esc(fp.description)}</p>"
            '<div class="fork-content">'
            '<div class="fork-side">'
            '<div class="fork-side-label">BEFORE (baseline)</div>'
            f"{_esc(fp.node_a.content, 300)}"
            "</div>"
            '<div class="fork-side">'
            '<div class="fork-side-label">AFTER (candidate)</div>'
            f"{_esc(fp.node_b.content, 300)}"
            "</div>"
            "</div>"
            "</div>"
        )

    # ── Step table ────────────────────────────────────────────────────────────
    rows_html = ""
    fork_step_a = diff_result.fork_point.step_a if diff_result.fork_point else None

    for step in diff_result.steps:
        node = step.step_a or step.step_b
        if node is None:
            continue

        is_fork = fork_step_a is not None and node.step == fork_step_a
        row_class = f"fork-here {step.status}" if is_fork else step.status

        step_a_cell = str(step.step_a.step) if step.step_a else "—"
        step_b_cell = str(step.step_b.step) if step.step_b else "—"

        badge_class = f"badge badge-{step.status}"
        type_icon = {
            "start": "▶",
            "llm": "🧠",
            "tool_call": "🔧",
            "tool_return": "↩",
            "end": "■",
        }.get(node.node_type.value, "?")

        content_a = _esc(step.step_a.content, 100) if step.step_a else "—"
        content_b = _esc(step.step_b.content, 100) if step.step_b else "—"
        detail = ""
        if step.status == "match":
            detail = content_a
        elif step.status == "changed":
            detail = (
                f'<span style="color:#f85149">{content_a}</span><br>'
                f'<span style="color:#3fb950">&#8594; {content_b}</span>'
            )
        elif step.status == "added":
            detail = f'<span style="color:#3fb950">{content_b}</span>'
        elif step.status == "removed":
            detail = f'<span style="color:#f85149">{content_a}</span>'

        fork_marker = " ⚡" if is_fork else ""
        rows_html += (
            f'<tr class="{row_class}">'
            f"<td>{step_a_cell}</td>"
            f"<td>{step_b_cell}</td>"
            f'<td><span class="{badge_class}">{step.status.upper()}{fork_marker}</span></td>'
            f'<td><span class="type-tag">{type_icon} {node.node_type.value}</span></td>'
            f"<td>{detail}</td>"
            "</tr>"
        )

    table_html = (
        "<table>"
        "<thead><tr>"
        "<th>Step A</th><th>Step B</th><th>Status</th><th>Type</th><th>Detail</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
    )

    # ── Full page ─────────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="subtitle">
    Baseline: <code>{html.escape(diff_result.run_id_a)}</code>
    &nbsp;vs&nbsp;
    Candidate: <code>{html.escape(diff_result.run_id_b)}</code>
  </p>
  {summary_html}
  {fork_html}
  {table_html}
</body>
</html>
"""
