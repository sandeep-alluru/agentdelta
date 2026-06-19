"""Human-readable and machine-readable diff output formatters."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentdelta.diff import DiffResult
from agentdelta.trace import NodeType

_console = Console()

_NODE_ICONS = {
    NodeType.START: "▶",
    NodeType.LLM: "🧠",
    NodeType.TOOL_CALL: "🔧",
    NodeType.TOOL_RETURN: "↩",
    NodeType.END: "■",
}

_STATUS_STYLE = {
    "match": "dim",
    "changed": "bold yellow",
    "added": "bold green",
    "removed": "bold red",
}


def _truncate(text: str, max_len: int = 72) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def print_diff(
    result: DiffResult,
    show_matches: bool = False,
    console: Console | None = None,
) -> None:
    """Print a Rich-formatted diff to the terminal."""
    con = console or _console

    # Header
    title = f"[bold]agentdelta[/bold]  [dim]{result.run_id_a}[/dim] vs [dim]{result.run_id_b}[/dim]"
    con.print(Panel(title, expand=False))

    # Summary line
    s = result.summary
    status_color = "red" if s["has_regression"] else "green"
    status_word = "REGRESSION DETECTED" if s["has_regression"] else "NO REGRESSION"
    con.print(
        f"  [{status_color}]{status_word}[/{status_color}]  "
        f"[dim]{s['matched']}/{s['total_steps']} steps matched "
        f"({s['similarity_pct']}%)[/dim]  "
        f"[yellow]{s['changed']} changed[/yellow]  "
        f"[green]+{s['added']} added[/green]  "
        f"[red]-{s['removed']} removed[/red]"
    )

    # Fork point callout
    if result.fork_point:
        fp = result.fork_point
        con.print()
        con.print(
            Panel(
                f"[bold yellow]⚡ First fork at step {fp.step_a}[/bold yellow]\n"
                f"[white]{fp.description}[/white]\n\n"
                f"  [dim]Before:[/dim] {_truncate(fp.node_a.content)}\n"
                f"  [dim]After: [/dim] {_truncate(fp.node_b.content)}",
                title="[bold yellow]Fork Point[/bold yellow]",
                border_style="yellow",
                expand=False,
            )
        )

    # Step table
    if result.changed_steps or result.added_steps or result.removed_steps or show_matches:
        con.print()
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("Step", style="dim", width=5)
        table.add_column("Status", width=8)
        table.add_column("Type", width=12)
        table.add_column("Detail", no_wrap=False)

        for step in result.steps:
            if step.status == "match" and not show_matches:
                continue

            style = _STATUS_STYLE[step.status]
            node = step.step_a or step.step_b
            if node is None:
                continue

            icon = _NODE_ICONS.get(node.node_type, "?")
            step_num = str(node.step)
            node_type = f"{icon} {node.node_type.value}"

            if step.status == "match":
                detail = Text(_truncate(node.content), style="dim")
            else:
                detail = Text(step.summary or _truncate(node.content), style=style)

            table.add_row(step_num, step.status.upper(), node_type, detail)

        con.print(table)

    con.print()


def to_json(result: DiffResult) -> str:
    """Serialize a DiffResult to JSON for CI/CD consumption."""
    steps_data = []
    for step in result.steps:
        node = step.step_a or step.step_b
        steps_data.append(
            {
                "status": step.status,
                "similarity": round(step.similarity, 4),
                "step_a": step.step_a.step if step.step_a else None,
                "step_b": step.step_b.step if step.step_b else None,
                "node_type": node.node_type.value if node else None,
                "summary": step.summary,
            }
        )

    data: dict[str, Any] = {
        "run_id_a": result.run_id_a,
        "run_id_b": result.run_id_b,
        "summary": result.summary,
        "fork_point": (
            {
                "step_a": result.fork_point.step_a,
                "step_b": result.fork_point.step_b,
                "similarity": round(result.fork_point.similarity, 4),
                "description": result.fork_point.description,
                "node_a_content": result.fork_point.node_a.content[:200],
                "node_b_content": result.fork_point.node_b.content[:200],
            }
            if result.fork_point
            else None
        ),
        "steps": steps_data,
    }
    return json.dumps(data, indent=2)


def to_markdown(result: DiffResult) -> str:
    """Generate a GitHub PR comment in Markdown."""
    s = result.summary
    status_emoji = "🔴" if s["has_regression"] else "🟢"
    status_word = "Regression detected" if s["has_regression"] else "No regression"

    lines = [
        "## agentdelta behavior diff",
        "",
        f"{status_emoji} **{status_word}** &nbsp;·&nbsp; "
        f"{s['matched']}/{s['total_steps']} steps matched ({s['similarity_pct']}%) &nbsp;·&nbsp; "
        f"**{s['changed']}** changed &nbsp; "
        f"**+{s['added']}** added &nbsp; **-{s['removed']}** removed",
        "",
    ]

    if result.fork_point:
        fp = result.fork_point
        lines += [
            "### ⚡ First fork point",
            "",
            f"> **Step {fp.step_a}** — {fp.description}",
            "",
            "```diff",
            f"- {fp.node_a.content[:200]}",
            f"+ {fp.node_b.content[:200]}",
            "```",
            "",
        ]

    changed = [s for s in result.steps if s.status != "match"]
    if changed:
        lines += [
            "### Changed steps",
            "",
            "| Step | Status | Type | Detail |",
            "|------|--------|------|--------|",
        ]
        for step in changed[:20]:  # cap at 20 rows
            node = step.step_a or step.step_b
            if not node:
                continue
            icon = {"added": "+", "removed": "-", "changed": "~"}.get(step.status, "")
            detail = (step.summary or node.content[:60]).replace("|", "\\|")
            lines.append(
                f"| {node.step} | {icon} {step.status} | {node.node_type.value} | {detail} |"
            )
        if len(changed) > 20:
            lines.append(f"| … | | | *{len(changed) - 20} more rows* |")

    lines += [
        "",
        "<details><summary>Full JSON report</summary>",
        "",
        "```json",
        to_json(result),
        "```",
        "",
        "</details>",
        "",
        "*Generated by [agentdelta](https://github.com/sandeep-alluru/agentdelta)*",
    ]

    return "\n".join(lines)
