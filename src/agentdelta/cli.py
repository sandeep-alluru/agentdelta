"""Command-line interface for agentdelta."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from agentdelta.diff import diff_traces
from agentdelta.report import print_diff, to_json, to_markdown
from agentdelta.score import compute_score
from agentdelta.trace import AgentTrace


@click.group()
@click.version_option(package_name="agentdelta")
def main() -> None:
    """Semantic diff engine for AI agent behavior traces."""


@main.command()
@click.argument("trace_a", type=click.Path(exists=True, path_type=Path))
@click.argument("trace_b", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["rich", "json", "markdown"]),
    default="rich",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--fork-threshold",
    type=float,
    default=0.70,
    show_default=True,
    help="Similarity below this marks a fork point.",
)
@click.option(
    "--match-threshold",
    type=float,
    default=0.85,
    show_default=True,
    help="Similarity above this is a match.",
)
@click.option(
    "--show-matches",
    is_flag=True,
    default=False,
    help="Include matched (unchanged) steps in output.",
)
@click.option(
    "--exit-code",
    is_flag=True,
    default=False,
    help="Exit with code 1 if a regression is detected (useful in CI).",
)
def diff(
    trace_a: Path,
    trace_b: Path,
    fmt: str,
    fork_threshold: float,
    match_threshold: float,
    show_matches: bool,
    exit_code: bool,
) -> None:
    """Diff two agent trace files and report behavioral divergence.

    TRACE_A is the baseline run; TRACE_B is the candidate run.
    """
    run_a = AgentTrace.load(trace_a)
    run_b = AgentTrace.load(trace_b)

    result = diff_traces(
        run_a,
        run_b,
        fork_threshold=fork_threshold,
        match_threshold=match_threshold,
    )

    if fmt == "rich":
        print_diff(result, show_matches=show_matches)
    elif fmt == "json":
        click.echo(to_json(result))
    elif fmt == "markdown":
        click.echo(to_markdown(result))

    if exit_code and result.has_regression:
        sys.exit(1)


@main.command()
@click.argument("trace_file", type=click.Path(path_type=Path))
@click.option("--run-id", default=None, help="Explicit run identifier.")
def inspect(trace_file: Path, run_id: str | None) -> None:
    """Print a summary of a single trace file."""

    from rich.console import Console
    from rich.table import Table

    from agentdelta.trace import NodeType

    _NODE_ICONS = {
        NodeType.START: "▶",
        NodeType.LLM: "🧠",
        NodeType.TOOL_CALL: "🔧",
        NodeType.TOOL_RETURN: "↩",
        NodeType.END: "■",
    }

    console = Console()

    if not trace_file.exists():
        raise click.ClickException(f"File not found: {trace_file}")

    trace = AgentTrace.load(trace_file)

    console.print(
        f"\n[bold]Trace:[/bold] [dim]{trace.run_id}[/dim]  "
        f"[dim]{len(trace.nodes)} nodes / {len(trace.edges)} edges[/dim]\n"
    )

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("Step", style="dim", width=5)
    table.add_column("Type", width=14)
    table.add_column("Content")

    for node in trace.nodes:
        icon = _NODE_ICONS.get(node.node_type, "?")
        content = node.content[:100].replace("\n", " ")
        if len(node.content) > 100:
            content += "…"
        table.add_row(str(node.step), f"{icon} {node.node_type.value}", content)

    console.print(table)
    console.print()


@main.command()
@click.argument("trace_a", type=click.Path(exists=True, path_type=Path))
@click.argument("trace_b", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--pass-threshold",
    type=float,
    default=80.0,
    show_default=True,
    help="Score >= this is PASS.",
)
@click.option(
    "--warn-threshold",
    type=float,
    default=60.0,
    show_default=True,
    help="Score >= this (but < pass) is WARN.",
)
@click.option(
    "--fork-threshold",
    type=float,
    default=0.70,
    show_default=True,
    help="Similarity below this marks a fork point.",
)
@click.option(
    "--match-threshold",
    type=float,
    default=0.85,
    show_default=True,
    help="Similarity above this is a match.",
)
def score(
    trace_a: Path,
    trace_b: Path,
    pass_threshold: float,
    warn_threshold: float,
    fork_threshold: float,
    match_threshold: float,
) -> None:
    """Compute a regression score and exit 0 (PASS/WARN) or 1 (FAIL).

    TRACE_A is the baseline run; TRACE_B is the candidate run.

    \b
    Examples:
      agentdelta score baseline.jsonl candidate.jsonl
      agentdelta score baseline.jsonl candidate.jsonl --pass-threshold 90
    """
    run_a = AgentTrace.load(trace_a)
    run_b = AgentTrace.load(trace_b)

    result = diff_traces(
        run_a,
        run_b,
        fork_threshold=fork_threshold,
        match_threshold=match_threshold,
    )
    reg_score = compute_score(result, pass_threshold=pass_threshold, warn_threshold=warn_threshold)

    verdict_colors = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}
    color = verdict_colors[reg_score.verdict]

    click.echo(
        click.style(f"[{reg_score.verdict}]", fg=color, bold=True)
        + f"  overall={reg_score.overall:.1f}  "
        f"structural={reg_score.structural:.1f}  "
        f"semantic={reg_score.semantic:.1f}  "
        f"tool_fidelity={reg_score.tool_fidelity:.1f}  "
        f"fork_penalty={reg_score.fork_penalty:.1f}"
    )
    click.echo(f"  thresholds: pass>={pass_threshold}  warn>={warn_threshold}")

    if reg_score.verdict == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
