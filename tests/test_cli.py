"""Tests for redline CLI (redline diff, redline inspect)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from redline.cli import main


def _save_traces(simple_trace_a, simple_trace_b_match):
    """Save two traces to temp files and return their paths."""
    tmpdir = tempfile.mkdtemp()
    path_a = Path(tmpdir) / "run_a.jsonl"
    path_b = Path(tmpdir) / "run_b.jsonl"
    simple_trace_a.save(path_a)
    simple_trace_b_match.save(path_b)
    return path_a, path_b


def test_diff_rich_output(simple_trace_a, simple_trace_b_match):
    runner = CliRunner()
    path_a, path_b = _save_traces(simple_trace_a, simple_trace_b_match)
    result = runner.invoke(main, ["diff", str(path_a), str(path_b)])
    assert result.exit_code == 0
    assert "redline" in result.output


def test_diff_json_output(simple_trace_a, simple_trace_b_match):
    runner = CliRunner()
    path_a, path_b = _save_traces(simple_trace_a, simple_trace_b_match)
    result = runner.invoke(main, ["diff", str(path_a), str(path_b), "--format", "json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "run_id_a" in parsed
    assert "summary" in parsed


def test_diff_markdown_output(simple_trace_a, simple_trace_b_match):
    runner = CliRunner()
    path_a, path_b = _save_traces(simple_trace_a, simple_trace_b_match)
    result = runner.invoke(main, ["diff", str(path_a), str(path_b), "--format", "markdown"])
    assert result.exit_code == 0
    assert "## redline behavior diff" in result.output


def test_diff_exit_code_no_regression(simple_trace_a):
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    simple_trace_a.save(path)
    result = runner.invoke(main, ["diff", str(path), str(path), "--exit-code"])
    assert result.exit_code == 0


def test_diff_missing_file():
    runner = CliRunner()
    result = runner.invoke(main, ["diff", "/nonexistent/a.jsonl", "/nonexistent/b.jsonl"])
    assert result.exit_code != 0


def test_inspect_output(simple_trace_a):
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    simple_trace_a.save(path)
    result = runner.invoke(main, ["inspect", str(path)])
    assert result.exit_code == 0
    assert "run_a" in result.output


def test_diff_show_matches(simple_trace_a):
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    simple_trace_a.save(path)
    result = runner.invoke(main, ["diff", str(path), str(path), "--show-matches"])
    assert result.exit_code == 0


def test_version_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output
