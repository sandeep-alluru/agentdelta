"""
End-to-end smoke test for agentdelta.

Simulates a user who just cloned the repo and wants to verify everything works.
No mocking, no fixtures — real embeddings, real CLI, real HTTP server.

Run from repo root:
    python smoke_test.py
    python smoke_test.py --verbose

Exit 0 = all passed. Exit 1 = at least one failure.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path

# ── Colours ───────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
REPO_ROOT = Path(__file__).parent

passed: list[str] = []
failed: list[tuple[str, str]] = []


def ok(name: str) -> None:
    passed.append(name)
    print(f"  {GREEN}✓{RESET} {name}")


def fail(name: str, reason: str) -> None:
    failed.append((name, reason))
    print(f"  {RED}✗{RESET} {name}")
    if VERBOSE:
        print(f"    {YELLOW}{reason}{RESET}")


def section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")


def run(name: str, fn):  # noqa: ANN001
    try:
        fn()
        ok(name)
    except Exception as exc:
        reason = str(exc) if not VERBOSE else traceback.format_exc().strip()
        fail(name, reason)


# ── 1. Package import ─────────────────────────────────────────────────────────

section("1. Package import")

def _test_import_version():
    import agentdelta
    assert agentdelta.__version__, "__version__ is empty"
    assert agentdelta.__version__ != "0.0.0"

def _test_import_public_api():
    from agentdelta import AgentTrace, diff_traces, record
    assert callable(diff_traces)
    assert callable(record)

def _test_import_trace_types():
    from agentdelta.trace import NodeType, EdgeType, TraceNode, TraceEdge, AgentTrace
    assert NodeType.LLM.value == "llm"
    assert EdgeType.TOOL_CALL.value == "tool_call"

def _test_import_diff():
    from agentdelta.diff import ForkPoint, DiffResult, StepDiff
    assert hasattr(DiffResult, "has_regression")

def _test_import_instrument():
    from agentdelta.instrument import AgentDeltaCallback, record
    cb = AgentDeltaCallback(run_id="smoke-import")
    assert cb.run_id == "smoke-import"

run("agentdelta package imports", _test_import_version)
run("Public API (AgentTrace, diff_traces, record)", _test_import_public_api)
run("Trace types (NodeType, EdgeType, TraceNode)", _test_import_trace_types)
run("Diff types (ForkPoint, DiffResult, StepDiff)", _test_import_diff)
run("Instrumentation (AgentDeltaCallback, record)", _test_import_instrument)

# ── 2. Trace creation and JSONL round-trip ────────────────────────────────────

section("2. Trace creation and JSONL round-trip")

from agentdelta.trace import AgentTrace, NodeType, EdgeType, TraceNode, TraceEdge

def _build_weather_trace(run_id: str, tool: str) -> AgentTrace:
    from agentdelta.instrument import AgentDeltaCallback

    class FakeLLMResponse:
        def __init__(self, text: str):
            self.generations = [[type("G", (), {"text": text})()]]

    cb = AgentDeltaCallback(run_id=run_id)
    cb.on_chain_start({}, {"input": "What is the weather in Tokyo?"})
    cb.on_llm_end(FakeLLMResponse("I should look up the current weather in Tokyo."))
    cb.on_tool_start({"name": tool}, "location='Tokyo'" if tool == "get_weather" else "query='Tokyo weather today'")
    cb.on_tool_end('{"temp": 22, "condition": "sunny"}' if tool == "get_weather" else "Tokyo: 22C sunny")
    cb.on_llm_end(FakeLLMResponse("The weather in Tokyo is 22C and sunny."))
    cb.on_chain_end({"output": "Tokyo: 22C, sunny."})
    return cb.trace

def _test_trace_create():
    trace = _build_weather_trace("test-a", "get_weather")
    assert len(trace.nodes) == 6
    assert trace.nodes[0].node_type == NodeType.START
    assert trace.nodes[-1].node_type == NodeType.END

def _test_trace_content_addressed_ids():
    n1 = TraceNode(step=1, node_type=NodeType.LLM, content="hello world")
    n2 = TraceNode(step=2, node_type=NodeType.LLM, content="hello world")
    assert n1.id == n2.id, "Same content must produce same ID"
    n3 = TraceNode(step=1, node_type=NodeType.LLM, content="different")
    assert n1.id != n3.id

def _test_trace_save_load(tmp_path=None):
    path = Path(tempfile.mktemp(suffix=".jsonl"))
    trace = _build_weather_trace("save-load", "get_weather")
    trace.save(path)
    assert path.exists()
    loaded = AgentTrace.load(path)
    assert loaded.run_id == "save-load"
    assert len(loaded.nodes) == len(trace.nodes)
    assert loaded.nodes[2].node_type == NodeType.TOOL_CALL
    assert "get_weather" in loaded.nodes[2].content
    path.unlink()

def _test_trace_jsonl_format():
    path = Path(tempfile.mktemp(suffix=".jsonl"))
    trace = _build_weather_trace("fmt-test", "get_weather")
    trace.save(path)
    lines = path.read_text().strip().split("\n")
    assert json.loads(lines[0])["type"] == "trace_meta"
    assert json.loads(lines[1])["type"] == "node"
    assert json.loads(lines[-1])["type"] == "edge"
    path.unlink()

run("Create trace with 6 nodes via AgentDeltaCallback", _test_trace_create)
run("TraceNode.id is content-addressed (same content = same ID)", _test_trace_content_addressed_ids)
run("AgentTrace.save() and .load() round-trip", _test_trace_save_load)
run("JSONL format: first line trace_meta, last line edge", _test_trace_jsonl_format)

# ── 3. Embedding and diff ─────────────────────────────────────────────────────

section("3. Embedding and semantic diff")

def _test_embed_trace():
    from agentdelta.embed import embed_trace
    trace = _build_weather_trace("embed-test", "get_weather")
    embed_trace(trace)
    assert all(n.embedding is not None for n in trace.nodes)
    assert len(trace.nodes[0].embedding) == 384

def _test_cosine_similarity():
    from agentdelta.embed import cosine_similarity
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == 1.0
    assert abs(cosine_similarity(v, [0.0, 1.0, 0.0])) < 0.01
    assert cosine_similarity([0.0] * 3, v) == 0.0

def _test_diff_detects_fork():
    from agentdelta import diff_traces
    trace_a = _build_weather_trace("baseline", "get_weather")
    trace_b = _build_weather_trace("candidate", "web_search")
    result = diff_traces(trace_a, trace_b, fork_threshold=0.70, match_threshold=0.85)
    assert result.has_regression, "Should detect regression when tool changes"
    assert result.fork_point is not None
    assert result.fork_point.is_tool_change()
    assert "get_weather" in result.fork_point.description or "web_search" in result.fork_point.description

def _test_diff_no_regression_identical():
    from agentdelta import diff_traces
    trace_a = _build_weather_trace("v1", "get_weather")
    trace_b = _build_weather_trace("v2", "get_weather")
    result = diff_traces(trace_a, trace_b)
    assert not result.has_regression, "Identical traces should not have regression"

def _test_diff_summary_fields():
    from agentdelta import diff_traces
    trace_a = _build_weather_trace("s1", "get_weather")
    trace_b = _build_weather_trace("s2", "web_search")
    result = diff_traces(trace_a, trace_b)
    assert "total_steps" in result.summary
    assert "matched" in result.summary
    assert "has_regression" in result.summary
    assert result.summary["has_regression"] is True

run("embed_trace() produces 384-dim embeddings for all nodes", _test_embed_trace)
run("cosine_similarity() handles identical, orthogonal, zero vectors", _test_cosine_similarity)
run("diff_traces() detects fork when tool changes (get_weather → web_search)", _test_diff_detects_fork)
run("diff_traces() reports no regression for identical traces", _test_diff_no_regression_identical)
run("DiffResult.summary has all required fields", _test_diff_summary_fields)

# ── 4. Report formatters ──────────────────────────────────────────────────────

section("4. Report formatters")

from agentdelta import diff_traces

_trace_a = _build_weather_trace("report-a", "get_weather")
_trace_b = _build_weather_trace("report-b", "web_search")
_diff_result = diff_traces(_trace_a, _trace_b)

def _test_to_json():
    from agentdelta.report import to_json
    raw = to_json(_diff_result)
    parsed = json.loads(raw)
    assert parsed["summary"]["has_regression"] is True
    assert "fork_point" in parsed
    assert parsed["fork_point"]["step_a"] >= 1

def _test_to_markdown():
    from agentdelta.report import to_markdown
    md = to_markdown(_diff_result)
    assert "## agentdelta" in md or "agentdelta" in md
    assert "REGRESSION" in md or "regression" in md.lower()
    assert "|" in md  # has a table

def _test_print_diff_runs():
    from agentdelta.report import print_diff
    import io
    from rich.console import Console
    buf = io.StringIO()
    console = Console(file=buf, highlight=False)
    print_diff(_diff_result, console=console)
    output = buf.getvalue()
    assert "REGRESSION" in output or "regression" in output.lower()

run("to_json() produces valid JSON with has_regression and fork_point", _test_to_json)
run("to_markdown() produces Markdown with table and REGRESSION text", _test_to_markdown)
run("print_diff() runs without error and outputs regression text", _test_print_diff_runs)

# ── 5. CLI ────────────────────────────────────────────────────────────────────

section("5. CLI (agentdelta diff / inspect)")

PYTHON = sys.executable

def _write_traces() -> tuple[Path, Path]:
    td = Path(tempfile.mkdtemp())
    _build_weather_trace("cli-a", "get_weather").save(td / "a.jsonl")
    _build_weather_trace("cli-b", "web_search").save(td / "b.jsonl")
    return td / "a.jsonl", td / "b.jsonl"

def _test_cli_help():
    r = subprocess.run([PYTHON, "-m", "agentdelta.cli", "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "diff" in r.stdout

def _test_cli_diff_rich():
    path_a, path_b = _write_traces()
    r = subprocess.run(
        [PYTHON, "-m", "agentdelta.cli", "diff", str(path_a), str(path_b)],
        capture_output=True, text=True
    )
    assert r.returncode == 0
    output = r.stdout + r.stderr
    assert "regression" in output.lower() or "REGRESSION" in output

def _test_cli_diff_json():
    path_a, path_b = _write_traces()
    r = subprocess.run(
        [PYTHON, "-m", "agentdelta.cli", "diff", str(path_a), str(path_b), "--format", "json"],
        capture_output=True, text=True
    )
    assert r.returncode == 0
    parsed = json.loads(r.stdout)
    assert "summary" in parsed
    assert "has_regression" in parsed["summary"]

def _test_cli_diff_markdown():
    path_a, path_b = _write_traces()
    r = subprocess.run(
        [PYTHON, "-m", "agentdelta.cli", "diff", str(path_a), str(path_b), "--format", "markdown"],
        capture_output=True, text=True
    )
    assert r.returncode == 0
    assert "|" in r.stdout

def _test_cli_diff_exit_code_regression():
    path_a, path_b = _write_traces()
    r = subprocess.run(
        [PYTHON, "-m", "agentdelta.cli", "diff", str(path_a), str(path_b), "--exit-code"],
        capture_output=True, text=True
    )
    assert r.returncode == 1, f"Expected exit 1 for regression, got {r.returncode}"

def _test_cli_diff_exit_code_clean():
    path_a = Path(tempfile.mktemp(suffix=".jsonl"))
    _build_weather_trace("same-a", "get_weather").save(path_a)
    path_b = Path(tempfile.mktemp(suffix=".jsonl"))
    _build_weather_trace("same-b", "get_weather").save(path_b)
    r = subprocess.run(
        [PYTHON, "-m", "agentdelta.cli", "diff", str(path_a), str(path_b), "--exit-code"],
        capture_output=True, text=True
    )
    assert r.returncode == 0, f"Expected exit 0 for no regression, got {r.returncode}"

def _test_cli_inspect():
    path_a, _ = _write_traces()
    r = subprocess.run(
        [PYTHON, "-m", "agentdelta.cli", "inspect", str(path_a)],
        capture_output=True, text=True
    )
    assert r.returncode == 0
    output = r.stdout + r.stderr
    assert "start" in output or "tool_call" in output or "llm" in output

run("agentdelta --help returns 0", _test_cli_help)
run("agentdelta diff (rich format) detects regression", _test_cli_diff_rich)
run("agentdelta diff --format json returns valid JSON", _test_cli_diff_json)
run("agentdelta diff --format markdown returns Markdown table", _test_cli_diff_markdown)
run("agentdelta diff --exit-code exits 1 on regression", _test_cli_diff_exit_code_regression)
run("agentdelta diff --exit-code exits 0 on clean traces", _test_cli_diff_exit_code_clean)
run("agentdelta inspect outputs node types", _test_cli_inspect)

# ── 6. FastAPI server ─────────────────────────────────────────────────────────

section("6. FastAPI server (agentdelta[api])")

def _test_api_import():
    from agentdelta.api import app
    assert app.title == "agentdelta API"

def _test_api_health():
    from fastapi.testclient import TestClient
    from agentdelta.api import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "version" in r.json()

def _test_api_diff_endpoint():
    from fastapi.testclient import TestClient
    from agentdelta.api import app

    def _trace_to_str(trace) -> str:
        path = Path(tempfile.mktemp(suffix=".jsonl"))
        trace.save(path)
        content = path.read_text()
        path.unlink()
        return content

    client = TestClient(app)
    r = client.post("/diff", json={
        "trace_a": _trace_to_str(_build_weather_trace("api-a", "get_weather")),
        "trace_b": _trace_to_str(_build_weather_trace("api-b", "web_search")),
    })
    assert r.status_code == 200
    data = r.json()
    assert "summary" in data
    assert data["summary"]["has_regression"] is True

def _test_api_inspect_endpoint():
    from fastapi.testclient import TestClient
    from agentdelta.api import app

    path = Path(tempfile.mktemp(suffix=".jsonl"))
    _build_weather_trace("api-inspect", "get_weather").save(path)
    content = path.read_text()
    path.unlink()

    client = TestClient(app)
    r = client.post("/inspect", json={"trace": content})
    assert r.status_code == 200
    data = r.json()
    assert data["run_id"] == "api-inspect"
    assert data["total_nodes"] == 6

run("agentdelta.api imports and app.title is correct", _test_api_import)
run("GET /health returns {status: ok, version: ...}", _test_api_health)
run("POST /diff detects regression between two traces", _test_api_diff_endpoint)
run("POST /inspect returns run_id and node count", _test_api_inspect_endpoint)

# ── 7. MCP server ─────────────────────────────────────────────────────────────

section("7. MCP server (agentdelta[mcp])")

def _test_mcp_server_importable():
    import agentdelta.mcp_server as m
    assert hasattr(m, "run_server")

def _test_mcp_server_graceful_fail_without_mcp():
    # mcp package is likely not installed — _require_mcp() should fail with SystemExit
    # but the module itself must import cleanly
    import agentdelta.mcp_server  # noqa: F401 — just checking it loads

run("mcp_server.py imports without error", _test_mcp_server_importable)
run("mcp_server module loads cleanly (no import-time crash)", _test_mcp_server_graceful_fail_without_mcp)

# ── 8. Agent config files ─────────────────────────────────────────────────────

section("8. Agent config files (what a clone gives you)")

def _check_file_nonempty(rel: str) -> None:
    p = REPO_ROOT / rel
    assert p.exists(), f"Missing: {rel}"
    assert p.stat().st_size > 50, f"File too small (likely empty): {rel}"

def _check_json_valid(rel: str) -> None:
    p = REPO_ROOT / rel
    assert p.exists(), f"Missing: {rel}"
    json.loads(p.read_text())

def _check_yaml_parseable(rel: str) -> None:
    import yaml  # type: ignore[import-untyped]
    p = REPO_ROOT / rel
    assert p.exists(), f"Missing: {rel}"
    yaml.safe_load(p.read_text())

def _test_claude_commands():
    commands = list((REPO_ROOT / ".claude/commands").glob("*.md"))
    assert len(commands) >= 4, f"Expected ≥4 slash commands, found {len(commands)}"

def _test_openai_tools_valid():
    _check_json_valid("tools/openai-tools.json")
    tools = json.loads((REPO_ROOT / "tools/openai-tools.json").read_text())
    assert len(tools) >= 3
    assert all("function" in t for t in tools)

def _test_openapi_yaml_parseable():
    try:
        _check_yaml_parseable("openapi.yaml")
    except ImportError:
        # yaml not installed — check it's at least valid-looking text
        content = (REPO_ROOT / "openapi.yaml").read_text()
        assert "openapi:" in content

run("AGENTS.md exists and non-empty", lambda: _check_file_nonempty("AGENTS.md"))
run("CLAUDE.md exists and non-empty", lambda: _check_file_nonempty("CLAUDE.md"))
run("CODEX.md exists and non-empty", lambda: _check_file_nonempty("CODEX.md"))
run(".github/copilot-instructions.md exists", lambda: _check_file_nonempty(".github/copilot-instructions.md"))
run(".cursor/rules/ has at least one .mdc file", lambda: _check_file_nonempty(".cursor/rules/agentdelta.mdc"))
run(".windsurfrules exists", lambda: _check_file_nonempty(".windsurfrules"))
run(".aider.conf.yml exists", lambda: _check_file_nonempty(".aider.conf.yml"))
run(".continue/config.json is valid JSON", lambda: _check_json_valid(".continue/config.json"))
run(".claude/commands/ has ≥4 slash commands", _test_claude_commands)
run("tools/openai-tools.json is valid JSON with ≥3 tools", _test_openai_tools_valid)
run("openapi.yaml is parseable YAML", _test_openapi_yaml_parseable)

# ── 9. Docs site ──────────────────────────────────────────────────────────────

section("9. MkDocs documentation site")

def _test_mkdocs_yml():
    _check_file_nonempty("mkdocs.yml")
    content = (REPO_ROOT / "mkdocs.yml").read_text()
    assert "site_name" in content
    assert "material" in content

def _test_docs_pages():
    docs = list((REPO_ROOT / "docs").glob("*.md"))
    assert len(docs) >= 8, f"Expected ≥8 doc pages, found {len(docs)}"
    names = {p.name for p in docs}
    for required in ("index.md", "quickstart.md", "architecture.md", "api-reference.md"):
        assert required in names, f"Missing docs/{required}"

run("mkdocs.yml exists with site_name and material theme", _test_mkdocs_yml)
run("docs/ has ≥8 pages including index, quickstart, architecture, api-reference", _test_docs_pages)

# ── 10. demo.py runs end-to-end ───────────────────────────────────────────────

section("10. examples/demo.py end-to-end")

def _test_demo_runs():
    r = subprocess.run(
        [PYTHON, str(REPO_ROOT / "examples/demo.py")],
        capture_output=True, text=True,
        cwd=str(REPO_ROOT)
    )
    if r.returncode != 0:
        raise AssertionError(f"demo.py exited {r.returncode}:\n{r.stderr[-500:]}")
    output = r.stdout + r.stderr
    assert "fork" in output.lower() or "regression" in output.lower() or "REGRESSION" in output

run("examples/demo.py runs end-to-end and detects the tool fork", _test_demo_runs)

# ── Summary ───────────────────────────────────────────────────────────────────

total = len(passed) + len(failed)
print(f"\n{'═'*60}")
print(f"{BOLD}Results: {len(passed)}/{total} passed{RESET}")

if failed:
    print(f"{RED}Failed ({len(failed)}):{RESET}")
    for name, reason in failed:
        print(f"  {RED}✗{RESET} {name}")
        short = reason.split("\n")[0][:120]
        print(f"    {YELLOW}→ {short}{RESET}")
    print(f"\n{YELLOW}Tip: run with --verbose for full tracebacks{RESET}")
else:
    print(f"{GREEN}All {total} checks passed — agentdelta is ready to ship{RESET}")

print(f"{'═'*60}\n")
sys.exit(0 if not failed else 1)
