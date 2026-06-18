"""FastAPI REST wrapper for redline.

Start with: uvicorn redline.api:app --reload
Install:    pip install "redline[api]"

Implements the openapi.yaml contract:
    POST /diff      — compare two JSONL traces
    POST /inspect   — summarise a single trace
    GET  /health    — liveness probe
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(
        "API server requires: pip install 'redline[api]'"
    ) from exc

from redline import AgentTrace, diff_traces
from redline.report import to_json
from redline.trace import NodeType

app = FastAPI(
    title="redline API",
    description="Semantic diff engine for AI agent behavior.",
    version="0.1.0",
    license_info={"name": "MIT", "url": "https://github.com/sandeep-alluru/redline/blob/main/LICENSE"},
)


# ── Request / Response models ─────────────────────────────────────────────────


class DiffRequest(BaseModel):
    trace_a: str = Field(..., description="Baseline JSONL trace content")
    trace_b: str = Field(..., description="Candidate JSONL trace content")
    fork_threshold: float = Field(0.70, ge=0.0, le=1.0, description="Fork detection threshold")
    match_threshold: float = Field(0.85, ge=0.0, le=1.0, description="Match detection threshold")


class InspectRequest(BaseModel):
    trace: str = Field(..., description="JSONL trace content to inspect")


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_trace_from_string(content: str, name: str) -> AgentTrace:
    """Write content to a temp file and load it as an AgentTrace."""
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, prefix=f"redline_{name}_"
        ) as f:
            f.write(content)
            tmp_path = f.name
        return AgentTrace.load(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid {name} trace: {exc}") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> dict[str, str]:
    """Liveness probe."""
    from redline import __version__
    return {"status": "ok", "version": __version__}


@app.post("/diff")
async def diff(request: DiffRequest) -> Any:
    """Compare two agent traces and return a DiffResult with fork point."""
    trace_a = _load_trace_from_string(request.trace_a, "trace_a")
    trace_b = _load_trace_from_string(request.trace_b, "trace_b")

    result = diff_traces(
        trace_a,
        trace_b,
        fork_threshold=request.fork_threshold,
        match_threshold=request.match_threshold,
    )
    return json.loads(to_json(result))


@app.post("/inspect")
async def inspect(request: InspectRequest) -> Any:
    """Summarise a single agent trace."""
    trace = _load_trace_from_string(request.trace, "trace")

    steps = [
        {
            "step": node.step,
            "type": node.node_type.value,
            "content_preview": node.content[:120],
            "id": node.id,
        }
        for node in trace.nodes
    ]

    node_type_counts: dict[str, int] = {}
    for node in trace.nodes:
        key = node.node_type.value
        node_type_counts[key] = node_type_counts.get(key, 0) + 1

    return {
        "run_id": trace.run_id,
        "total_nodes": len(trace.nodes),
        "total_edges": len(trace.edges),
        "node_type_counts": node_type_counts,
        "has_tool_calls": any(n.node_type == NodeType.TOOL_CALL for n in trace.nodes),
        "steps": steps,
        "metadata": trace.metadata,
    }
