"""Tool-call brittleness gate — BITTER-TOOL (arXiv 2608.06370).

Public case: *The Bitter Lesson of Tool Calling* (Track B research). Tool use
turns LLMs into agents; failures concentrate on **invocation shape** (JSON vs
programmatic), **orphan calls** (no result), **silent tool errors** under a
clean END claim, and **context-rot** reuse of stale tool results.

Twin of TRAJDEBUG (generic intermediate errors): this module is **tool-path
specific** — schema completeness, call↔return pairing, parallel partial fail,
and stale result reuse.

Non-Ornament:
  Call ``gate_tool_calls`` in CI on agent traces before accepting a run as
  tool-success. Pair with ``gate_error_lifecycle`` for non-tool intermediate
  failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

from agentdelta.closed_loop import ClosedLoopError, GateOutcome
from agentdelta.trace import AgentTrace, NodeType, TraceNode

CallStyle = Literal["json", "programmatic", "unknown"]

_FAIL_STATUSES = frozenset(
    {"error", "fail", "failed", "timeout", "exception", "denied", "invalid"}
)
_OK_STATUSES = frozenset({"ok", "success", "pass", "passed", "done", "completed", ""})


@dataclass(frozen=True)
class ToolCallEvent:
    """One tool invocation (request side)."""

    call_id: str
    name: str
    step: int
    arguments: dict[str, Any] = field(default_factory=dict)
    style: CallStyle = "unknown"
    parallel_group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "step": self.step,
            "arguments": dict(self.arguments),
            "style": self.style,
            "parallel_group": self.parallel_group,
        }


@dataclass(frozen=True)
class ToolResultEvent:
    """One tool result (return side)."""

    call_id: str
    name: str
    step: int
    status: str = "ok"
    content: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "step": self.step,
            "status": self.status,
            "content": self.content,
            "error": self.error,
        }


@dataclass(frozen=True)
class ToolCallAnalysis:
    """Structured analysis of tool-call hygiene on a trajectory."""

    call_count: int
    result_count: int
    orphan_call_ids: tuple[str, ...]
    failed_result_ids: tuple[str, ...]
    schema_invalid_ids: tuple[str, ...]
    parallel_partial_groups: tuple[str, ...]
    stale_reuse_ids: tuple[str, ...]
    claimed_success: bool
    styles: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_count": self.call_count,
            "result_count": self.result_count,
            "orphan_call_ids": list(self.orphan_call_ids),
            "failed_result_ids": list(self.failed_result_ids),
            "schema_invalid_ids": list(self.schema_invalid_ids),
            "parallel_partial_groups": list(self.parallel_partial_groups),
            "stale_reuse_ids": list(self.stale_reuse_ids),
            "claimed_success": self.claimed_success,
            "styles": list(self.styles),
        }


def _status_failed(status: str) -> bool:
    return (status or "").strip().lower() in _FAIL_STATUSES


def _infer_style(meta: dict[str, Any], content: str) -> CallStyle:
    explicit = str(meta.get("style") or meta.get("call_style") or "").strip().lower()
    if explicit in {"json", "programmatic", "ptc", "code"}:
        if explicit in {"ptc", "code"}:
            return "programmatic"
        return explicit  # type: ignore[return-value]
    # Heuristic: code-looking content → programmatic
    blob = (content or "") + " " + str(meta.get("code") or "")
    if "import " in blob or "def " in blob or "await " in blob or "```" in blob:
        return "programmatic"
    if meta.get("arguments") is not None or meta.get("args") is not None:
        return "json"
    return "unknown"


def _node_meta(node: TraceNode) -> dict[str, Any]:
    meta = getattr(node, "metadata", None) or getattr(node, "meta", None) or {}
    return dict(meta) if isinstance(meta, dict) else {}


def _events_from_trace(trace: AgentTrace) -> tuple[list[ToolCallEvent], list[ToolResultEvent], bool]:
    calls: list[ToolCallEvent] = []
    results: list[ToolResultEvent] = []
    step_i = 0
    for node in trace.nodes:
        step_i += 1
        meta = _node_meta(node)
        name = str(
            meta.get("tool")
            or meta.get("name")
            or getattr(node, "content", "")
            or f"tool_{step_i}"
        )[:120]
        call_id = str(
            meta.get("call_id")
            or meta.get("tool_call_id")
            or meta.get("id")
            or f"{name}@{step_i}"
        )
        if node.node_type == NodeType.TOOL_CALL:
            args = meta.get("arguments") or meta.get("args") or {}
            if not isinstance(args, dict):
                args = {"_raw": args}
            content = str(getattr(node, "content", "") or "")
            calls.append(
                ToolCallEvent(
                    call_id=call_id,
                    name=name,
                    step=step_i,
                    arguments=dict(args),
                    style=_infer_style(meta, content),
                    parallel_group=(
                        str(meta["parallel_group"])
                        if meta.get("parallel_group") is not None
                        else None
                    ),
                )
            )
        elif node.node_type == NodeType.TOOL_RETURN:
            status = str(meta.get("status") or "ok").strip().lower()
            err = str(meta.get("error") or "")
            content = str(getattr(node, "content", "") or "")
            if meta.get("ok") is False or meta.get("success") is False:
                status = "error"
            # content heuristics for errors
            low = content.lower()
            if any(tok in low for tok in ("traceback", "exception:", "error:", "timeout")):
                if status in _OK_STATUSES:
                    status = "error"
                    err = err or content[:200]
            results.append(
                ToolResultEvent(
                    call_id=call_id,
                    name=name,
                    step=step_i,
                    status=status,
                    content=content,
                    error=err,
                )
            )

    ends = [n for n in trace.nodes if n.node_type == NodeType.END]
    claimed = bool(ends) and not any(
        _status_failed(str(_node_meta(n).get("status") or "ok")) for n in ends
    )
    return calls, results, claimed


def _events_from_sequences(
    calls: Sequence[ToolCallEvent | dict[str, Any]] | None,
    results: Sequence[ToolResultEvent | dict[str, Any]] | None,
    *,
    claimed_success: bool,
) -> tuple[list[ToolCallEvent], list[ToolResultEvent], bool]:
    out_c: list[ToolCallEvent] = []
    out_r: list[ToolResultEvent] = []
    if calls:
        for i, c in enumerate(calls):
            if isinstance(c, ToolCallEvent):
                out_c.append(c)
                continue
            if not isinstance(c, dict):
                raise TypeError(f"tool call must be ToolCallEvent or dict, got {type(c)!r}")
            cid = str(c.get("call_id") or c.get("id") or f"call_{i+1}")
            name = str(c.get("name") or c.get("tool") or "tool")
            step = int(c.get("step", i + 1))
            args = c.get("arguments") or c.get("args") or {}
            if not isinstance(args, dict):
                args = {"_raw": args}
            style_raw = str(c.get("style") or "unknown").lower()
            style: CallStyle = (
                "programmatic"
                if style_raw in {"programmatic", "ptc", "code"}
                else "json"
                if style_raw == "json"
                else "unknown"
            )
            pg = c.get("parallel_group")
            out_c.append(
                ToolCallEvent(
                    call_id=cid,
                    name=name,
                    step=step,
                    arguments=dict(args),
                    style=style,
                    parallel_group=str(pg) if pg is not None else None,
                )
            )
    if results:
        for i, r in enumerate(results):
            if isinstance(r, ToolResultEvent):
                out_r.append(r)
                continue
            if not isinstance(r, dict):
                raise TypeError(f"tool result must be ToolResultEvent or dict, got {type(r)!r}")
            cid = str(r.get("call_id") or r.get("id") or f"result_{i+1}")
            name = str(r.get("name") or r.get("tool") or "tool")
            step = int(r.get("step", i + 1))
            status = str(r.get("status") or "ok").strip().lower()
            if r.get("ok") is False or r.get("success") is False:
                status = "error"
            out_r.append(
                ToolResultEvent(
                    call_id=cid,
                    name=name,
                    step=step,
                    status=status,
                    content=str(r.get("content") or r.get("output") or ""),
                    error=str(r.get("error") or ""),
                )
            )
    return out_c, out_r, claimed_success


def _schema_invalid(
    call: ToolCallEvent,
    schemas: dict[str, Sequence[str]],
) -> bool:
    """True if required argument names for this tool are missing."""
    required = schemas.get(call.name) or schemas.get(call.name.lower())
    if not required:
        return False
    args = call.arguments or {}
    for key in required:
        if key not in args or args[key] is None or args[key] == "":
            return True
    return False


def analyze_tool_calls(
    source: AgentTrace
    | None = None,
    *,
    calls: Sequence[ToolCallEvent | dict[str, Any]] | None = None,
    results: Sequence[ToolResultEvent | dict[str, Any]] | None = None,
    claimed_success: bool | None = None,
    schemas: dict[str, Sequence[str]] | None = None,
    max_result_age_steps: int | None = None,
    total_steps: int | None = None,
) -> ToolCallAnalysis:
    """Analyse tool-call hygiene (pairing, schema, parallel partial, stale reuse).

    Args:
        source: Optional :class:`AgentTrace` with TOOL_CALL / TOOL_RETURN nodes.
        calls / results: Explicit event sequences (alternative to trace).
        claimed_success: Override success claim.
        schemas: Map tool name → required argument names.
        max_result_age_steps: If set, a result used after this many steps from
            its call without a newer call of the same name is **stale reuse**
            (context-rot class from the paper).
        total_steps: Horizon length for stale check (default max step seen).
    """
    if source is not None:
        c_list, r_list, inferred = _events_from_trace(source)
        if claimed_success is None:
            claimed_success = inferred
    else:
        if claimed_success is None:
            claimed_success = True
        c_list, r_list, claimed_success = _events_from_sequences(
            calls, results, claimed_success=claimed_success
        )

    result_by_id: dict[str, ToolResultEvent] = {}
    for r in r_list:
        # last result wins for same call_id
        result_by_id[r.call_id] = r

    orphans: list[str] = []
    for c in c_list:
        if c.call_id not in result_by_id:
            orphans.append(c.call_id)

    failed: list[str] = []
    for r in r_list:
        if _status_failed(r.status) or r.error:
            failed.append(r.call_id)

    schema_bad: list[str] = []
    if schemas:
        for c in c_list:
            if _schema_invalid(c, schemas):
                schema_bad.append(c.call_id)

    # Parallel partial: group has mixed ok/fail results
    groups: dict[str, list[ToolResultEvent]] = {}
    call_group = {c.call_id: c.parallel_group for c in c_list if c.parallel_group}
    for r in r_list:
        g = call_group.get(r.call_id)
        if g:
            groups.setdefault(g, []).append(r)
    partial: list[str] = []
    for g, rs in groups.items():
        statuses = {_status_failed(r.status) or bool(r.error) for r in rs}
        if True in statuses and False in statuses:
            partial.append(g)
        elif True in statuses and len(rs) < sum(
            1 for c in c_list if c.parallel_group == g
        ):
            partial.append(g)

    # Stale reuse: end uses tool output older than max_result_age_steps
    stale: list[str] = []
    if max_result_age_steps is not None and max_result_age_steps >= 0:
        horizon = total_steps
        if horizon is None:
            horizon = max(
                [c.step for c in c_list] + [r.step for r in r_list] + [0]
            )
        for r in r_list:
            age = int(horizon) - int(r.step)
            if age > max_result_age_steps and not _status_failed(r.status):
                # only flag if a later END/success claim exists (claimed_success)
                if claimed_success:
                    stale.append(r.call_id)

    styles = tuple(sorted({c.style for c in c_list}))

    return ToolCallAnalysis(
        call_count=len(c_list),
        result_count=len(r_list),
        orphan_call_ids=tuple(orphans),
        failed_result_ids=tuple(dict.fromkeys(failed)),
        schema_invalid_ids=tuple(schema_bad),
        parallel_partial_groups=tuple(partial),
        stale_reuse_ids=tuple(dict.fromkeys(stale)),
        claimed_success=bool(claimed_success),
        styles=styles,
    )


def gate_tool_calls(
    source: AgentTrace | None = None,
    *,
    calls: Sequence[ToolCallEvent | dict[str, Any]] | None = None,
    results: Sequence[ToolResultEvent | dict[str, Any]] | None = None,
    claimed_success: bool | None = None,
    schemas: dict[str, Sequence[str]] | None = None,
    require_tools: bool = False,
    refuse_orphans: bool = True,
    refuse_failed_with_success: bool = True,
    refuse_schema_invalid: bool = True,
    refuse_parallel_partial: bool = True,
    max_result_age_steps: int | None = None,
    total_steps: int | None = None,
    prefer_programmatic: bool = False,
) -> GateOutcome:
    """Refuse brittle tool-call trajectories (BITTER-TOOL / arXiv 2608.06370).

    Rules:

    * No calls when ``require_tools`` → **FAIL_LOUD**
    * Orphan TOOL_CALL (no return) → **FAIL**
    * Failed tool result + claimed success → **FAIL** (silent tool error)
    * Missing required schema args → **FAIL**
    * Parallel group mixed success/fail → **FAIL**
    * Stale result reuse beyond ``max_result_age_steps`` → **FAIL** (context rot)
    * ``prefer_programmatic`` and only json/unknown styles with failures → **WARN/FAIL**
      only if other failures; pure style preference alone is **WARN** not hard fail
    * Clean paired tool path → **PASS**
    """
    # Empty input?
    if source is None and not calls and not results:
        if require_tools:
            return GateOutcome(
                ok=False,
                verdict="FAIL_LOUD",
                reason=(
                    "BITTER-TOOL: no tool calls or results — require_tools=True "
                    "(write-only ornament / no tool path evidence; arXiv 2608.06370)"
                ),
                exit_code=2,
                human_required=True,
            )
        return GateOutcome(
            ok=True,
            verdict="PASS",
            reason="BITTER-TOOL: no tools required; nothing to gate",
            exit_code=0,
            human_required=False,
        )

    try:
        analysis = analyze_tool_calls(
            source,
            calls=calls,
            results=results,
            claimed_success=claimed_success,
            schemas=schemas,
            max_result_age_steps=max_result_age_steps,
            total_steps=total_steps,
        )
    except (TypeError, ValueError) as exc:
        return GateOutcome(
            ok=False,
            verdict="FAIL_LOUD",
            reason=f"BITTER-TOOL: invalid tool events: {exc}",
            exit_code=2,
            human_required=True,
        )

    run_id = getattr(source, "run_id", None) if isinstance(source, AgentTrace) else None

    if require_tools and analysis.call_count == 0:
        return GateOutcome(
            ok=False,
            verdict="FAIL_LOUD",
            reason=(
                "BITTER-TOOL: require_tools but call_count=0 — agent claimed a "
                "tool-using task without any TOOL_CALL evidence"
            ),
            exit_code=2,
            error_step_count=0,
            human_required=True,
            run_id_a=run_id,
        )

    if refuse_orphans and analysis.orphan_call_ids:
        return GateOutcome(
            ok=False,
            verdict="FAIL",
            reason=(
                f"BITTER-TOOL: {len(analysis.orphan_call_ids)} orphan tool call(s) "
                f"without TOOL_RETURN ids={list(analysis.orphan_call_ids)[:8]} — "
                "refuse incomplete tool path (arXiv 2608.06370)"
            ),
            exit_code=1,
            error_step_count=len(analysis.orphan_call_ids),
            critical_step=None,
            human_required=True,
            run_id_a=run_id,
        )

    if (
        refuse_failed_with_success
        and analysis.claimed_success
        and analysis.failed_result_ids
    ):
        return GateOutcome(
            ok=False,
            verdict="FAIL",
            reason=(
                f"BITTER-TOOL: claimed success with {len(analysis.failed_result_ids)} "
                f"failed tool result(s) ids={list(analysis.failed_result_ids)[:8]} — "
                "silent tool error under clean END (bitter tool-calling lesson)"
            ),
            exit_code=1,
            error_step_count=len(analysis.failed_result_ids),
            human_required=True,
            run_id_a=run_id,
        )

    if refuse_schema_invalid and analysis.schema_invalid_ids:
        return GateOutcome(
            ok=False,
            verdict="FAIL",
            reason=(
                f"BITTER-TOOL: {len(analysis.schema_invalid_ids)} tool call(s) missing "
                f"required schema args ids={list(analysis.schema_invalid_ids)[:8]} — "
                "JSON/tool stub contract broken"
            ),
            exit_code=1,
            error_step_count=len(analysis.schema_invalid_ids),
            human_required=True,
            run_id_a=run_id,
        )

    if refuse_parallel_partial and analysis.parallel_partial_groups:
        return GateOutcome(
            ok=False,
            verdict="FAIL",
            reason=(
                f"BITTER-TOOL: parallel fan-out partial failure groups="
                f"{list(analysis.parallel_partial_groups)[:8]} — some arms failed "
                "while others succeeded; refuse incomplete fan-out merge"
            ),
            exit_code=1,
            error_step_count=len(analysis.parallel_partial_groups),
            human_required=True,
            run_id_a=run_id,
        )

    if analysis.stale_reuse_ids:
        return GateOutcome(
            ok=False,
            verdict="FAIL",
            reason=(
                f"BITTER-TOOL: context-rot stale tool result reuse "
                f"ids={list(analysis.stale_reuse_ids)[:8]} "
                f"max_age_steps={max_result_age_steps} — re-invoke tool before "
                "treating result as current"
            ),
            exit_code=1,
            error_step_count=len(analysis.stale_reuse_ids),
            human_required=True,
            run_id_a=run_id,
        )

    # Soft note when prefer_programmatic and only json — WARN still ok=True
    style_note = ""
    if prefer_programmatic and analysis.styles and "programmatic" not in analysis.styles:
        style_note = (
            f" (prefer_programmatic: styles={list(analysis.styles)} — "
            "paper finds PTC matches/exceeds JSON on BFCL v4; not hard-fail)"
        )
        return GateOutcome(
            ok=True,
            verdict="WARN",
            reason=(
                f"BITTER-TOOL warn: calls={analysis.call_count} results="
                f"{analysis.result_count} paired ok but no programmatic style"
                f"{style_note}"
            ),
            exit_code=0,
            error_step_count=0,
            human_required=False,
            run_id_a=run_id,
        )

    return GateOutcome(
        ok=True,
        verdict="PASS",
        reason=(
            f"BITTER-TOOL ok: calls={analysis.call_count} results="
            f"{analysis.result_count} orphans=0 failed_under_success=0 "
            f"styles={list(analysis.styles)}"
        ),
        exit_code=0,
        error_step_count=0,
        human_required=False,
        run_id_a=run_id,
    )


def assert_tool_calls_ok(
    source: AgentTrace | None = None,
    **kwargs: Any,
) -> GateOutcome:
    """Raise :class:`ClosedLoopError` unless :func:`gate_tool_calls` is ok."""
    outcome = gate_tool_calls(source, **kwargs)
    if not outcome.ok:
        raise ClosedLoopError(f"{outcome.verdict}: {outcome.reason}")
    return outcome


def extract_tool_events(
    trace: AgentTrace,
) -> tuple[list[ToolCallEvent], list[ToolResultEvent]]:
    """Public helper: pull tool call/result events from a trace."""
    calls, results, _ = _events_from_trace(trace)
    return calls, results
