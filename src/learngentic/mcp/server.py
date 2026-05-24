"""
Learngentic MCP server.

Exposes five tools to Hermes (and Claude Code):
  - query_risk_assessment   pre-execution: should we ask clarifying questions?
  - query_outcome_history   find similar past sessions and their scores
  - get_file_stability      stability stats for a specific file
  - record_task             start of a Hermes task; creates session row + returns TaskStandard
  - report_outcome          end of a Hermes task; records self-assessment + checklist verdict

Run via: python -m learngentic.mcp.server
Register in Claude Code / Hermes settings as a stdio MCP server.
"""
from __future__ import annotations

import collections
import json
import logging
import math
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from learngentic.store.db import get_conn
from learngentic.store.vector_index import search_similar, index_session
from learngentic.classifier import classify_session
from learngentic.scoring.efficiency import efficiency_score

app = Server("learngentic")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool-call session tracking (Features 1 + 5)
# ---------------------------------------------------------------------------

_call_log_lock = threading.Lock()
# session_id → list of {tool, ts} entries accumulated during the session
_session_call_log: dict[str, list[dict]] = {}
# Ring buffer of recent calls (max 50) — used to capture pre-task guidance calls
# that happen before record_task creates a session_id
_recent_tool_calls: collections.deque[dict] = collections.deque(maxlen=50)

_GUIDANCE_TOOLS = frozenset({
    "query_risk_assessment",
    "query_outcome_history",
    "record_task",
    "run_local_task",
    "report_local_result",
    "report_outcome",
})


def _ensure_api_key() -> None:
    from learngentic.store.db import _load_config
    _load_config()  # raises RuntimeError with a clear message if credentials are missing


# ---------------------------------------------------------------------------
# Tool: query_risk_assessment
# ---------------------------------------------------------------------------

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_risk_assessment",
            description=(
                "Pre-execution risk check. Given a task prompt and working directory, "
                "returns a risk level and specific clarifying questions grounded in "
                "historical failure patterns for this type of task. Call this before "
                "starting any non-trivial task."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The task prompt"},
                    "cwd": {"type": "string", "description": "Current working directory (repo path)"},
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="query_outcome_history",
            description=(
                "Find the most similar past sessions and their outcome scores. "
                "Returns blended global + project-specific history. "
                "Use this to understand how similar tasks have gone before."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Description of the current task"},
                    "cwd": {"type": "string", "description": "Current working directory (for project-level filtering)"},
                    "k": {"type": "integer", "description": "Number of results to return (default 5)", "default": 5},
                },
                "required": ["description"],
            },
        ),
        Tool(
            name="get_file_stability",
            description=(
                "Return stability statistics for a specific file: "
                "how often it gets changed, revert rate, and average churn. "
                "High churn files are risky to modify without extra care."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Repo-relative or absolute file path"},
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="record_task",
            description=(
                "Register the start of a Hermes task. Creates a session row in Learngentic, "
                "classifies the prompt, generates a TaskStandard (benchmarks + completion checklist), "
                "and checks for recent file conflicts. "
                "Returns a session_id that MUST be passed to report_outcome when the task finishes. "
                "Call this before starting any non-trivial task (>3 turns expected)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt":          {"type": "string",  "description": "The task prompt"},
                    "cwd":             {"type": "string",  "description": "Working directory (repo path)"},
                    "files_mentioned": {
                        "type":        "array",
                        "items":       {"type": "string"},
                        "description": "Files the task is expected to touch (optional, for conflict detection)",
                    },
                    "agent_type":      {"type": "string",  "description": "Agent identifier, default 'hermes'"},
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="report_outcome",
            description=(
                "Record the outcome of a completed Hermes task. "
                "Computes a self-assessment score, stores it separately from outcome_quality, "
                "triggers async prompt grading, and returns a pass/fail verdict against the TaskStandard. "
                "Call this before returning a completed result to the user."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id":      {"type": "string",  "description": "session_id from record_task"},
                    "sent_back":       {"type": "boolean", "description": "Did the user reject or ask for revision?"},
                    "already_existed": {"type": "boolean", "description": "Was the code being built already present?"},
                    "is_rewrite":      {"type": "boolean", "description": "Is this rewriting something built recently?"},
                    "turn_count":      {"type": "integer", "description": "Number of internal turns taken"},
                    "hermes_notes":    {"type": "string",  "description": "Optional free-text notes about the task"},
                    "checklist_passed": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "0-based indices of completion_checklist items you satisfied. "
                            "Omit if none. Used to compute checklist_compliance rate."
                        ),
                    },
                },
                "required": ["session_id", "sent_back", "already_existed", "is_rewrite", "turn_count"],
            },
        ),
        Tool(
            name="run_local_task",
            description=(
                "Delegate a task to a local Ollama model. "
                "The server classifies the task, checks whether a local model is capable based on "
                "historical pass rates, picks the right model, runs it, and logs the attempt. "
                "If local_capable is false in the response, handle the task yourself — do not retry. "
                "On a failed attempt, call again with attempt_number incremented, previous_run_id set "
                "to the last run_id, and a refined system_prompt_override. Max 5 attempts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description":      {"type": "string",  "description": "What you want done — used for classification and logging"},
                    "user_input":            {"type": "string",  "description": "The content to process (diff, code, text, etc.)"},
                    "attempt_number":        {"type": "integer", "description": "Attempt number 1–5 (default 1)"},
                    "system_prompt_override":{"type": "string",  "description": "Override the default system prompt — use on retries to refine"},
                    "previous_run_id":       {"type": "string",  "description": "run_id from the previous failed attempt — marks it as failed"},
                    "session_id":            {"type": "string",  "description": "Optional Learngentic session_id to link this attempt"},
                },
                "required": ["task_description", "user_input"],
            },
        ),
        Tool(
            name="report_local_result",
            description=(
                "Record whether a local model attempt passed or failed. "
                "Call with verdict='pass' when you accept the output, 'fail' when retrying, "
                "or 'gave_up' after 5 failed attempts. This closes the feedback loop for "
                "fine-tuning data collection."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id":  {"type": "string", "description": "run_id from run_local_task"},
                    "verdict": {"type": "string", "description": "'pass', 'fail', or 'gave_up'"},
                },
                "required": ["run_id", "verdict"],
            },
        ),
    ]


def _record_tool_call(name: str, arguments: dict) -> None:
    """Log every tool call to the ring buffer and to any active session log."""
    ts = datetime.now(timezone.utc).isoformat()
    entry = {"tool": name, "ts": ts}
    with _call_log_lock:
        _recent_tool_calls.append(entry)
        session_id = arguments.get("session_id")
        if session_id and session_id in _session_call_log:
            _session_call_log[session_id].append(entry)


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    _record_tool_call(name, arguments)
    if name == "query_risk_assessment":
        result = _risk_assessment(
            prompt=arguments.get("prompt", ""),
            cwd=arguments.get("cwd", ""),
        )
    elif name == "query_outcome_history":
        result = _outcome_history(
            description=arguments.get("description", ""),
            cwd=arguments.get("cwd", ""),
            k=int(arguments.get("k", 5)),
        )
    elif name == "get_file_stability":
        result = _file_stability(file_path=arguments.get("file_path", ""))
    elif name == "record_task":
        result = _record_task(
            prompt=arguments.get("prompt", ""),
            cwd=arguments.get("cwd", ""),
            files_mentioned=arguments.get("files_mentioned", []),
            agent_type=arguments.get("agent_type", "hermes"),
        )
    elif name == "report_outcome":
        result = _report_outcome(
            session_id=arguments.get("session_id", ""),
            sent_back=bool(arguments.get("sent_back", False)),
            already_existed=bool(arguments.get("already_existed", False)),
            is_rewrite=bool(arguments.get("is_rewrite", False)),
            turn_count=int(arguments.get("turn_count", 1)),
            hermes_notes=arguments.get("hermes_notes", ""),
            checklist_passed=arguments.get("checklist_passed"),
        )
    elif name == "run_local_task":
        from learngentic.local_tasks.runner import run_local_task
        result = run_local_task(
            task_description=arguments.get("task_description", ""),
            user_input=arguments.get("user_input", ""),
            attempt_number=int(arguments.get("attempt_number", 1)),
            system_prompt_override=arguments.get("system_prompt_override"),
            previous_run_id=arguments.get("previous_run_id"),
            session_id=arguments.get("session_id"),
        )
    elif name == "report_local_result":
        from learngentic.local_tasks.runner import report_local_result
        result = report_local_result(
            run_id=arguments.get("run_id", ""),
            verdict=arguments.get("verdict", ""),
        )
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def _risk_assessment(prompt: str, cwd: str) -> dict:
    """
    Classify the prompt, look up historical patterns for that type,
    and return risk level + grounded clarifying questions.
    """
    with get_conn() as conn:
        # Classify the incoming prompt
        try:
            cls = classify_session(initial_prompt=prompt, files_changed=[])
            change_type = cls.change_type
            code_region = cls.code_region_type
        except Exception:
            change_type = "other"
            code_region = "unknown"

        # Look up global patterns for this type
        pattern = conn.execute("""
            SELECT avg_outcome_quality, avg_efficiency, avg_durability,
                   common_failure_modes, sample_count
            FROM global_patterns
            WHERE change_type = ? AND code_region_type = ?
        """, (change_type, code_region)).fetchone()

        # Also look up project-specific history
        project_id = _project_id(cwd)
        recent = conn.execute("""
            SELECT s.initial_prompt, s.turn_count, s.outcome_quality,
                   s.execution_efficiency, s.durability, s.failure_mode_tags,
                   s.change_type, s.code_region_type
            FROM sessions s
            WHERE s.change_type = ? AND s.code_region_type = ?
              AND (? = '' OR s.project_id = ?)
            ORDER BY s.started_at DESC
            LIMIT 5
        """, (change_type, code_region, project_id, project_id)).fetchall()

        # Compute risk level
        risk = "low"
        risk_reasons: list[str] = []
        questions: list[str] = []

        if pattern:
            avg_eff = pattern["avg_efficiency"] or 0.3
            avg_dur = pattern["avg_durability"] or 0.65
            n = pattern["sample_count"] or 0

            if avg_eff < 0.18:
                risk = "high"
                risk_reasons.append(
                    f"Tasks like this historically take many turns "
                    f"(avg efficiency {avg_eff:.2f} across {n} sessions)"
                )
            elif avg_eff < 0.25:
                risk = "medium"

            if avg_dur < 0.5:
                risk = "high"
                risk_reasons.append(
                    f"Changes of this type have low durability "
                    f"(avg {avg_dur:.2f}) — they tend to get reworked"
                )

            # Surface common failure modes as questions
            if pattern["common_failure_modes"]:
                try:
                    modes = json.loads(pattern["common_failure_modes"])
                    for mode, count in sorted(modes.items(), key=lambda x: -x[1]):
                        if count >= 2:
                            questions.append(_mode_to_question(mode, change_type, code_region))
                except (json.JSONDecodeError, AttributeError):
                    pass

        # Add questions from recent low-scoring sessions
        for row in recent:
            tags = []
            if row["failure_mode_tags"]:
                try:
                    tags = json.loads(row["failure_mode_tags"])
                except json.JSONDecodeError:
                    pass
            for tag in tags:
                q = _mode_to_question(tag, change_type, code_region)
                if q not in questions:
                    questions.append(q)

        # Summarise recent sessions
        recent_summary = []
        for row in recent:
            oq = row["outcome_quality"]
            turns = row["turn_count"] or "?"
            p = (row["initial_prompt"] or "")[:60].replace("\n", " ")
            recent_summary.append({
                "prompt_preview": p,
                "turns": turns,
                "outcome": round(oq, 2) if oq is not None else None,
            })

        return {
            "change_type": change_type,
            "code_region_type": code_region,
            "risk_level": risk,
            "risk_reasons": risk_reasons,
            "clarifying_questions": questions[:4],   # cap at 4
            "recent_similar_sessions": recent_summary,
        }


def _outcome_history(description: str, cwd: str, k: int) -> dict:
    """Return the k most relevant past sessions using semantic + classification matching."""
    change_type, code_region = _keyword_classify(description)

    project_id = _project_id(cwd)

    # Semantic search — find similar sessions by prompt text
    similar = search_similar(description, k=k * 2)
    semantic_ids = [r["session_id"] for r in similar]

    with get_conn() as conn:
        # Fetch semantically similar sessions first, then fall back to classification match
        if semantic_ids:
            placeholders = ",".join("?" * len(semantic_ids))
            semantic_rows = conn.execute(f"""
                SELECT session_id, initial_prompt, turn_count, outcome_quality,
                       execution_efficiency, durability, prompt_quality,
                       change_type, code_region_type, started_at
                FROM sessions
                WHERE session_id IN ({placeholders})
            """, semantic_ids).fetchall()
            # Reorder by semantic similarity score
            sem_map = {r["session_id"]: dict(r) for r in semantic_rows}
            ordered = [sem_map[sid] for sid in semantic_ids if sid in sem_map]
        else:
            ordered = []

        # Fill remaining slots with classification-based matches not already included
        found_ids = {r["session_id"] for r in ordered}
        if len(ordered) < k:
            class_rows = conn.execute("""
                SELECT session_id, initial_prompt, turn_count, outcome_quality,
                       execution_efficiency, durability, prompt_quality,
                       change_type, code_region_type, started_at
                FROM sessions
                WHERE (change_type = ? OR code_region_type = ?)
                ORDER BY
                    CASE WHEN project_id = ? THEN 0 ELSE 1 END,
                    started_at DESC
                LIMIT ?
            """, (change_type, code_region, project_id, k * 2)).fetchall()
            for r in class_rows:
                if r["session_id"] not in found_ids:
                    ordered.append(dict(r))
                    found_ids.add(r["session_id"])

        results = []
        for row in ordered[:k]:
            oq = row["outcome_quality"]
            eff = row["execution_efficiency"]
            dur = row["durability"]
            results.append({
                "date": (row["started_at"] or "")[:10],
                "prompt_preview": (row["initial_prompt"] or "")[:80].replace("\n", " "),
                "change_type": row["change_type"],
                "code_region_type": row["code_region_type"],
                "turns": row["turn_count"],
                "outcome_quality": round(oq, 2) if oq is not None else None,
                "efficiency": round(eff, 2) if eff is not None else None,
                "durability": round(dur, 2) if dur is not None else None,
            })

        # Global averages for context
        agg = conn.execute("""
            SELECT ROUND(AVG(outcome_quality),2) as avg_oq,
                   ROUND(AVG(execution_efficiency),2) as avg_eff,
                   ROUND(AVG(durability),2) as avg_dur,
                   COUNT(*) as n
            FROM sessions
            WHERE change_type = ? AND code_region_type = ?
        """, (change_type, code_region)).fetchone()

        return {
            "classified_as": {"change_type": change_type, "code_region_type": code_region},
            "global_averages": {
                "outcome_quality": agg["avg_oq"],
                "efficiency": agg["avg_eff"],
                "durability": agg["avg_dur"],
                "sample_count": agg["n"],
            },
            "similar_sessions": results,
        }


def _file_stability(file_path: str) -> dict:
    """Return stability stats for a file path (matches on suffix)."""
    # Normalise to repo-relative style
    fp = Path(file_path).as_posix()
    # Try exact match first, then suffix match
    with get_conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                    AS total_changes,
                AVG(days_until_next_touch)  AS avg_days_stable,
                SUM(was_reverted)           AS revert_count,
                AVG(churn_count_30d)        AS avg_churn_30d,
                MAX(churn_count_30d)        AS max_churn_30d
            FROM change_events
            WHERE repo_relative_path = ?
               OR repo_relative_path LIKE ?
        """, (fp, f"%/{Path(file_path).name}")).fetchone()

        if not row or row["total_changes"] == 0:
            return {"file_path": file_path, "found": False, "message": "No history for this file."}

        avg_stable = row["avg_days_stable"]
        avg_churn = row["avg_churn_30d"] or 0
        risk = "low"
        if avg_churn > 3 or (row["revert_count"] or 0) > 0:
            risk = "high"
        elif avg_churn > 1.5:
            risk = "medium"

        return {
            "file_path": file_path,
            "found": True,
            "total_changes": row["total_changes"],
            "avg_days_until_next_touch": round(avg_stable, 1) if avg_stable is not None else None,
            "revert_count": row["revert_count"] or 0,
            "avg_churn_30d": round(avg_churn, 1),
            "max_churn_30d": row["max_churn_30d"] or 0,
            "stability_risk": risk,
        }


# ---------------------------------------------------------------------------
# Implementation: record_task
# ---------------------------------------------------------------------------

def _record_task(
    prompt: str,
    cwd: str,
    files_mentioned: list[str],
    agent_type: str,
) -> dict:
    """
    Start a Hermes task: create a session row, classify the prompt, check for
    recent file conflicts, and return a TaskStandard + session_id.
    """
    _ensure_api_key()

    # Classify prompt → (change_type, code_region_type)
    try:
        cls = classify_session(initial_prompt=prompt, files_changed=files_mentioned or [])
        change_type = cls.change_type
        code_region = cls.code_region_type
    except Exception:
        change_type = "other"
        code_region = "unknown"

    session_id = str(uuid.uuid4())
    project_id = _project_id(cwd)
    started_at = datetime.now(timezone.utc).isoformat()

    with get_conn() as conn:
        conn.execute("""
            INSERT INTO sessions (
                session_id, project_id, cwd, started_at,
                initial_prompt, agent_type, change_type, code_region_type,
                assessment_source, files_mentioned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (session_id, project_id, cwd, started_at, prompt,
              agent_type or "hermes", change_type, code_region,
              json.dumps(files_mentioned) if files_mentioned else None))

        try:
            index_session(session_id, prompt)
        except Exception:
            pass

        # Layer-2 memory: flag files touched in the last 24 hours
        file_warnings: list[dict] = []
        if files_mentioned:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            for fp in files_mentioned:
                fp_posix = Path(fp).as_posix()
                row = conn.execute("""
                    SELECT COUNT(*) AS n
                    FROM change_events ce
                    JOIN sessions s ON s.session_id = ce.session_id
                    WHERE (ce.repo_relative_path = ? OR ce.repo_relative_path LIKE ?)
                      AND ce.event_timestamp > ?
                      AND (? = '' OR s.project_id = ?)
                """, (fp_posix, f"%/{Path(fp).name}", cutoff, project_id, project_id)).fetchone()
                if row and (row.get("n") or 0) > 0:
                    file_warnings.append({
                        "file": fp,
                        "warning": (
                            f"Modified {row['n']} time(s) in the last 24 hours. "
                            "Risk of conflict or duplicate work."
                        ),
                    })

    # Seed session call log — capture any pre-task guidance calls made in the
    # last 5 minutes (e.g. query_risk_assessment, query_outcome_history)
    # plus the record_task call itself.
    pre_task_cutoff = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    with _call_log_lock:
        pre_calls = [
            c for c in _recent_tool_calls
            if c["ts"] >= pre_task_cutoff
            and c["tool"] in _GUIDANCE_TOOLS
            and c["tool"] != "record_task"
        ]
        _session_call_log[session_id] = pre_calls + [{"tool": "record_task", "ts": started_at}]

    # Write session pointer so the PostToolUse hook can associate events
    _write_current_session(session_id, cwd, files_mentioned or [])

    # Generate TaskStandard (reads DB separately)
    from learngentic.standards.generator import generate_standard
    standard = generate_standard(change_type, code_region)

    # Fetch recent prompt_quality scores so Claude can see its own track record
    recent_pq: list[float] = []
    try:
        with get_conn() as conn:
            rows = conn.execute("""
                SELECT prompt_quality FROM sessions
                WHERE change_type = ? AND code_region_type = ?
                  AND prompt_quality IS NOT NULL
                ORDER BY started_at DESC
                LIMIT 5
            """, (change_type, code_region)).fetchall()
        recent_pq = [round(r["prompt_quality"], 3) for r in rows]
    except Exception:
        pass

    return {
        "session_id": session_id,
        "classified_as": {"change_type": change_type, "code_region_type": code_region},
        "standard": standard.to_dict(),
        "file_recency_warnings": file_warnings,
        "prompt_quality_history": recent_pq,
    }


# ---------------------------------------------------------------------------
# Implementation: report_outcome
# ---------------------------------------------------------------------------

def _report_outcome(
    session_id: str,
    sent_back: bool,
    already_existed: bool,
    is_rewrite: bool,
    turn_count: int,
    hermes_notes: str,
    checklist_passed: list[int] | None = None,
) -> dict:
    """
    Close a Hermes task: compute self-assessment, persist it, trigger async
    prompt grading, and return a pass/fail verdict against the stored standard.
    """
    if not session_id:
        return {"error": "session_id is required."}

    with get_conn() as conn:
        session = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

    if not session:
        return {"error": f"Session '{session_id}' not found."}

    change_type = session.get("change_type") or "other"
    code_region  = session.get("code_region_type") or "unknown"
    prompt       = session.get("initial_prompt") or ""

    # ── Score computation ──────────────────────────────────────────────────
    score = 0.7   # neutral base
    breakdown: list[str] = []

    if sent_back:
        score -= 0.40
        breakdown.append("sent_back=True: user rejected/revised result  −0.40")
    if already_existed:
        score -= 0.20
        breakdown.append("already_existed=True: built code already present  −0.20")
    if is_rewrite:
        score -= 0.15
        breakdown.append("is_rewrite=True: rewrites recent work, durability concern  −0.15")

    # Efficiency bonus when turn count is within the historical threshold
    from learngentic.standards.generator import generate_standard
    standard = generate_standard(change_type, code_region)
    turn_eff  = efficiency_score(turn_count)
    threshold = standard.benchmarks.get("efficiency_threshold", 0.2)

    if not sent_back and turn_eff >= threshold:
        bonus = round(min(0.15, turn_eff * 0.2), 3)
        score += bonus
        breakdown.append(
            f"efficiency_bonus: turn_count={turn_count}, eff={turn_eff:.3f} ≥ threshold={threshold:.3f}  +{bonus}"
        )

    # ── Objective signals from hook-observed tool events ──────────────────
    session_events = _sync_tool_event_buffer(session_id)
    files_mentioned_raw = session.get("files_mentioned") or "[]"
    try:
        files_mentioned_list = json.loads(files_mentioned_raw) if isinstance(files_mentioned_raw, str) else files_mentioned_raw
    except Exception:
        files_mentioned_list = []

    obj = _compute_objective_signals(session_events, files_mentioned_list, standard, turn_count)

    if obj["overrun_penalty"] > 0:
        score -= obj["overrun_penalty"]
        breakdown.append(
            f"overrun_penalty: objective_turns={obj['objective_turn_count']} "
            f"> 2× expected_high={standard.benchmarks.get('expected_turns_high')}  "
            f"−{obj['overrun_penalty']}"
        )

    hermes_assessment = round(max(0.0, min(1.0, score)), 3)

    # ── Checklist compliance (Feature 3) ───────────────────────────────────
    checklist_compliance: float | None = None
    if checklist_passed is not None:
        total_items = len(standard.completion_checklist)
        valid = [i for i in checklist_passed if 0 <= i < total_items]
        checklist_compliance = round(len(valid) / max(total_items, 1), 3)
        breakdown.append(
            f"checklist_compliance: {len(valid)}/{total_items} items self-reported satisfied"
        )

    # ── Guidance adherence (Feature 5) ────────────────────────────────────
    report_ts = datetime.now(timezone.utc).isoformat()
    with _call_log_lock:
        session_log = list(_session_call_log.get(session_id, []))
        session_log.append({"tool": "report_outcome", "ts": report_ts})
        _session_call_log.pop(session_id, None)

    tools_used = {entry["tool"] for entry in session_log} & _GUIDANCE_TOOLS
    adherence_rate = round(len(tools_used) / len(_GUIDANCE_TOOLS), 3)

    # ── Persist ────────────────────────────────────────────────────────────
    turn_eff_final = efficiency_score(turn_count)
    with get_conn() as conn:
        conn.execute("""
            UPDATE sessions SET
                hermes_assessment     = ?,
                assessment_source     = 'hermes_self',
                turn_count            = ?,
                hermes_notes          = ?,
                execution_efficiency  = ?,
                tool_calls            = ?,
                checklist_compliance  = ?,
                objective_turn_count  = ?,
                scope_expansion_ratio = ?
            WHERE session_id = ?
        """, (
            hermes_assessment, turn_count, hermes_notes or None, turn_eff_final,
            json.dumps(session_log), checklist_compliance,
            obj["objective_turn_count"] or None,
            obj["scope_expansion_ratio"],
            session_id,
        ))

    # ── Async git ingest + prompt grading ─────────────────────────────────
    cwd        = session.get("cwd") or ""
    started_at = session.get("started_at") or ""
    _fire_git_ingest_async(session_id, cwd, started_at)
    _fire_prompt_grader_async(session_id, prompt, turn_count, hermes_assessment, cwd, started_at)
    _fire_pattern_refresh_async()

    # ── Verdict against standard ───────────────────────────────────────────
    # Map each negative signal to the checklist item it violated.
    checklist_failed: list[str] = []
    if sent_back:
        checklist_failed.append(standard.completion_checklist[-1])
    if already_existed:
        checklist_failed.append(
            "Confirm no in-progress changes exist for the same files (check git status)."
        )
    if is_rewrite:
        checklist_failed.append(
            "Clarify scope: which files/functions are in scope and which should not be touched."
        )

    verdict = "pass" if hermes_assessment >= 0.5 and not sent_back else "fail"

    next_steps: list[str] = []
    if sent_back:
        next_steps.append(
            "Review the rejection reason and restart with updated acceptance criteria."
        )
    if already_existed:
        next_steps.append(
            "Check for existing implementations before starting similar tasks."
        )
    if is_rewrite:
        next_steps.append(
            "Investigate why a recent change needed rewriting — consider adding a scope-check to the standard checklist."
        )
    high = standard.benchmarks.get("expected_turns_high", 30)
    if turn_count > high:
        next_steps.append(
            f"Turn count ({turn_count}) exceeded expected high ({high}). "
            "Consider breaking into smaller tasks."
        )

    # Flag significant divergences for visibility
    obj_warnings: list[str] = []
    if obj["turn_divergence"] is not None and obj["turn_divergence"] > 0.5:
        obj_warnings.append(
            f"turn_count divergence: self-reported {turn_count} vs "
            f"observed {obj['objective_turn_count']} "
            f"(delta {obj['turn_divergence']:.0%})"
        )
    if obj["scope_expansion_ratio"] is not None and obj["scope_expansion_ratio"] > 0.5:
        obj_warnings.append(
            f"scope expansion: {obj['scope_expansion_ratio']:.0%} of edits "
            f"outside declared files_mentioned"
        )

    return {
        "session_id": session_id,
        "verdict": verdict,
        "hermes_assessment": hermes_assessment,
        "score_breakdown": breakdown,
        "standard_confidence": standard.confidence,
        "checklist_items_failed": checklist_failed,
        "checklist_compliance": checklist_compliance,
        "guidance_adherence": adherence_rate,
        "tools_used_this_session": sorted(tools_used),
        "objective_signals": {
            "observed_tool_calls": obj["objective_turn_count"],
            "self_reported_turns": turn_count,
            "turn_divergence": obj["turn_divergence"],
            "scope_expansion_ratio": obj["scope_expansion_ratio"],
            "edited_files": obj["edited_files"],
        },
        "objective_warnings": obj_warnings,
        "next_steps": next_steps,
        "note": (
            "hermes_assessment is stored separately from outcome_quality. "
            "Git signals arriving after a push may override this via 'git_grounded'."
        ),
    }


_SESSION_FILE = Path.home() / ".learngentic" / "current_session.json"
_BUFFER_FILE  = Path.home() / ".learngentic" / "tool_event_buffer.jsonl"


def _write_current_session(session_id: str, cwd: str, files_mentioned: list[str]) -> None:
    try:
        _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SESSION_FILE.write_text(json.dumps({
            "session_id": session_id,
            "cwd": cwd,
            "files_mentioned": files_mentioned,
        }))
    except Exception:
        pass


def _sync_tool_event_buffer(session_id: str) -> list[dict]:
    """Read the local JSONL buffer, batch-insert all events to Turso, return session's events."""
    if not _BUFFER_FILE.exists():
        return []
    try:
        lines = _BUFFER_FILE.read_text().splitlines()
    except Exception:
        return []

    events: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not events:
        return []

    try:
        from learngentic.store.db import insert_tool_events_batch
        with get_conn() as conn:
            insert_tool_events_batch(conn, events)
        _BUFFER_FILE.write_text("")  # Clear the buffer file after successful sync
    except Exception:
        _log.warning("tool event buffer sync failed", exc_info=True)

    return [e for e in events if e.get("session_id") == session_id]


def _compute_objective_signals(
    session_events: list[dict],
    files_mentioned: list[str],
    standard,
    turn_count: int,
) -> dict:
    """Derive objective signals from hook-observed tool events."""
    from pathlib import Path as _Path

    objective_turn_count = len(session_events)

    # Files actually edited (Edit / Write tools only)
    edit_tools = {"Edit", "Write", "NotebookEdit"}
    edited_paths = {
        e["file_path"] for e in session_events
        if e.get("tool_name") in edit_tools and e.get("file_path")
    }

    # Scope expansion: fraction of edited files not in files_mentioned
    scope_expansion_ratio: float | None = None
    if files_mentioned and edited_paths:
        mentioned_basenames = {_Path(f).name for f in files_mentioned}
        edited_basenames   = {_Path(f).name for f in edited_paths}
        in_scope = edited_basenames & mentioned_basenames
        scope_expansion_ratio = round(
            1.0 - len(in_scope) / max(len(edited_basenames), 1), 3
        )

    # Turn count divergence: how much did Claude under/over-report?
    turn_divergence: float | None = None
    if objective_turn_count > 0:
        turn_divergence = round(
            abs(turn_count - objective_turn_count) / objective_turn_count, 3
        )

    # Penalty: objective turns greatly exceed the historical high
    expected_high = standard.benchmarks.get("expected_turns_high", 9999)
    overrun_penalty = 0.0
    if objective_turn_count > expected_high * 2:
        overrun_penalty = 0.10

    return {
        "objective_turn_count": objective_turn_count,
        "edited_files": sorted(edited_paths),
        "scope_expansion_ratio": scope_expansion_ratio,
        "turn_divergence": turn_divergence,
        "overrun_penalty": overrun_penalty,
    }


def _fire_git_ingest_async(session_id: str, cwd: str, started_at: str) -> None:
    """Parse git commits made after the session started and create change_events rows."""
    def _run() -> None:
        try:
            import hashlib
            from datetime import datetime, timezone, timedelta
            from learngentic.pipeline.git_parser import find_repo_root, parse_repo
            from learngentic.store.db import get_conn, insert_git_change_event

            if not cwd:
                return

            root = find_repo_root(cwd)
            if not root:
                return

            commits = parse_repo(root)
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))

            # Partition commits: session commits (after started) vs all others
            session_commits = [c for c in commits if c.timestamp > started]
            if not session_commits:
                return

            with get_conn() as conn:
                for commit in session_commits:
                    for file_path in commit.files_changed:
                        event_id = hashlib.sha1(
                            f"{session_id}:{commit.commit_hash}:{file_path}".encode()
                        ).hexdigest()[:24]
                        insert_git_change_event(
                            conn,
                            event_id=event_id,
                            session_id=session_id,
                            repo_relative_path=file_path,
                            event_timestamp=commit.timestamp.isoformat(),
                        )

            # ── Durability computation (Feature 4) ────────────────────────
            # Durability = fraction of session files NOT re-touched within 7 days
            session_files = {f for c in session_commits for f in c.files_changed}
            if session_files:
                session_end = max(c.timestamp for c in session_commits)
                window_end = session_end + timedelta(days=7)
                re_touched: set[str] = set()
                for commit in commits:
                    if session_end < commit.timestamp <= window_end:
                        for f in commit.files_changed:
                            if f in session_files:
                                re_touched.add(f)
                durability = round(1.0 - len(re_touched) / len(session_files), 3)
                with get_conn() as conn:
                    # Only write if no durability score already set (don't override human scores)
                    conn.execute(
                        "UPDATE sessions SET durability = ? WHERE session_id = ? AND durability IS NULL",
                        (durability, session_id),
                    )
        except Exception:
            _log.warning("git ingest background thread failed", exc_info=True)

    threading.Thread(target=_run, daemon=True).start()


def _read_claude_jsonl_reprompts(cwd: str, started_at: str) -> list[str]:
    """Find the Claude Code JSONL for this session and extract mid-session reprompts."""
    try:
        slug = "-" + cwd.lstrip("/").replace("/", "-")
        jsonl_dir = Path.home() / ".claude" / "projects" / slug
        if not jsonl_dir.is_dir():
            return []

        try:
            cutoff = datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp()
        except Exception:
            cutoff = 0.0

        candidates = [f for f in jsonl_dir.glob("*.jsonl") if f.stat().st_mtime >= cutoff]
        if not candidates:
            return []

        jsonl_file = max(candidates, key=lambda f: f.stat().st_mtime)
        entries: list[dict] = []
        for line in jsonl_file.read_text(errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        from learngentic.scoring.prompt_grader import extract_reprompts
        return extract_reprompts(entries)
    except Exception:
        return []


def _fire_prompt_grader_async(
    session_id: str,
    prompt: str,
    turn_count: int,
    hermes_assessment: float,
    cwd: str = "",
    started_at: str = "",
) -> None:
    """Run the prompt grader in a daemon thread — does not block the MCP response."""
    def _run() -> None:
        try:
            from learngentic.scoring.prompt_grader import grade_session
            reprompts = _read_claude_jsonl_reprompts(cwd, started_at)
            result = grade_session(
                initial_prompt=prompt,
                reprompts=reprompts,
                outcome_rating=hermes_assessment,
                turn_count=turn_count,
            )
            with get_conn() as conn:
                conn.execute("""
                    UPDATE sessions SET
                        prompt_quality    = ?,
                        failure_mode_tags = ?
                    WHERE session_id = ?
                """, (result.prompt_quality, json.dumps(result.failure_modes), session_id))
        except Exception:
            _log.warning("prompt grader background thread failed", exc_info=True)

    threading.Thread(target=_run, daemon=True).start()


def _fire_pattern_refresh_async() -> None:
    def _run() -> None:
        try:
            from learngentic.store.db import refresh_global_patterns
            with get_conn() as conn:
                refresh_global_patterns(conn)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keyword_classify(text: str) -> tuple[str, str]:
    """
    Offline keyword-based classifier for MCP tools that don't warrant an LLM call.
    Returns (change_type, code_region_type). Accuracy is lower than the LLM classifier
    but costs nothing and is fast enough for similarity lookups.
    """
    t = text.lower()
    if any(w in t for w in ("fix", "bug", "error", "crash", "broken", "issue", "revert")):
        change_type = "bug_fix"
    elif any(w in t for w in ("refactor", "clean", "rename", "reorgani", "restructur")):
        change_type = "refactor"
    elif any(w in t for w in ("test", "spec", "assert", "coverage")):
        change_type = "test_addition"
    elif any(w in t for w in ("doc", "readme", "comment", "docstring")):
        change_type = "documentation"
    elif any(w in t for w in ("add", "new", "creat", "implement", "build", "introduc")):
        change_type = "feature_addition"
    else:
        change_type = "other"

    if any(w in t for w in ("auth", "login", "password", "token", "oauth", "permission")):
        code_region = "auth"
    elif any(w in t for w in ("api", "endpoint", "route", "request", "response", "http")):
        code_region = "api"
    elif any(w in t for w in ("ui", "component", "button", "form", "page", "frontend", "style")):
        code_region = "ui"
    elif any(w in t for w in ("db", "database", "model", "schema", "migration", "query", "sql")):
        code_region = "data_layer"
    elif any(w in t for w in ("config", "setting", "env", "deploy", "ci", "pipeline")):
        code_region = "infrastructure"
    elif any(w in t for w in ("test", "spec", "coverage")):
        code_region = "testing"
    elif any(w in t for w in ("tool", "script", "make", "hook")):
        code_region = "tooling"
    else:
        code_region = "unknown"

    return change_type, code_region


def _project_id(cwd: str) -> str:
    import hashlib
    return hashlib.sha1(cwd.encode()).hexdigest()[:12] if cwd else ""


def _mode_to_question(mode: str, change_type: str, region: str) -> str:
    questions = {
        "missing_acceptance_criteria": "What does success look like? Define the acceptance criteria before we start.",
        "underspecified_scope": "What should Claude change, and what should it leave alone?",
        "missing_constraints": "Are there constraints to preserve — error handling, performance, existing API contracts?",
        "context_gap": "Is there codebase context Claude will need that isn't obvious from the file names?",
        "over_specified": "Are you over-constraining the approach? Sometimes specifying the 'what' not the 'how' gets better results.",
    }
    return questions.get(mode, f"Consider clarifying: {mode}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    _ensure_api_key()
    asyncio.run(main())
