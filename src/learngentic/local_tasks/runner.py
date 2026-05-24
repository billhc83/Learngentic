from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI


# ---------------------------------------------------------------------------
# Task taxonomy
# ---------------------------------------------------------------------------

TASK_TYPES = {"structured_output", "code_task", "analytical", "summarization"}

_DEFAULT_TASK_MODELS = {
    "structured_output": "qwen3-14b-nothink",
    "code_task":         "qwen2.5-coder:14b",
    "analytical":        "qwen3:14b",
    "summarization":     "qwen3-14b-nothink",
}

_DEFAULT_SYSTEM_PROMPTS = {
    "structured_output": (
        "You are a precise data formatter. Given input content and a task description, "
        "produce only the requested structured output (JSON, list, table, or other format). "
        "Do not include explanations. Output only the result."
    ),
    "code_task": (
        "You are an expert software developer. Given the input, produce concise, correct "
        "code or code-related output as described. Output only the result with no explanation."
    ),
    "analytical": (
        "You are a thoughtful analyst. Given the input, reason through the task carefully "
        "and provide your analysis. Be thorough but concise."
    ),
    "summarization": (
        "You are a precise summarizer. Given the input, produce a clear, accurate summary "
        "as described. Output only the summary."
    ),
}

_TASK_CLASSIFIER_SYSTEM = """\
You are a task router for a local LLM system.
Given a task description, classify it into exactly one category:

- structured_output: producing JSON, lists, tables, or other fixed-format data
- code_task: writing code, commit messages, code review, code analysis
- analytical: reasoning, comparison, root cause analysis, planning
- summarization: condensing or extracting key points from text

Respond ONLY with valid JSON: {"task_type": "<category>"}
"""


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path.home() / ".learngentic" / "config.json"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _ollama_client(cfg: dict) -> OpenAI:
    base_url = cfg.get("ollama_base_url", "http://localhost:11434/v1")
    return OpenAI(base_url=base_url, api_key="ollama")


def _classifier_model(cfg: dict) -> str:
    return cfg.get("ollama_model", "hermes3")


def _task_model(task_type: str, cfg: dict) -> str:
    task_models = cfg.get("task_models", {})
    return task_models.get(task_type, _DEFAULT_TASK_MODELS.get(task_type, cfg.get("ollama_model", "hermes3")))


# ---------------------------------------------------------------------------
# Task classification
# ---------------------------------------------------------------------------

def _classify_task(task_description: str, client: OpenAI, model: str) -> str:
    """Classify task_description into one of TASK_TYPES using a local model."""
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=32,
            messages=[
                {"role": "system", "content": _TASK_CLASSIFIER_SYSTEM},
                {"role": "user", "content": task_description},
            ],
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            t = data.get("task_type", "")
            if t in TASK_TYPES:
                return t
    except Exception:
        pass

    # Keyword fallback when the model call fails
    t = task_description.lower()
    if any(w in t for w in ("json", "list", "table", "format", "schema", "structured")):
        return "structured_output"
    if any(w in t for w in ("code", "commit", "function", "class", "implement", "review", "diff")):
        return "code_task"
    if any(w in t for w in ("summarize", "summary", "condense", "extract", "key points")):
        return "summarization"
    return "analytical"


# ---------------------------------------------------------------------------
# local_capable check
# ---------------------------------------------------------------------------

def _check_local_capable(task_type: str, cfg: dict) -> bool:
    """Return True if historical pass rate for task_type is above threshold."""
    threshold = float(cfg.get("local_capable_threshold", 0.4))
    min_samples = int(cfg.get("local_capable_min_samples", 5))
    try:
        from learngentic.store.db import get_conn
        with get_conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN verdict = 'pass' THEN 1 ELSE 0 END) AS passes
                FROM local_model_runs
                WHERE task_type = ? AND verdict IS NOT NULL
            """, (task_type,)).fetchone()
        if not row:
            return True
        total = row.get("total") or 0
        passes = row.get("passes") or 0
        if total < min_samples:
            return True
        return (passes / total) >= threshold
    except Exception:
        return True  # assume capable if DB unavailable


# ---------------------------------------------------------------------------
# DB logging
# ---------------------------------------------------------------------------

def _log_run(
    run_id: str,
    task_type: str,
    task_description: str,
    system_prompt: str,
    user_input: str,
    model_output: str | None,
    model_used: str,
    attempt_number: int,
    previous_run_id: str | None,
    session_id: str | None,
    local_capable: bool,
) -> None:
    try:
        from learngentic.store.db import get_conn
        with get_conn() as conn:
            conn.execute("""
                INSERT INTO local_model_runs (
                    run_id, task_type, task_description, system_prompt,
                    user_input, model_output, model_used, attempt_number,
                    previous_run_id, session_id, local_capable, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, task_type, task_description, system_prompt,
                user_input, model_output, model_used, attempt_number,
                previous_run_id, session_id, int(local_capable),
                datetime.now(timezone.utc).isoformat(),
            ))
            if previous_run_id:
                conn.execute("""
                    UPDATE local_model_runs SET verdict = 'fail'
                    WHERE run_id = ? AND verdict IS NULL
                """, (previous_run_id,))
    except Exception:
        pass  # logging failure must never crash the tool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_local_task(
    task_description: str,
    user_input: str,
    attempt_number: int = 1,
    system_prompt_override: str | None = None,
    previous_run_id: str | None = None,
    session_id: str | None = None,
) -> dict:
    cfg = _load_config()
    client = _ollama_client(cfg)
    run_id = str(uuid.uuid4())

    task_type = _classify_task(task_description, client, _classifier_model(cfg))
    capable = _check_local_capable(task_type, cfg)

    if not capable:
        _log_run(
            run_id=run_id, task_type=task_type, task_description=task_description,
            system_prompt="", user_input=user_input, model_output=None,
            model_used="", attempt_number=attempt_number,
            previous_run_id=previous_run_id, session_id=session_id, local_capable=False,
        )
        return {
            "run_id": run_id,
            "task_type": task_type,
            "local_capable": False,
            "output": None,
            "model_used": None,
            "attempt_number": attempt_number,
            "message": (
                f"Local model pass rate for '{task_type}' is below threshold. "
                "Handle this task yourself."
            ),
        }

    system_prompt = system_prompt_override or _DEFAULT_SYSTEM_PROMPTS.get(task_type, "")
    model = _task_model(task_type, cfg)

    output = None
    error = None
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"TASK: {task_description}\n\nINPUT:\n{user_input}"},
            ],
        )
        output = response.choices[0].message.content.strip()
    except Exception as exc:
        output = None
        error = str(exc)

    _log_run(
        run_id=run_id, task_type=task_type, task_description=task_description,
        system_prompt=system_prompt, user_input=user_input, model_output=output,
        model_used=model, attempt_number=attempt_number,
        previous_run_id=previous_run_id, session_id=session_id, local_capable=True,
    )

    result = {
        "run_id": run_id,
        "task_type": task_type,
        "local_capable": True,
        "output": output,
        "model_used": model,
        "attempt_number": attempt_number,
    }

    if error is not None:
        result["error"] = error

    return result


def report_local_result(run_id: str, verdict: str) -> dict:
    if verdict not in ("pass", "fail", "gave_up"):
        return {"error": f"Invalid verdict '{verdict}'. Must be pass, fail, or gave_up."}
    try:
        from learngentic.store.db import get_conn
        with get_conn() as conn:
            conn.execute(
                "UPDATE local_model_runs SET verdict = ? WHERE run_id = ?",
                (verdict, run_id),
            )
        return {"ok": True, "run_id": run_id, "verdict": verdict}
    except Exception as exc:
        return {"error": str(exc)}
