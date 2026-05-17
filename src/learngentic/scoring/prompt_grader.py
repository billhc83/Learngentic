from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


def _ollama_client() -> OpenAI:
    config_path = Path.home() / ".learngentic" / "config.json"
    base_url = "http://localhost:11434/v1"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            base_url = cfg.get("ollama_base_url", base_url)
        except (json.JSONDecodeError, OSError):
            pass
    return OpenAI(base_url=base_url, api_key="ollama")


def _ollama_model() -> str:
    config_path = Path.home() / ".learngentic" / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            return cfg.get("ollama_model", "hermes3")
        except (json.JSONDecodeError, OSError):
            pass
    return "hermes3"


VALID_FAILURE_MODES = {
    "missing_acceptance_criteria",
    "underspecified_scope",
    "missing_constraints",
    "context_gap",
    "over_specified",
}

GRADER_SYSTEM = """\
You are a prompt quality analyst for an AI coding assistant system.
Your job is to evaluate how well an initial task prompt specified what was needed,
given what the session actually produced and how many turns it took.

You have access to:
- The initial prompt (what the user asked at the start)
- The reprompt sequence (clarifications the user had to add mid-session)
- The outcome rating (0-1, where 1 = fully what was expected)
- The turn count (how many assistant turns it took)

Your task:
1. Score the initial prompt quality from 0.0 to 1.0
   - 1.0 = the initial prompt fully specified what was needed; minimal reprompting required
   - 0.0 = the initial prompt was so vague or wrong it caused session failure
2. Identify which failure modes apply (list only those that clearly apply):
   - missing_acceptance_criteria: success conditions were not defined
   - underspecified_scope: what to change/not change was unclear
   - missing_constraints: error handling, performance, style requirements omitted
   - context_gap: Claude lacked codebase knowledge needed to proceed correctly
   - over_specified: prompt was too prescriptive, constrained Claude incorrectly

Respond ONLY with valid JSON in this exact format:
{
  "prompt_quality": <float 0.0-1.0>,
  "failure_modes": [<zero or more of the valid mode strings>],
  "reasoning": "<one sentence explaining the score>"
}
"""


@dataclass
class GraderResult:
    prompt_quality: float
    failure_modes: list[str]
    reasoning: str


def _build_user_message(
    initial_prompt: str,
    reprompts: list[str],
    outcome_rating: float,
    turn_count: int,
) -> str:
    parts = [
        f"INITIAL PROMPT:\n{initial_prompt or '(empty)'}",
        f"\nOUTCOME RATING: {outcome_rating:.2f} / 1.0",
        f"TURN COUNT: {turn_count}",
    ]
    if reprompts:
        parts.append("\nREPROMPT SEQUENCE (clarifications added mid-session):")
        for i, rp in enumerate(reprompts, 1):
            parts.append(f"  [{i}] {rp[:300]}")
    else:
        parts.append("\nREPROMPT SEQUENCE: (none — single-turn session)")
    return "\n".join(parts)


def grade_session(
    initial_prompt: str,
    reprompts: list[str],
    outcome_rating: float,
    turn_count: int,
    model: str | None = None,
) -> GraderResult:
    """Grade the initial prompt quality using the local Ollama model."""
    client = _ollama_client()
    if model is None:
        model = _ollama_model()

    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[
            {"role": "system", "content": GRADER_SYSTEM},
            {
                "role": "user",
                "content": _build_user_message(
                    initial_prompt, reprompts, outcome_rating, turn_count
                ),
            },
        ],
    )

    raw = response.choices[0].message.content.strip()

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        return GraderResult(
            prompt_quality=0.5,
            failure_modes=[],
            reasoning="parse error — defaulting to neutral score",
        )

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return GraderResult(
            prompt_quality=0.5,
            failure_modes=[],
            reasoning="json decode error — defaulting to neutral score",
        )

    quality = float(data.get("prompt_quality", 0.5))
    quality = max(0.0, min(1.0, quality))

    modes = [
        m for m in data.get("failure_modes", [])
        if m in VALID_FAILURE_MODES
    ]

    return GraderResult(
        prompt_quality=quality,
        failure_modes=modes,
        reasoning=str(data.get("reasoning", "")),
    )


def extract_reprompts(session_entries: list[dict]) -> list[str]:
    """
    Pull user text messages after the first one from raw JSONL entries.
    These are the clarifications the user added mid-session.
    """
    reprompts: list[str] = []
    first_skipped = False

    for entry in session_entries:
        if entry.get("type") != "user":
            continue
        if entry.get("sourceToolAssistantUUID"):
            continue  # tool result, not user text

        content = entry.get("message", {}).get("content", [])
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").strip()
                if text and not text.startswith("<ide_"):
                    if not first_skipped:
                        first_skipped = True  # skip initial prompt
                    else:
                        reprompts.append(text)
                    break

    return reprompts
