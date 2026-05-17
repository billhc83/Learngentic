from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


CHANGE_TYPES = {
    "feature_addition",
    "bug_fix",
    "refactor",
    "config_change",
    "test",
    "documentation",
    "dependency_update",
    "investigation",
    "other",
}

CODE_REGION_TYPES = {
    "auth",
    "data_layer",
    "ui",
    "api",
    "business_logic",
    "infrastructure",
    "testing",
    "tooling",
    "cross_cutting",
    "unknown",
}

CLASSIFIER_SYSTEM = """\
You are a code change classifier for a software quality system.
Given a task prompt and a list of files changed, assign two labels:

1. change_type — what kind of change was made:
   feature_addition, bug_fix, refactor, config_change, test,
   documentation, dependency_update, investigation, other

2. code_region_type — what part of the codebase was touched:
   auth, data_layer, ui, api, business_logic, infrastructure,
   testing, tooling, cross_cutting, unknown

Respond ONLY with valid JSON:
{"change_type": "<type>", "code_region_type": "<region>"}
"""


@dataclass
class ClassifierResult:
    change_type: str
    code_region_type: str


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


def classify_session(
    initial_prompt: str,
    files_changed: list[str],
    model: str | None = None,
) -> ClassifierResult:
    """Classify a session's change type and code region using the local Ollama model."""
    if not initial_prompt and not files_changed:
        return ClassifierResult(change_type="other", code_region_type="unknown")

    client = _ollama_client()
    if model is None:
        model = _ollama_model()

    file_list = "\n".join(f"  - {f}" for f in files_changed[:20])
    user_msg = f"PROMPT:\n{initial_prompt or '(empty)'}\n\nFILES CHANGED:\n{file_list or '  (none)'}"

    response = client.chat.completions.create(
        model=model,
        max_tokens=64,
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = response.choices[0].message.content.strip()
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)

    if not json_match:
        return ClassifierResult(change_type="other", code_region_type="unknown")

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return ClassifierResult(change_type="other", code_region_type="unknown")

    ct = data.get("change_type", "other")
    cr = data.get("code_region_type", "unknown")

    return ClassifierResult(
        change_type=ct if ct in CHANGE_TYPES else "other",
        code_region_type=cr if cr in CODE_REGION_TYPES else "unknown",
    )
