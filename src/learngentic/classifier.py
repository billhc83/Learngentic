"""
LLM-based classifier for change_type and code_region_type.

Runs once per session at ingest time. Uses the initial prompt +
file paths touched to assign abstract cross-project labels that
power the global model.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import anthropic

CHANGE_TYPES = {
    "feature_addition",
    "bug_fix",
    "refactor",
    "config_change",
    "test",
    "documentation",
    "dependency_update",
    "investigation",   # sessions that explore/explain without writing
    "other",
}

CODE_REGION_TYPES = {
    "auth",
    "data_layer",       # database, ORM, migrations, storage
    "ui",               # frontend components, templates, styling
    "api",              # HTTP routes, endpoints, serializers
    "business_logic",   # domain rules, algorithms, services
    "infrastructure",   # CI/CD, config, Docker, env
    "testing",
    "tooling",          # build, scripts, dev tooling
    "cross_cutting",    # changes spanning multiple regions
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


def classify_session(
    initial_prompt: str,
    files_changed: list[str],
    model: str = "claude-haiku-4-5-20251001",
) -> ClassifierResult:
    """Classify a session's change type and code region using Haiku."""
    if not initial_prompt and not files_changed:
        return ClassifierResult(change_type="other", code_region_type="unknown")

    client = anthropic.Anthropic()

    file_list = "\n".join(f"  - {f}" for f in files_changed[:20])
    user_msg = f"PROMPT:\n{initial_prompt or '(empty)'}\n\nFILES CHANGED:\n{file_list or '  (none)'}"

    response = client.messages.create(
        model=model,
        max_tokens=64,
        system=CLASSIFIER_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
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
