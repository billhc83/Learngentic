#!/usr/bin/env python3

import sys, json
from pathlib import Path

HOME         = Path.home()
SESSION_FILE = HOME / ".learngentic" / "current_session.json"
GATE_FILE    = HOME / ".learngentic" / "rlt_gate.json"

# These paths are owned by the Learngentic system — the model must never write them.
# Writing them directly bypasses the gate and fabricates training data.
_PROTECTED = [
    str((HOME / ".learngentic").resolve()),
    str(Path("/mnt/data/projects/Learngentic/.claude/hooks").resolve()),
]


def _is_protected(file_path: str) -> bool:
    try:
        fp = str(Path(file_path).expanduser().resolve())
        return any(fp == p or fp.startswith(p + "/") for p in _PROTECTED)
    except Exception:
        return False


# --- Hard block: protected gate paths (no session check needed) ---
payload = None
try:
    payload = json.loads(sys.stdin.read())
except Exception:
    pass

if payload is not None:
    fp_str = (payload.get("tool_input") or {}).get("file_path", "")
    if fp_str and _is_protected(fp_str):
        print(
            f"[LEARNGENTIC GATE] Write BLOCKED — protected gate path.\n\n"
            f"'{fp_str}' is a Learngentic system file. Direct writes are forbidden.\n"
            f"Gate files may only be modified by the Learngentic MCP server or the user manually.\n"
            f"Writing them directly bypasses the run_local_task gate and fabricates training data.\n"
        )
        sys.exit(2)

# --- Gate check: run_local_task must be called before any edit in an active session ---
try:
    session_data = json.loads(SESSION_FILE.read_text())
    session_id = session_data.get("session_id")
    if not session_id:
        sys.exit(0)

    try:
        gate_data = json.loads(GATE_FILE.read_text())
        if gate_data.get("session_id") == session_id:
            sys.exit(0)
    except FileNotFoundError:
        pass

    print("""[LEARNGENTIC GATE] Edit/Write blocked.

run_local_task has not been called this session.
Delegate the work to the local model first:

  mcp__learngentic__run_local_task(
    task_description="describe what you need generated",
    user_input="the content/context for the model",
    session_id="<session_id from record_task>"
  )

If the response shows local_capable=false, the gate opens automatically and you may proceed.
""")
    sys.exit(2)
except Exception:
    sys.exit(0)
