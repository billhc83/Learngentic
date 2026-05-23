#!/usr/bin/env python3

import sys, json
from pathlib import Path

SESSION_FILE = Path.home() / ".learngentic" / "current_session.json"
GATE_FILE = Path.home() / ".learngentic" / "rlt_gate.json"

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
