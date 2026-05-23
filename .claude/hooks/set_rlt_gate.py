#!/usr/bin/env python3

import sys, json
from pathlib import Path
from datetime import datetime, timezone

SESSION_FILE = Path.home() / ".learngentic" / "current_session.json"
GATE_FILE = Path.home() / ".learngentic" / "rlt_gate.json"

try:
    payload = json.load(sys.stdin)
    if payload.get("tool_name") != "mcp__learngentic__run_local_task":
        sys.exit(0)

    session_data = json.loads(SESSION_FILE.read_text())
    session_id = session_data.get("session_id")
    if not session_id:
        sys.exit(0)

    gate_file_content = {
        "session_id": session_id,
        "set_at": datetime.now(timezone.utc).isoformat()
    }
    GATE_FILE.write_text(json.dumps(gate_file_content))
except Exception:
    sys.exit(0)

sys.exit(0)
