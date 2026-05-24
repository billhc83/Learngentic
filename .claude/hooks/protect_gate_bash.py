#!/usr/bin/env python3
"""PreToolUse hook — blocks Bash commands that write to Learngentic gate paths."""

import json
import os
import sys

_PROTECTED = [
    os.path.expanduser("~/.learngentic"),
    "/mnt/data/projects/Learngentic/.claude/hooks",
]

_WRITE_INDICATORS = [">", "tee ", "cp ", "mv "]

try:
    payload = json.loads(sys.stdin.read())
    command = (payload.get("tool_input") or {}).get("command", "")

    if command:
        for path in _PROTECTED:
            if path in command and any(ind in command for ind in _WRITE_INDICATORS):
                print(
                    f"[LEARNGENTIC GATE] Bash write BLOCKED — protected gate path.\n\n"
                    f"Command targets '{path}', which is a Learngentic system path.\n"
                    f"Gate files may only be modified by the Learngentic MCP server "
                    f"or the user manually.\n"
                )
                sys.exit(2)
except Exception:
    pass

sys.exit(0)
