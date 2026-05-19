#!/usr/bin/env python3
"""Stop hook: warn if any record_task sessions were never closed with report_outcome."""
import sys
import signal

signal.alarm(5)  # hard timeout — never block Claude's response for more than 5s

sys.path.insert(0, "/mnt/data/projects/Learngentic/src")
try:
    from learngentic.store.db import get_conn
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT session_id, started_at, initial_prompt "
            "FROM sessions WHERE assessment_source = 'pending' "
            "ORDER BY started_at DESC LIMIT 5"
        ).fetchall()
    if rows:
        print("\nLEARNGENTIC WARNING: Unclosed sessions — report_outcome was never called:")
        for r in rows:
            sid = (r["session_id"] or "")[:8]
            ts = (r["started_at"] or "")[:16]
            prompt = (r["initial_prompt"] or "")[:80]
            print(f"  [{sid}] {ts} — {prompt}")
        print("  → You MUST call report_outcome before declaring this task done.\n")
except Exception:
    pass
