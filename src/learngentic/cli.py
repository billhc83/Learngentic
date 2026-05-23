import os
import sys
import json
import click
import requests
from learngentic.store.db import get_conn


def check_config():
    config_path = os.path.expanduser("~/.learngentic/config.json")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"[FAIL] Config: could not read config file: {e}")
        return False

    required_keys = ["turso_url", "turso_auth_token", "ollama_base_url", "ollama_model"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        print(f"[FAIL] Config: missing keys {', '.join(missing)}")
        return False

    masked_token = config['turso_auth_token'][:8] + "..."
    print(
        f"[OK] Config: turso_url={config['turso_url']} "
        f"turso_auth_token={masked_token} "
        f"ollama_base_url={config['ollama_base_url']} "
        f"ollama_model={config['ollama_model']}"
    )
    return True


def check_turso():
    try:
        with get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()
        print(f"[OK] Turso: {row['n']} sessions")
        return True
    except Exception as e:
        print(f"[FAIL] Turso: {e}")
        return False


def check_ollama():
    try:
        config_path = os.path.expanduser("~/.learngentic/config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        # ollama_base_url may end with /v1 — strip it to reach /api/tags
        base = config.get('ollama_base_url', 'http://localhost:11434/v1')
        base = base.rstrip('/').removesuffix('/v1')
        url = f"{base}/api/tags"

        response = requests.get(url, timeout=5)
        response.raise_for_status()
        models = [m['name'] for m in response.json().get('models', [])]
        print(f"[OK] Ollama: {', '.join(models) if models else 'no models found'}")
        return True
    except Exception as e:
        print(f"[FAIL] Ollama: {e}")
        return False


@click.group()
def cli():
    pass


@cli.command(name="sessions")
@click.option("--limit", default=10, help="Number of sessions to show")
def sessions(limit):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT session_id, started_at, agent_type, outcome_quality, initial_prompt "
            "FROM sessions ORDER BY started_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    for r in rows:
        sid = (r["session_id"] or "")[:8]
        ts = (r["started_at"] or "")[:16]
        agent = (r["agent_type"] or "?")[:10]
        score = f"{r['outcome_quality']:.2f}" if r["outcome_quality"] is not None else "—"
        prompt = (r["initial_prompt"] or "")[:60]
        print(f"{sid}  {ts}  {agent:10}  {score:4}  {prompt}")


@cli.command()
@click.option("--days", default=7, show_default=True, help="Rolling window in days")
def logs(days):
    """Visual trend graph for the last N days.

    Shows a daily breakdown table and per-metric sparklines across all
    sessions in the window. Use this to spot whether scores, prompt
    quality, and efficiency are moving in the right direction.
    """
    from datetime import date, timedelta as td

    today     = date.today()
    since     = today - td(days=days - 1)
    since_iso = since.isoformat()

    with get_conn() as conn:
        rows = conn.execute("""
            SELECT started_at, turn_count,
                   COALESCE(outcome_quality, hermes_assessment) AS score,
                   hermes_assessment, prompt_quality,
                   execution_efficiency, durability,
                   assessment_source
            FROM sessions
            WHERE (outcome_quality IS NOT NULL OR hermes_assessment IS NOT NULL)
              AND started_at >= ?
            ORDER BY started_at ASC
        """, (since_iso,)).fetchall()

    total    = len(rows)
    n_git    = sum(1 for r in rows if r["assessment_source"] == "git_grounded")
    n_hermes = sum(1 for r in rows if r["assessment_source"] == "hermes_self")
    n_human  = sum(1 for r in rows if r.get("score") is not None
                   and r["assessment_source"] not in ("git_grounded", "hermes_self", "pending"))
    n_pending = total - n_git - n_hermes - n_human

    width = 60
    click.echo("=" * width)
    click.echo(f"  Learngentic  |  {since.strftime('%b %d')} - {today.strftime('%b %d, %Y')}")
    click.echo(f"  {total} sessions  |  "
               f"git_grounded: {n_git}  hermes_self: {n_hermes}  "
               f"human: {n_human}  pending: {n_pending}")
    click.echo("=" * width)

    day_range = [since + td(days=i) for i in range(days)]

    def _day_avg(day_rows, key):
        vals = [r[key] for r in day_rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    def _fc(v):
        return f"{v:.2f}" if v is not None else " --- "

    click.echo(f"\n  {'Date':<10}  {'#':>3}  {'Score':>6}  {'Prompt':>6}  "
               f"{'Effic.':>6}  {'Turns':>5}  {'Durability':>10}")
    click.echo(f"  {'-'*10}  {'-'*3}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*10}")

    by_day: dict[str, list] = {}
    for r in rows:
        d = (r["started_at"] or "")[:10]
        by_day.setdefault(d, []).append(r)

    for day in day_range:
        iso  = day.isoformat()
        dr   = by_day.get(iso, [])
        n    = len(dr)
        sc   = _day_avg(dr, "score")
        pq   = _day_avg(dr, "prompt_quality")
        ef   = _day_avg(dr, "execution_efficiency")
        dur  = _day_avg(dr, "durability")
        tc   = _day_avg(dr, "turn_count")
        tc_s = f"{tc:.0f}" if tc is not None else "  ---"
        click.echo(f"  {day.strftime('%b %d'):<10}  {n:>3}  {_fc(sc):>6}  "
                   f"{_fc(pq):>6}  {_fc(ef):>6}  {tc_s:>5}  {_fc(dur):>10}")

    if not rows:
        click.echo(f"\n  No scored sessions in the last {days} days.")
        return

    BLOCKS = " .+#@"

    def _spark(values: list, lo: float = 0.0, hi: float = 1.0) -> str:
        out = ""
        for v in values:
            if v is None:
                out += "."
            else:
                idx = int((v - lo) / max(hi - lo, 1e-6) * (len(BLOCKS) - 1))
                out += BLOCKS[max(0, min(idx, len(BLOCKS) - 1))]
        return out

    def _spark_row(label: str, key: str, lo: float = 0.0, hi: float = 1.0):
        vals    = [r[key] for r in rows]
        spark   = _spark(vals, lo, hi)
        defined = [v for v in vals if v is not None]
        if not defined:
            click.echo(f"  {label:<18}  (no data)")
            return
        first = defined[0]
        last  = defined[-1]
        avg   = sum(defined) / len(defined)
        diff  = last - first
        arrow = "up" if diff > 0.03 else ("dn" if diff < -0.03 else "--")
        click.echo(f"  {label:<18}  {spark}  {first:.2f}->{last:.2f}  avg {avg:.2f}  {arrow}")

    click.echo(f"\n  Trends across {total} sessions (oldest -> newest):")
    click.echo(f"  {'-'*18}  {'-'*min(total, 40)}  {'-'*18}")
    _spark_row("score",          "score")
    _spark_row("hermes_assess",  "hermes_assessment")
    _spark_row("prompt quality", "prompt_quality")
    _spark_row("efficiency",     "execution_efficiency")
    _spark_row("durability",     "durability")

    tc_vals = [r["turn_count"] for r in rows]
    max_tc  = max((v for v in tc_vals if v), default=1)
    if max_tc:
        tc_defined = [v for v in tc_vals if v is not None]
        spark_tc   = _spark(tc_vals, lo=0, hi=max_tc)
        avg_tc     = sum(tc_defined) / len(tc_defined) if tc_defined else 0
        click.echo(f"  {'turn count':<18}  {spark_tc}  avg {avg_tc:.0f} turns")

    click.echo()


@cli.command(name="init")
@click.argument("project_dir", default=".", type=click.Path(exists=True, file_okay=False))
def init(project_dir):
    """Scaffold a project for Learngentic tracking.

    Appends the Learngentic workflow rules to CLAUDE.md in PROJECT_DIR
    (defaults to the current directory). Creates the file if absent.
    """
    import textwrap
    project_dir = os.path.abspath(project_dir)

    rules_block = textwrap.dedent("""\

        # Learngentic — Workflow Rules

        Every session is a data point. These rules apply here as they do globally.

        1. Call `record_task` **first** — before reading files, running commands, or writing code.
        2. Use `run_local_task` for all local model work — never `curl` or raw HTTP.
        3. Call `report_local_result` immediately after evaluating `run_local_task` output.
        4. Call `report_outcome` before declaring any task done.

        Skip only for single read-only lookups or one-turn explanations with no tool calls.
        If `record_task` fails due to a server error, proceed and note the failure.
    """)

    claude_md = os.path.join(project_dir, "CLAUDE.md")
    if os.path.exists(claude_md):
        with open(claude_md) as f:
            content = f.read()
        if "record_task" in content:
            click.echo(f"CLAUDE.md already contains Learngentic rules — skipping.")
        else:
            with open(claude_md, "a") as f:
                f.write(rules_block)
            click.echo(f"Appended Learngentic rules to {claude_md}")
    else:
        with open(claude_md, "w") as f:
            f.write(rules_block.lstrip("\n"))
        click.echo(f"Created {claude_md}")

    click.echo(f"[OK] {project_dir} is ready for Learngentic tracking.")


@cli.command(name="status")
def status():
    failures = 0
    if not check_config():
        failures += 1
    if not check_turso():
        failures += 1
    if not check_ollama():
        failures += 1
    sys.exit(failures)


@cli.command(name="track-tool-use")
def track_tool_use():
    """PostToolUse hook target — reads Claude Code hook JSON from stdin, appends to local buffer.

    Never makes a network call; the MCP server syncs the buffer to Turso at report_outcome time.
    Always exits 0 so Claude is never blocked.
    """
    import uuid
    from datetime import datetime, timezone
    from pathlib import Path

    _BUFFER = Path.home() / ".learngentic" / "tool_event_buffer.jsonl"
    _SESSION_FILE = Path.home() / ".learngentic" / "current_session.json"

    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    tool_name = payload.get("tool_name", "")
    if not tool_name:
        sys.exit(0)

    tool_input = payload.get("tool_input") or {}
    hook_cwd = payload.get("cwd", "")

    # File path: present on Edit, Write, Read, Notebook* tools
    file_path = (
        tool_input.get("file_path")
        or tool_input.get("notebook_path")
        or None
    )

    # Associate with current Learngentic session via the session pointer file
    session_id = None
    try:
        cs = json.loads(_SESSION_FILE.read_text())
        session_cwd = cs.get("cwd", "")
        # Match when the hook cwd is inside (or equal to) the session's cwd
        if hook_cwd and session_cwd and (
            hook_cwd == session_cwd
            or hook_cwd.startswith(session_cwd + "/")
            or session_cwd.startswith(hook_cwd + "/")
        ):
            session_id = cs.get("session_id")
    except Exception:
        pass

    if not session_id:
        sys.exit(0)

    event = {
        "event_id": uuid.uuid4().hex[:20],
        "session_id": session_id,
        "tool_name": tool_name,
        "file_path": file_path,
        "cwd": hook_cwd,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        _BUFFER.parent.mkdir(parents=True, exist_ok=True)
        with open(_BUFFER, "a") as fh:
            fh.write(json.dumps(event) + "\n")
    except Exception:
        pass

    sys.exit(0)
