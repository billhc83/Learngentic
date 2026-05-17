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
