"""
Learngentic CLI — ingest, score, rate, report.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import click


LEARNGENTIC_CONFIG_PATH = Path.home() / ".learngentic" / "config.json"


def _ensure_anthropic_api_key() -> None:
    """
    Resolve the Anthropic API key for learngentic LLM calls.

    Lookup order:
      1. ANTHROPIC_API_KEY env var (if you set it system-wide)
      2. ~/.learngentic/config.json  { "anthropic_api_key": "sk-ant-..." }

    DO NOT read from ~/.claude/settings.json — that file's env block is
    injected into Claude Code itself, which causes Claude Code to bill
    every message to your API key instead of using your subscription.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if not LEARNGENTIC_CONFIG_PATH.exists():
        return
    try:
        data = json.loads(LEARNGENTIC_CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    key = data.get("anthropic_api_key")
    if key and not key.startswith("YOUR_"):
        os.environ["ANTHROPIC_API_KEY"] = key


_ensure_anthropic_api_key()

from learngentic.pipeline.git_parser import parse_repo, find_repo_root
from learngentic.store.db import (
    DEFAULT_DB_PATH,
    get_conn,
    upsert_session,
    upsert_change_event,
    update_change_signals,
    update_session_scores,
    get_all_sessions,
    get_change_events_for_session,
)
from learngentic.scoring.bayesian import score_from_signals, aggregate_session_durability, PRIOR
from learngentic.scoring.efficiency import efficiency_score, performance_score
from learngentic.scoring.prompt_grader import grade_session, extract_reprompts
from learngentic.classifier import classify_session
from learngentic.store.vector_index import build_index


@click.group()
def cli():
    """Learngentic — quality signal system for Claude Code."""





@cli.command()
@click.option("--db", default=None, type=click.Path(path_type=Path))
@click.option(
    "--projects-dir", default=None, type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--skip-llm", is_flag=True, default=False,
    help="Skip LLM prompt grading and classification (fast, offline mode)",
)
@click.option("--limit", default=0, help="Max sessions to score with LLM (0 = all)")
def score(db, projects_dir, skip_llm, limit):
    """
    Compute all scoring dimensions for ingested sessions and write back to DB.

    Dimensions computed:
      - execution_efficiency  (from turn_count, no API needed)
      - durability            (Bayesian from git signals, no API needed)
      - robustness_delta      (outcome_quality - prompt_quality, needs both)
      - prompt_quality        (LLM grader — Haiku, one call per session)
      - change_type / code_region_type  (LLM classifier — Haiku, one call per session)
    """
    projects_root = projects_dir or (Path.home() / ".claude" / "projects")

    with get_conn() as conn:
        sessions = get_all_sessions(conn)
        click.echo(f"Scoring {len(sessions)} sessions...")

        llm_count = 0
        llm_cap = limit if limit > 0 else len(sessions)

        for row in sessions:
            session_id = row["session_id"]
            turn_count = row["turn_count"] or 1
            change_events = get_change_events_for_session(conn, session_id)

            # --- execution_efficiency (always computed) ---
            eff = efficiency_score(turn_count)

            # --- durability from git signals ---
            change_scores = []
            for ce in change_events:
                if ce["days_until_next_touch"] is not None or ce["was_reverted"] is not None:
                    s = score_from_signals(
                        days_until_next_touch=ce["days_until_next_touch"],
                        was_reverted=bool(ce["was_reverted"]),
                        churn_count_30d=ce["churn_count_30d"] or 0,
                        blast_radius_7d=ce["blast_radius_7d"] or 0,
                    )
                    change_scores.append(s)

            durability = aggregate_session_durability(change_scores) if change_scores else None

            # --- robustness_delta (needs outcome_quality and prompt_quality) ---
            oq = row["outcome_quality"]
            pq = row["prompt_quality"]
            robustness = (oq - pq) if (oq is not None and pq is not None) else None

            update_kwargs: dict = dict(
                execution_efficiency=eff,
                durability=durability,
                robustness_delta=robustness,
            )

            # --- Phase 4: flip hermes_self to git_grounded once we have real git signals ---
            assessment_source = row.get("assessment_source")
            if assessment_source == "hermes_self" and durability is not None:
                hermes_assessment = row.get("hermes_assessment")
                if hermes_assessment is not None:
                    divergence = abs(durability - hermes_assessment)
                    if divergence > 0.2:
                        click.echo(
                            f"  git_grounded divergence ({session_id[:8]}): "
                            f"hermes_self={hermes_assessment:.2f} "
                            f"vs git_durability={durability:.2f} "
                            f"(delta={divergence:+.2f})"
                        )
                update_kwargs["assessment_source"] = "git_grounded"

            # --- LLM scoring (prompt grader + classifier) ---
            if not skip_llm and llm_count < llm_cap:
                # Locate the JSONL file for this session
                jsonl_path = _find_session_jsonl(projects_root, session_id)
                entries = _load_jsonl_entries(jsonl_path)

                files_touched = [ce["repo_relative_path"] for ce in change_events if ce["repo_relative_path"]]

                # Classifier — skip if already classified
                if not row["change_type"]:
                    try:
                        cls_result = classify_session(
                            initial_prompt=row["initial_prompt"] or "",
                            files_changed=files_touched,
                        )
                        update_kwargs["change_type"] = cls_result.change_type
                        update_kwargs["code_region_type"] = cls_result.code_region_type
                    except Exception as e:
                        click.echo(f"  classifier error ({session_id[:8]}): {e}", err=True)

                # Prompt grader — only if we have an outcome rating
                if row["outcome_quality"] is not None and entries:
                    reprompts = extract_reprompts(entries)
                    try:
                        grade = grade_session(
                            initial_prompt=row["initial_prompt"] or "",
                            reprompts=reprompts,
                            outcome_rating=row["outcome_quality"],
                            turn_count=turn_count,
                        )
                        update_kwargs["prompt_quality"] = grade.prompt_quality
                        update_kwargs["failure_mode_tags"] = grade.failure_modes
                        # Recompute robustness now we have prompt_quality
                        if row["outcome_quality"] is not None:
                            update_kwargs["robustness_delta"] = row["outcome_quality"] - grade.prompt_quality
                    except Exception as e:
                        click.echo(f"  grader error ({session_id[:8]}): {e}", err=True)

                llm_count += 1

            update_session_scores(conn, session_id, **update_kwargs)

        click.echo(f"Done. LLM calls made: {llm_count}")

    # Rebuild vector index after scoring so MCP queries are current
    click.echo("Rebuilding similarity index...")
    idx = build_index()
    click.echo(f"  Indexed {idx['indexed']} sessions.")


def _find_session_jsonl(projects_root: Path, session_id: str) -> Path | None:
    """Locate the JSONL file containing a given session_id."""
    for jsonl in projects_root.rglob("*.jsonl"):
        try:
            content = jsonl.read_text(encoding="utf-8")
            if session_id in content:
                return jsonl
        except OSError:
            pass
    return None


def _load_jsonl_entries(jsonl_path: Path | None) -> list[dict]:
    if not jsonl_path:
        return []
    entries = []
    try:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return entries


QUEUE_PATH = Path.home() / ".learngentic" / "pending_ratings.jsonl"


@cli.command()
@click.option("--db", default=None, type=click.Path(path_type=Path))
@click.option(
    "--projects-dir", default=None, type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--all", "rate_all", is_flag=True, default=False, help="Rate all unrated sessions, not just pending queue")
def rate(db, projects_dir, rate_all):
    """
    Interactively rate completed sessions (0-1) and run the LLM prompt grader.

    Reads the pending queue written by the Stop hook. For each unrated session,
    shows the initial prompt and asks for a score. Then runs the retrospective
    LLM grader to produce prompt_quality + failure_mode_tags.
    """
    from learngentic.pipeline.session_parser import parse_session_file
    from learngentic.pipeline.joiner import _repo_relative
    db_path = Path(db) if db else DEFAULT_DB_PATH
    projects_root = projects_dir or (Path.home() / ".claude" / "projects")

    # Determine which sessions to rate
    pending_ids: list[str] = []

    if rate_all:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT session_id FROM sessions WHERE outcome_quality IS NULL ORDER BY started_at DESC"
            ).fetchall()
            pending_ids = [r["session_id"] for r in rows]
    elif QUEUE_PATH.exists():
        for line in QUEUE_PATH.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                sid = entry.get("session_id", "")
                if sid:
                    pending_ids.append(sid)
            except json.JSONDecodeError:
                pass

    if not pending_ids:
        click.echo("No sessions pending rating. Run `ingest` or wait for the Stop hook to fire.")
        return

    click.echo(f"\n{len(pending_ids)} session(s) to rate. Press Ctrl+C to stop.\n")

    rated: list[str] = []

    with get_conn() as conn:
        for session_id in pending_ids:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not row:
                # Auto-ingest this session on the fly — no need for a full ingest run
                jsonl_path = _find_session_jsonl(projects_root, session_id)
                if jsonl_path:
                    session = parse_session_file(jsonl_path)
                    if session:
                        upsert_session(conn, session)
                        for event in session.change_events:
                            root = find_repo_root(event.repo_path)
                            rel = _repo_relative(
                                str(root) if root else event.repo_path,
                                event.file_path,
                            )
                            upsert_change_event(conn, event, rel)
                        conn.commit()
                        click.echo(f"  Auto-ingested session {session_id[:8]}.")
                        row = conn.execute(
                            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
                        ).fetchone()
                if not row:
                    click.echo(f"  Session {session_id[:8]} not found in JSONL — skipping.")
                    continue

            if row["outcome_quality"] is not None and not rate_all:
                rated.append(session_id)
                continue

            # Show session context
            title = (row["ai_title"] or "").strip()
            prompt_text = (row["initial_prompt"] or "(no prompt captured)").strip()[:200]
            click.echo("-" * 60)
            if title:
                click.echo(f"Title:    {title}")
            click.echo(f"Session:  {session_id[:12]}  [{(row['started_at'] or '')[:16]}]")
            click.echo(f"Project:  {row['cwd'] or '?'}")
            click.echo(f"Turns:    {row['turn_count']}  (efficiency: {row['execution_efficiency']:.2f})" if row['execution_efficiency'] else f"Turns:    {row['turn_count']}")
            click.echo(f"Type:     {row['change_type'] or '?'} / {row['code_region_type'] or '?'}")
            click.echo(f"Prompt:   {prompt_text}")
            click.echo()

            # Get rating
            while True:
                raw = click.prompt(
                    "Outcome quality (0.0 = bad, 1.0 = perfect, Enter to skip)",
                    default="",
                    show_default=False,
                ).strip()
                if raw == "":
                    click.echo("  Skipped.\n")
                    break
                try:
                    score = float(raw)
                    if 0.0 <= score <= 1.0:
                        break
                    click.echo("  Must be between 0.0 and 1.0.")
                except ValueError:
                    click.echo("  Enter a number like 0.7 or press Enter to skip.")

            if raw == "":
                continue

            # Store outcome_quality and recompute efficiency
            update_session_scores(conn, session_id, outcome_quality=score)
            rated.append(session_id)

            # Run LLM grader now that we have the rating
            click.echo("  Running prompt grader...")
            jsonl_path = _find_session_jsonl(projects_root, session_id)
            entries = _load_jsonl_entries(jsonl_path)
            if entries:
                reprompts = extract_reprompts(entries)
                try:
                    grade = grade_session(
                        initial_prompt=row["initial_prompt"] or "",
                        reprompts=reprompts,
                        outcome_rating=score,
                        turn_count=row["turn_count"] or 1,
                    )
                    robustness = score - grade.prompt_quality
                    update_session_scores(
                        conn, session_id,
                        prompt_quality=grade.prompt_quality,
                        failure_mode_tags=grade.failure_modes,
                        robustness_delta=robustness,
                    )
                    click.echo(f"  prompt_quality={grade.prompt_quality:.2f}  robustness_delta={robustness:+.2f}")
                    if grade.failure_modes:
                        click.echo(f"  failure_modes: {', '.join(grade.failure_modes)}")
                    click.echo(f"  grader note: {grade.reasoning}")
                except Exception as e:
                    click.echo(f"  Grader error: {e}", err=True)
            else:
                click.echo("  (no JSONL found for grading)")

            click.echo()

    # Clear rated sessions from queue
    if QUEUE_PATH.exists() and rated:
        remaining = []
        for line in QUEUE_PATH.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                if entry.get("session_id") not in rated:
                    remaining.append(line)
            except json.JSONDecodeError:
                pass
        QUEUE_PATH.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")

    click.echo(f"Done. Rated {len(rated)} session(s).")


@cli.command()
@click.option("--db", default=None, type=click.Path(path_type=Path))
@click.option("--limit", default=20, show_default=True, help="Rows to show per table")
def report(db, limit):
    """Print a summary of ingested sessions and their signals."""
    with get_conn() as conn:
        sessions = get_all_sessions(conn)
        click.echo(f"\n{'='*60}")
        click.echo(f"  Total sessions: {len(sessions)}")
        click.echo(f"{'='*60}\n")

        # Sessions with most turn counts (potential efficiency problems)
        click.echo("Top sessions by turn count (potential efficiency issues):")
        by_turns = sorted(sessions, key=lambda r: r["turn_count"] or 0, reverse=True)
        for row in by_turns[:limit]:
            prompt_preview = (row["initial_prompt"] or "")[:60].replace("\n", " ")
            click.echo(
                f"  turns={row['turn_count']:3d}  "
                f"[{row['started_at'][:10]}]  "
                f"{prompt_preview!r}"
            )

        click.echo()

        # Change events with worst signals
        click.echo("Change events with revert or high churn (worst signals):")
        rows = conn.execute("""
            SELECT ce.*, s.initial_prompt, s.started_at
            FROM change_events ce
            JOIN sessions s ON s.session_id = ce.session_id
            WHERE ce.was_reverted = 1 OR ce.churn_count_30d >= 3
            ORDER BY ce.was_reverted DESC, ce.churn_count_30d DESC
            LIMIT ?
        """, (limit,)).fetchall()

        if rows:
            for row in rows:
                prompt_preview = (row["initial_prompt"] or "")[:50].replace("\n", " ")
                click.echo(
                    f"  reverted={row['was_reverted']}  "
                    f"churn_30d={row['churn_count_30d']}  "
                    f"file={row['repo_relative_path']}  "
                    f"prompt={prompt_preview!r}"
                )
        else:
            click.echo("  (none yet — git signals may still be pending)")

        click.echo()

        # File stability leaderboard
        click.echo("Most unstable files (highest average churn):")
        rows = conn.execute("""
            SELECT
                repo_relative_path,
                COUNT(*)            AS change_count,
                SUM(was_reverted)   AS revert_count,
                AVG(churn_count_30d) AS avg_churn
            FROM change_events
            WHERE repo_relative_path IS NOT NULL
            GROUP BY repo_relative_path
            HAVING change_count > 1
            ORDER BY avg_churn DESC, revert_count DESC
            LIMIT ?
        """, (limit,)).fetchall()

        if rows:
            for row in rows:
                click.echo(
                    f"  changes={row['change_count']}  "
                    f"reverts={row['revert_count']}  "
                    f"avg_churn={row['avg_churn']:.1f}  "
                    f"{row['repo_relative_path']}"
                )
        else:
            click.echo("  (no multi-change files yet)")

        click.echo()

        # Coverage summary
        total_events = conn.execute("SELECT COUNT(*) FROM change_events").fetchone()[0]
        scored_events = conn.execute(
            "SELECT COUNT(*) FROM change_events WHERE days_until_next_touch IS NOT NULL"
        ).fetchone()[0]
        click.echo(
            f"Signal coverage: {scored_events}/{total_events} change events "
            f"have git signals ({100*scored_events//max(total_events,1)}%)"
        )


@cli.command()
@click.option("--db", default=None, type=click.Path(path_type=Path))
@click.option("--window", default=10, show_default=True, help="Sessions per comparison window")
def trend(db, window):
    """
    Show whether your sessions are improving over time.

    Uses outcome_quality (human-rated) when available, falls back to
    hermes_assessment for Hermes sessions. Source shown per row.
    """
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT session_id, started_at, ai_title, initial_prompt,
                   outcome_quality, hermes_assessment, prompt_quality,
                   robustness_delta, execution_efficiency, assessment_source,
                   COALESCE(outcome_quality, hermes_assessment) AS score
            FROM sessions
            WHERE outcome_quality IS NOT NULL
               OR hermes_assessment IS NOT NULL
            ORDER BY started_at ASC
        """).fetchall()

    if not rows:
        click.echo("No scored sessions yet. Hermes sessions appear automatically after report_outcome.")
        return

    total = len(rows)
    n_grounded  = sum(1 for r in rows if r["assessment_source"] == "git_grounded")
    n_hermes    = sum(1 for r in rows if r["assessment_source"] == "hermes_self")
    n_human     = sum(1 for r in rows if r["outcome_quality"] is not None)
    click.echo(f"\n{total} scored session(s)  "
               f"[human-rated: {n_human}  hermes_self: {n_hermes}  git_grounded: {n_grounded}]\n")

    # Sparkline — one char per session, oldest to newest
    BAR = "._,-+*#@"
    line = ""
    for r in rows:
        q = r["score"] or 0.0
        line += BAR[min(int(q * len(BAR)), len(BAR) - 1)]
    click.echo("Score over time (oldest -> newest):")
    click.echo(f"  {line}")
    click.echo(f"  0{' ' * max(len(line) - 2, 0)}1")
    click.echo()

    if total < 2:
        click.echo("Need at least 2 sessions to compute a trend.")
        return

    recent   = list(rows[-window:])
    previous = list(rows[max(0, len(rows) - 2 * window):len(rows) - window])

    def _avg(lst, key):
        vals = [r[key] for r in lst if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    def _dir(now, then):
        if now is None or then is None:
            return "?"
        diff = now - then
        if abs(diff) < 0.03:
            return "-> flat"
        return f"up +{diff:.2f}" if diff > 0 else f"dn  {diff:.2f}"

    def _fmt(v):
        return f"{v:.2f}" if v is not None else "n/a"

    r_sc = _avg(recent,   "score")
    p_sc = _avg(previous, "score")
    r_pq = _avg(recent,   "prompt_quality")
    p_pq = _avg(previous, "prompt_quality")
    r_ef = _avg(recent,   "execution_efficiency")
    p_ef = _avg(previous, "execution_efficiency")
    r_rd = _avg(recent,   "robustness_delta")
    p_rd = _avg(previous, "robustness_delta")

    prev_label = f"prev {len(previous)}" if previous else "n/a"
    rec_label  = f"last {len(recent)}"

    click.echo(f"{'Metric':<24}  {prev_label:>10}  {rec_label:>10}  {'trend':>12}")
    click.echo("-" * 60)
    click.echo(f"{'score (oq|hermes)':<24}  {_fmt(p_sc):>10}  {_fmt(r_sc):>10}  {_dir(r_sc, p_sc):>12}")
    click.echo(f"{'prompt quality':<24}  {_fmt(p_pq):>10}  {_fmt(r_pq):>10}  {_dir(r_pq, p_pq):>12}")
    click.echo(f"{'execution efficiency':<24}  {_fmt(p_ef):>10}  {_fmt(r_ef):>10}  {_dir(r_ef, p_ef):>12}")
    click.echo(f"{'robustness delta':<24}  {_fmt(p_rd):>10}  {_fmt(r_rd):>10}  {_dir(r_rd, p_rd):>12}")
    click.echo()

    # Interpretation
    if r_sc is not None and p_sc is not None:
        if r_sc > p_sc + 0.03:
            click.echo("Score is improving.")
        elif r_sc < p_sc - 0.03:
            click.echo("Score has declined — check recent failure_mode_tags for patterns.")
        else:
            click.echo("Score is holding steady.")

    if r_rd is not None:
        if r_rd > 0.1:
            click.echo("Robustness delta is positive: outcomes beating prompt quality predictions.")
        elif r_rd < -0.1:
            click.echo("Robustness delta is negative: prompts over-promising vs. actual outcomes.")

    # Recent session list
    SOURCE_LABEL = {
        "git_grounded": "git",
        "hermes_self":  "h  ",
        "pending":      "?  ",
    }
    click.echo(f"\nMost recent {len(recent)} session(s):")
    for r in reversed(recent):
        label  = (r["ai_title"] or r["initial_prompt"] or "")[:50].replace("\n", " ")
        src    = SOURCE_LABEL.get(r["assessment_source"] or "", "   ")
        click.echo(f"  [{(r['started_at'] or '')[:10]}] {src} {_fmt(r['score'])}  {label}")
    click.echo()


@cli.command(name="git-score")
def git_score():
    """Incrementally score change_events with git signals.

    Auto-discovers repos from unscored rows — only processes new data.
    Computes days_until_next_touch, was_reverted, churn_count_30d,
    blast_radius_7d, then rolls up durability to sessions and flips
    assessment_source from hermes_self to git_grounded where applicable.
    Safe to run frequently; does nothing if there is no new data.
    """
    from learngentic.pipeline.git_scorer import run_incremental
    click.echo("Scanning for unscored change events...")
    result = run_incremental()
    if result["events_scored"] == 0:
        click.echo("Nothing to score — all change events are up to date.")
        return
    click.echo(f"  Repos processed : {result['repos_processed']}")
    click.echo(f"  Events scored   : {result['events_scored']}")
    click.echo(f"  Sessions updated: {result['sessions_updated']}")
    if result["sessions_updated"] > 0:
        click.echo("Rebuilding similarity index...")
        idx = build_index()
        click.echo(f"  Indexed {idx['indexed']} sessions.")


@cli.command(name="index")
@click.option("--db", default=None, type=click.Path(path_type=Path))
def rebuild_index(db):
    """Rebuild the TF-IDF similarity index for all sessions with prompts."""
    result = build_index()
    click.echo(f"Index built: {result['indexed']} sessions indexed.")


@cli.command()
@click.option("--db", default=None, type=click.Path(path_type=Path))
@click.option(
    "--repo-path",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    multiple=True,
    help="Git repo(s) to load commit history from.",
)
@click.option(
    "--skip-llm", is_flag=True, default=False,
    help="Skip LLM classification and grading (fast, offline mode).",
)
@click.pass_context
def sync(ctx, db, repo_path, skip_llm):
    """Run the full weekly sync: ingest → score → trend → report.

    One command for the day-to-day maintenance loop. Equivalent to running:

      learngentic ingest && learngentic score && learngentic trend && learngentic report

    The similarity index is rebuilt automatically at the end of `score`.
    Use --skip-llm for a fast offline run (no API calls).
    """
    click.echo("=" * 60)
    click.echo("  Learngentic Sync")
    click.echo("=" * 60)
    click.echo()

    click.echo("[ 1/3 ] Ingesting sessions and git history...")
    ctx.invoke(ingest, db=db, repo_path=repo_path, projects_dir=None)
    click.echo()

    click.echo("[ 2/3 ] Scoring sessions...")
    ctx.invoke(score, db=db, projects_dir=None, skip_llm=skip_llm, limit=0)
    click.echo()

    click.echo("[ 3/3 ] Summary")
    click.echo("-" * 40)
    ctx.invoke(trend, db=db, window=10)
    ctx.invoke(report, db=db, limit=10)


@cli.command()
@click.option("--days", default=7, show_default=True, help="Rolling window in days")
def logs(days):
    """Visual trend graph for the last N days.

    Shows a daily breakdown table and per-metric sparklines across all
    sessions in the window. Use this to spot whether scores, prompt
    quality, and efficiency are moving in the right direction.
    """
    from datetime import date, timedelta as td

    today      = date.today()
    since      = today - td(days=days - 1)
    since_iso  = since.isoformat()

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

    total      = len(rows)
    n_git      = sum(1 for r in rows if r["assessment_source"] == "git_grounded")
    n_hermes   = sum(1 for r in rows if r["assessment_source"] == "hermes_self")
    n_human    = sum(1 for r in rows if r.get("score") is not None
                     and r["assessment_source"] not in ("git_grounded", "hermes_self", "pending"))
    n_pending  = total - n_git - n_hermes - n_human

    width = 60
    click.echo("=" * width)
    click.echo(f"  Learngentic  |  {since.strftime('%b %d')} - {today.strftime('%b %d, %Y')}")
    click.echo(f"  {total} sessions  |  "
               f"git_grounded: {n_git}  hermes_self: {n_hermes}  "
               f"human: {n_human}  pending: {n_pending}")
    click.echo("=" * width)

    # ── Daily breakdown table ──────────────────────────────────────────────
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

    # ── Sparklines across all sessions ────────────────────────────────────
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

    def _spark_row(label: str, key: str, lo=0.0, hi=1.0):
        vals  = [r[key] for r in rows]
        spark = _spark(vals, lo, hi)
        defined = [v for v in vals if v is not None]
        if not defined:
            click.echo(f"  {label:<18}  (no data)")
            return
        first = defined[0]
        last  = defined[-1]
        avg   = sum(defined) / len(defined)
        diff  = last - first
        arrow = "up" if diff > 0.03 else ("dn" if diff < -0.03 else "--")
        click.echo(f"  {label:<18}  {spark}  "
                   f"{first:.2f}->{last:.2f}  avg {avg:.2f}  {arrow}")

    click.echo(f"\n  Trends across {total} sessions (oldest -> newest):")
    click.echo(f"  {'-'*18}  {'-'*min(total,40)}  {'-'*18}")
    _spark_row("score",         "score")
    _spark_row("hermes_assess", "hermes_assessment")
    _spark_row("prompt quality","prompt_quality")
    _spark_row("efficiency",    "execution_efficiency")
    _spark_row("durability",    "durability")

    # ── Turn count bar ─────────────────────────────────────────────────────
    tc_vals = [r["turn_count"] for r in rows]
    max_tc  = max((v for v in tc_vals if v), default=1)
    if max_tc:
        tc_defined = [v for v in tc_vals if v is not None]
        spark_tc   = _spark(tc_vals, lo=0, hi=max_tc)
        avg_tc     = sum(tc_defined) / len(tc_defined) if tc_defined else 0
        click.echo(f"  {'turn count':<18}  {spark_tc}  avg {avg_tc:.0f} turns")

    click.echo()


@cli.command()
def patterns():
    """Recompute global_patterns from all scored sessions in Turso.

    Aggregates per (change_type, code_region_type): avg scores, failure mode
    counts, and sample count. Run after a batch of new sessions has been scored
    so that TaskStandards reflect the current data.
    """
    with get_conn() as conn:
        types = conn.execute("""
            SELECT
                change_type,
                code_region_type,
                AVG(CASE WHEN outcome_quality IS NOT NULL THEN outcome_quality END)       AS avg_oq,
                AVG(CASE WHEN execution_efficiency IS NOT NULL THEN execution_efficiency END) AS avg_eff,
                AVG(CASE WHEN prompt_quality IS NOT NULL THEN prompt_quality END)          AS avg_pq,
                AVG(CASE WHEN durability IS NOT NULL THEN durability END)                  AS avg_dur,
                COUNT(*)                                                                    AS n
            FROM sessions
            WHERE change_type IS NOT NULL AND code_region_type IS NOT NULL
            GROUP BY change_type, code_region_type
        """).fetchall()

        updated = 0
        for t in types:
            mode_rows = conn.execute("""
                SELECT failure_mode_tags FROM sessions
                WHERE change_type = ? AND code_region_type = ?
                  AND failure_mode_tags IS NOT NULL AND failure_mode_tags != ''
            """, (t["change_type"], t["code_region_type"])).fetchall()

            mode_counts: dict[str, int] = {}
            for mr in mode_rows:
                try:
                    modes = json.loads(mr["failure_mode_tags"])
                    if isinstance(modes, list):
                        for m in modes:
                            mode_counts[m] = mode_counts.get(m, 0) + 1
                    elif isinstance(modes, dict):
                        for m, c in modes.items():
                            mode_counts[m] = mode_counts.get(m, 0) + int(c)
                except (json.JSONDecodeError, TypeError):
                    pass

            now = datetime.now(timezone.utc).isoformat()
            conn.execute("""
                INSERT INTO global_patterns (
                    change_type, code_region_type,
                    avg_outcome_quality, avg_efficiency, avg_prompt_quality,
                    avg_durability, common_failure_modes, sample_count, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(change_type, code_region_type) DO UPDATE SET
                    avg_outcome_quality  = excluded.avg_outcome_quality,
                    avg_efficiency       = excluded.avg_efficiency,
                    avg_prompt_quality   = excluded.avg_prompt_quality,
                    avg_durability       = excluded.avg_durability,
                    common_failure_modes = excluded.common_failure_modes,
                    sample_count         = excluded.sample_count,
                    last_updated         = excluded.last_updated
            """, (
                t["change_type"], t["code_region_type"],
                t["avg_oq"], t["avg_eff"], t["avg_pq"], t["avg_dur"],
                json.dumps(mode_counts),
                t["n"],
                now,
            ))
            updated += 1
            click.echo(
                f"  {t['change_type']}/{t['code_region_type']}: "
                f"{t['n']} session(s)"
            )

        click.echo(f"\nUpdated {updated} pattern(s) in global_patterns.")


if __name__ == "__main__":
    cli()
