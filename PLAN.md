# Learngentic × Hermes — Implementation Plan

## Background

### What is Learngentic?

Learngentic is a quality signal system for AI coding agents. It records every coding session — what was asked, what files were changed, how many turns it took — and scores those sessions across five dimensions:

| Signal | Source | What it measures |
|--------|--------|-----------------|
| `execution_efficiency` | Turn count vs task complexity | Did the agent take too many attempts? |
| `durability` | Git history (days until next touch, revert rate) | Did the changes hold up? |
| `outcome_quality` | Human or auto-rater | Was the result actually correct? |
| `prompt_quality` | LLM grader | Was the task well-specified? |
| `robustness_delta` | Blast radius + churn | Did the change cause downstream edits? |

Over time, these scores aggregate into `global_patterns` — statistical standards per task type (e.g. "feature additions to UI components historically take ~18 turns and have 60% durability"). These patterns become the standard Hermes measures itself against.

The project was originally wired to Claude Code's JSONL session logs (a passive ingestion pipeline). That was a mismatch — it made Learngentic a retrospective observer rather than an active participant in the agent loop. This plan fixes that.

---

### What is Hermes?

Hermes is an autonomous coding agent built on NousResearch's Hermes-3/4 models. Unlike Claude Code (which is human-driven, turn by turn), Hermes delegates full tasks and returns completed results. It has:

- XML-based function calling (structured tool use)
- Native MCP (Model Context Protocol) support — it can call external tool servers
- Atropos RL training on 1000+ task verifiers
- Its own session and memory management

The key difference: **Hermes decides when a task is done**. That decision is currently made with no historical grounding. Learngentic gives it that grounding.

---

### Why connect them?

The goal is a feedback loop:

1. **Before a task**: Hermes checks what similar tasks have looked like historically — how many turns, what went wrong, what a good outcome looks like. It uses this to set a self-standard before starting.
2. **During a task**: The standard's completion checklist acts as a gate. Hermes must satisfy it before declaring done.
3. **After a task**: Hermes reports back — was it sent back by the user? Did we already build this? Is this a rewrite? These signals auto-rate the session.
4. **Over time**: Git history validates the ratings. A session rated good but immediately reworked gets its durability score updated. The patterns table improves.

The result: an agent that gets measurably better at knowing when it is and isn't done, grounded in real historical evidence.

---

## Current State

### What's built

- **Database layer** (`src/learngentic/store/db.py`): Fully migrated from local SQLite to Turso (cloud-hosted libSQL). The rest of the codebase uses a sqlite3-compatible interface with no changes needed.
- **Scoring pipeline**: `bayesian.py` (durability from git), `efficiency.py` (turn count scoring), `prompt_grader.py` (LLM grader), `signals.py` (git signal collection) — all intact and reusable.
- **MCP server** (`src/learngentic/mcp/server.py`): Three tools exposed to Hermes — `query_risk_assessment`, `query_outcome_history`, `get_file_stability`. These are read-only; they let Hermes look up history but can't record new sessions yet.
- **Classifier** (`src/learngentic/classifier.py`): Classifies a prompt into (change_type, code_region_type) using an LLM call. Used by the MCP server already.
- **Seed data**: 82 sessions, 1,069 change_events, 28 global_pattern rows migrated to Turso. Enough to bootstrap pattern matching.
- **Schema**: Two new columns — `hermes_assessment REAL` and `assessment_source TEXT` — separate Hermes self-ratings from externally-validated scores.

### What's missing

The MCP server currently has **no write path**. Hermes can query history but cannot record that it started a task, what happened, or what the outcome was. Without that, no new data accumulates and the system never improves.

The Claude Code ingestion pipeline (`session_parser.py`, `hooks/`) is still present but is the wrong data source for a Hermes integration. It reads JSONL files written by Claude Code — Hermes doesn't produce those.

---

## Architecture

```
┌─────────────────────────────────────┐
│              Hermes Agent           │
│                                     │
│  1. Gets task                       │
│  2. → record_task (MCP)        ─────┼──► Turso DB (creates session row)
│  3. Works through checklist         │         │
│  4. → report_outcome (MCP)     ─────┼──► Turso DB (writes hermes_assessment)
│  5. Returns result                  │         │
└─────────────────────────────────────┘         │
                                                 │
┌─────────────────────────────────────┐         │
│         Git Signal Collector        │         │
│  (learngentic score - existing)     │         │
│  Runs after git pushes              │         │
│  Updates durability scores     ─────┼──► Turso DB (flips assessment_source
└─────────────────────────────────────┘          to 'git_grounded')

┌─────────────────────────────────────┐
│         global_patterns table       │
│  Aggregated nightly or on-demand    │
│  Per (change_type, code_region)     │
│  avg_efficiency, avg_durability,    │
│  common_failure_modes, etc.         │
└─────────────────────────────────────┘
```

The MCP server is the only interface between Hermes and Learngentic. Hermes never touches the DB directly.

---

## Implementation Phases

---

### Phase 1 — Standards Generator ✅ DONE
**File**: `src/learngentic/standards/generator.py`

**Why first**: Both new MCP tools (`record_task` and `report_outcome`) need to produce and consume a `TaskStandard` object. The generator is the shared foundation.

**What it does**:

Given a `(change_type, code_region_type)` pair, it queries `global_patterns` and returns a `TaskStandard` containing:

- **Benchmarks**: expected turn range, efficiency threshold, durability floor
- **Completion checklist**: derived by inverting the common failure modes. If `missing_acceptance_criteria` appears in ≥2 sessions, the checklist includes "Confirm acceptance criteria are defined before starting". This turns past failures into a pre-flight check.
- **Data confidence**: `high` (≥10 samples), `medium` (3–9 samples), `low` (<3 samples). Hermes needs to know how much to weight the standard.
- **Cold-start fallback**: When there's no data for a type, return a generic checklist with a `first_instance: true` flag. Hermes still gets structure, just no historical numbers.

**Tiered confidence** matters because Learngentic has two classes of signal:
- **Tier 1** (immediately available): `execution_efficiency` and `durability` — these come from turn counts and git history, no human rating needed.
- **Tier 2** (builds over time): `outcome_quality` and `prompt_quality` — require either a human rating or the prompt grader to run.

The standard is honest about which tier it's drawing from so Hermes can calibrate how strictly to apply it.

**Implementation notes**:
- `_invert_efficiency` converts avg_efficiency back to approximate turn count using `2^(1/eff - 1)`, capped at 200 turns. Without the cap, a very slow session type (eff=0.05) produced 524,288 — Hermes could never exceed it and the "turn count exceeded" warning would never fire.
- Expected turn range: `[expected * 0.6, expected * 1.5]`. Efficiency threshold: `avg_efficiency * 0.75`. Durability floor: `avg_durability * 0.70`.
- Checklist items are sorted by failure mode frequency, most common first. A generic 5-item checklist is used as fallback when no failure modes meet the count≥2 threshold.

---

### Phase 2 — `record_task` MCP Tool ✅ DONE
**File**: `src/learngentic/mcp/server.py` (new tool added)

**Why**: This is the entry point. Without it, nothing is recorded. Every Hermes task starts here.

**What it does**:

Hermes calls this at the start of every non-trivial task, passing:
- `prompt`: the task description
- `cwd`: working directory (repo path)
- `files_mentioned`: list of files the task touches (optional, for layer 2 memory)
- `agent_type`: `"hermes"` (so Claude Code benchmarks don't contaminate Hermes stats)

The tool:
1. Classifies the prompt → (change_type, code_region_type)
2. Creates a session row in Turso with `assessment_source = 'pending'` and `agent_type = 'hermes'`
3. Checks **layer 2 memory** — have any of the mentioned files been touched in the last 24 hours? If yes, flag it (risk of conflict or duplicate work)
4. Calls the standards generator for the classified type
5. Returns: `session_id` (Hermes must store this to pass to `report_outcome`), the `TaskStandard`, and any file recency warnings

**Why the session_id return matters**: Hermes is the session manager here. It must carry the `session_id` through the task and hand it back at `report_outcome`. This is different from Claude Code where session IDs came from the JSONL file. Hermes owns its session lifecycle.

**Known limitation**: The layer-2 file recency check matches on filename only (using `repo_relative_path LIKE '%/filename'`), not full path. A file named `index.js` in one repo will flag against `index.js` changes in a different repo. Low-impact with current data volume; tighten by filtering on `project_id` once cross-project collisions become observable.

---

### Phase 3 — `report_outcome` MCP Tool ✅ DONE
**File**: `src/learngentic/mcp/server.py` (new tool added)

**Why**: This closes the loop. Without it, `record_task` creates orphaned sessions with no outcome data, and the patterns table never improves.

**What it does**:

Hermes calls this before returning a completed task result, passing:
- `session_id`: from the `record_task` response
- `sent_back` (bool): did the user reject or revise the result?
- `already_existed` (bool): was the code Hermes "built" already there?
- `is_rewrite` (bool): is this rewriting something that was built recently?
- `turn_count`: how many internal turns Hermes took
- `hermes_notes` (optional): free text Hermes can use to flag anything unusual

The tool:
1. Computes a preliminary `hermes_assessment` score from the signals:
   - `sent_back = True` → −0.40 (strong negative)
   - `already_existed = True` → −0.20 (wasted work)
   - `is_rewrite = True` → −0.15 (durability concern)
   - All false, turn_count within efficiency threshold → +up to 0.15 efficiency bonus
   - Base score: 0.70; result clamped to [0.0, 1.0]
2. Writes `hermes_assessment` (NOT `outcome_quality` — that remains for externally-validated scores)
3. Sets `assessment_source = 'hermes_self'`
4. Triggers `prompt_grader` to score the initial prompt asynchronously (daemon thread, non-blocking)
5. Maps each negative signal to the specific checklist item it violated
6. Returns: pass/fail verdict, `checklist_items_failed`, `score_breakdown`, and `next_steps`

**Checklist failure mapping**:
- `sent_back` → final "verify acceptance criteria" item
- `already_existed` → "confirm no in-progress changes" item
- `is_rewrite` → "clarify scope" item

**Why separate `hermes_assessment` from `outcome_quality`**:
Hermes is the judge of its own work. That creates a bias problem — a model trained to complete tasks will tend to rate its own completions as complete. `hermes_assessment` is stored separately so it can never directly become the ground truth. Git signals arriving later (a revert, a rapid re-edit) can contradict the self-rating and trigger a score update. The two values coexist; the external signal wins.

**Deferred — `already_existed` deduction is unconditional**: The plan noted this could be "wasted work" OR "neutral (caught early)". Currently it always deducts −0.20. If Hermes detects the duplicate before doing significant work, that's good behaviour and should not be penalised. To fix: add a `caught_early: bool` parameter to `report_outcome`; apply the deduction only when `already_existed=True AND caught_early=False`. Deferred until Hermes usage data shows this matters in practice.

**Known limitation — prompt_quality grading is circular**: The async prompt grader is called with `outcome_rating=hermes_assessment`. Hermes is grading its own prompt using its own outcome score. When git signals later override `hermes_assessment`, `prompt_quality` will not be re-graded. Acceptable for v1; re-grading on `assessment_source` flip is a Phase 4 candidate.

---

### Phase 4 — Git Signal Backfill for Hermes Sessions
**File**: `src/learngentic/cli.py` (update `learngentic score` command)

**Why**: Hermes's self-ratings are preliminary. Git history is the most reliable ground truth we have — files that get immediately reworked, reverted, or churned are evidence the original change wasn't right, regardless of what Hermes said.

**What changes**:

When `learngentic score` runs and finds a session with `assessment_source = 'hermes_self'`, after computing the git signals it should:
1. Recompute durability from the git data (days_until_next_touch, was_reverted, churn_count_30d)
2. If durability diverges significantly from the self-rating, log the discrepancy
3. Flip `assessment_source` to `'git_grounded'`

This makes the system self-correcting. Hermes's self-ratings are a starting point; the git signals are the adjudication.

---

### Phase 5 — Retire Claude Code Ingestion
**Files to remove**:
- `src/learngentic/pipeline/session_parser.py`
- `hooks/pre_execution.py`
- `hooks/post_session.py`
- CLI commands `learngentic ingest` and `learngentic rate`

**Why last**: The new write path (Phases 1–3) must be proven working before removing the old one. There's no urgency — these files don't interfere with Hermes. Retire them once the first real Hermes session has been recorded and retrieved end-to-end.

---

### Phase 6 — Hermes System Prompt
**Why**: The MCP tools only work if Hermes actually calls them. A system prompt is the instruction that makes `record_task` and `report_outcome` part of Hermes's default behaviour, not optional extras.

**What it includes**:
- Instruction to call `record_task` before starting any non-trivial task (>3 turns expected)
- Instruction to treat the returned checklist as a **gate**, not guidance — all items must be checked before calling `report_outcome`
- Instruction to report `sent_back`, `already_existed`, and `is_rewrite` signals honestly — these are diagnostic, not judgements
- Note that `session_id` must be preserved across the task and passed to `report_outcome`

---

## Build Order Summary

| Phase | Deliverable | Depends On | Status |
|-------|-------------|------------|--------|
| 1 | `standards/generator.py` | Nothing (reads existing DB) | ✅ Done |
| 2 | `record_task` MCP tool | Phase 1 | ✅ Done |
| 3 | `report_outcome` MCP tool | Phase 1 | ✅ Done |
| 4 | Git signal backfill | Phases 2–3 (needs real sessions) | Pending |
| 5 | Retire old ingestion | Phase 4 (prove new path works first) | Pending |
| 6 | Hermes system prompt | Phases 2–3 | ✅ Done |

Phases 2 and 3 were built in parallel (same file, no dependency between the two tools). Phase 6 is written last but tested alongside Phase 2.

---

## What Success Looks Like

After Phase 3, a Hermes session should look like this in Turso:

```
sessions row:
  session_id:         "abc-123"
  agent_type:         "hermes"
  change_type:        "feature_addition"
  code_region_type:   "api"
  turn_count:         12
  hermes_assessment:  0.72
  assessment_source:  "hermes_self"   ← will flip to "git_grounded" after Phase 4

After git signals arrive:
  durability:         0.45            ← file was touched again 2 days later
  assessment_source:  "git_grounded"  ← external signal has spoken
```

And in `global_patterns`, the (feature_addition, api) row's `avg_durability` ticks down slightly. The next Hermes task of that type gets a standard that reflects what actually happened.
