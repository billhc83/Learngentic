# Learngentic

A quality signal system for AI coding agents. Learngentic records coding sessions, scores them across five dimensions, and aggregates those scores into historical standards — so agents can measure themselves against what has actually worked before.

## What it does

Every time a supported agent (Hermes, Claude Code) starts a task, Learngentic:

1. **Classifies** the prompt into a `(change_type, code_region_type)` pair (e.g. `feature_addition / api`)
2. **Looks up** historical patterns for that type — average turn counts, efficiency, durability, common failure modes
3. **Returns a `TaskStandard`** — a completion checklist derived by inverting past failures, plus benchmark ranges for turns and efficiency
4. **Records the outcome** — self-reported by the agent on completion, then validated later by git history

Over time the patterns table improves: a session rated good that gets immediately reworked has its durability score updated automatically. The standards get tighter.

## Scoring dimensions

| Signal | Source | What it measures |
|--------|--------|-----------------|
| `execution_efficiency` | Turn count vs complexity | Did the agent take too many attempts? |
| `durability` | Git history (days until next touch, revert rate) | Did the changes hold up? |
| `outcome_quality` | Human or auto-rater | Was the result actually correct? |
| `prompt_quality` | LLM grader | Was the task well-specified? |
| `robustness_delta` | Blast radius + churn | Did the change cause downstream edits? |

## Architecture

```
┌─────────────────────────────────────┐
│              Agent (Hermes)         │
│                                     │
│  1. Gets task                       │
│  2. → record_task (MCP)        ─────┼──► Turso DB (creates session row,
│  3. Works through checklist         │         writes current_session.json)
│  4. → run_local_task (MCP)     ─────┼──► Turso DB (logs local model attempt)
│  5. → report_local_result (MCP)─────┼──► Turso DB (records pass/fail verdict)
│  6. → report_outcome (MCP)     ─────┼──► Turso DB (writes hermes_assessment,
│  7. Returns result                  │         syncs tool_event buffer,
└─────────────────────────────────────┘         triggers async git ingest)
                                                 │
┌─────────────────────────────────────┐         │
│   PostToolUse hook (track-tool-use) │         │
│   Captures every Claude tool call   │         │
│   to ~/.learngentic/                │         │
│   tool_event_buffer.jsonl      ─────┼──► synced to Turso at report_outcome
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│         global_patterns table       │
│  Aggregated nightly or on-demand    │
│  Per (change_type, code_region)     │
│  avg_efficiency, avg_durability,    │
│  common_failure_modes, etc.         │
└─────────────────────────────────────┘
```

The MCP server is the only interface between an agent and Learngentic. Agents never touch the database directly.

## MCP tools

Seven tools are exposed via stdio MCP:

| Tool | When to call | Returns |
|------|-------------|---------|
| `record_task` | Before starting any non-trivial task | `session_id`, `TaskStandard`, file recency warnings, recent prompt quality history |
| `run_local_task` | To delegate any subtask to a local Ollama model | `run_id`, `task_type`, `local_capable` flag, `output`, `model_used` |
| `report_local_result` | After evaluating every `run_local_task` output | Confirmation the verdict was stored |
| `report_outcome` | Before returning a completed result to the user | Pass/fail verdict, score breakdown, checklist compliance, objective signals, next steps |
| `query_risk_assessment` | Pre-execution: should we ask clarifying questions? | Risk level, grounded questions from past failures |
| `query_outcome_history` | Find similar past sessions and their scores | Semantic + classification-matched sessions |
| `get_file_stability` | Before touching a high-churn file | Change frequency, revert rate, churn stats |

## Installation

Requires Python 3.11+ and a [Turso](https://turso.tech) database.

```bash
pip install -e .
```

Create `~/.learngentic/config.json`:

```json
{
  "turso_url": "libsql://your-db.turso.io",
  "turso_auth_token": "your-token",
  "ollama_base_url": "http://localhost:11434/v1",
  "ollama_model": "hermes3"
}
```

Optionally set per-task-type model overrides and local model capability thresholds:

```json
{
  "task_models": {
    "code_task":         "qwen2.5-coder:14b",
    "analytical":        "qwen3:14b",
    "structured_output": "qwen3-14b-nothink",
    "summarization":     "qwen3-14b-nothink"
  },
  "local_capable_threshold": 0.4,
  "local_capable_min_samples": 5
}
```

`local_capable_threshold` (default 0.4) is the minimum historical pass rate below which `run_local_task` returns `local_capable: false`. `local_capable_min_samples` (default 5) is the minimum number of recorded attempts before the threshold is applied — below that sample count the model is always considered capable.

Verify the setup:

```bash
learngentic status
```

## Running the MCP server

```bash
python -m learngentic.mcp.server
```

Register it as a stdio MCP server in your agent's settings. For Claude Code on Linux, add to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "learngentic": {
      "command": "/path/to/venv/bin/python3",
      "args": ["-m", "learngentic.mcp.server"]
    }
  }
}
```

## CLI

```bash
learngentic sessions             # list recent sessions
learngentic sessions --limit 25

learngentic logs                 # trend graph for the last 7 days
learngentic logs --days 14       # extend the rolling window

learngentic status               # check config, Turso, and Ollama connectivity

learngentic init [project_dir]   # scaffold a project for Learngentic tracking
                                 # appends workflow rules to CLAUDE.md

learngentic track-tool-use       # PostToolUse hook target (see Hooks section)
```

### `learngentic logs`

Shows a daily breakdown table and per-metric sparklines across all scored sessions in the window. Each row reports session count, average score, prompt quality, execution efficiency, average turn count, and durability. Sparklines below the table show how each metric trends across individual sessions (oldest to newest), with a direction arrow and first→last comparison.

```
============================================================
  Learngentic  |  May 17 - May 23, 2026
  42 sessions  |  git_grounded: 12  hermes_self: 28  pending: 2
============================================================

  Date          #   Score  Prompt  Effic.  Turns  Durability
  ----------  ---  ------  ------  ------  -----  ----------
  May 17        6    0.71    0.68    0.34     11        0.81
  ...

  Trends across 42 sessions (oldest -> newest):
  score               +#++@  0.62->0.78  avg 0.72  up
  prompt quality      .++#@  0.51->0.74  avg 0.65  up
```

## Hooks (Claude Code integration)

The `.claude/hooks/` directory contains three hooks:

1. **PostToolUse — `track-tool-use`**: The `learngentic track-tool-use` CLI command reads Claude Code hook JSON from stdin, appending tool call events to `~/.learngentic/tool_event_buffer.jsonl`. It never makes network calls and always exits with status 0. Add to `.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "",
        "hooks": [{ "type": "command", "command": "learngentic track-tool-use" }]
      }
    ]
  }
}
```

2. **PreToolUse — `check_rlt_gate.py`**: Blocks `Edit` and `Write` tool calls until `run_local_task` has been invoked in the current session. Opens automatically when `local_capable: false` is returned, or after any successful `run_local_task` call.

3. **PreToolUse — `check_open_sessions.py`**: Displays a warning if a new session starts while a prior session has not yet called `report_outcome`. Prevents orphaned session rows.

## Objective signals

`record_task` writes `~/.learngentic/current_session.json` so the PostToolUse hook can associate every tool call event with the correct session. At `report_outcome` time, the event buffer is batch-synced to Turso's `tool_events` table.

The `objective_signals` block returned by `report_outcome` includes:

- `observed_tool_calls`: count of tool calls captured by the hook
- `self_reported_turns`: the `turn_count` value passed to `report_outcome`
- `turn_divergence`: relative delta between observed and self-reported turns
- `scope_expansion_ratio`: fraction of files edited that were outside `files_mentioned`
- `edited_files`: list of files actually touched during the session

`objective_warnings` fires when `turn_divergence > 50%` or `scope_expansion_ratio > 50%`. An `overrun_penalty` of `0.10` is applied to `hermes_assessment` when observed turns exceed `2×` the historical `expected_turns_high`.

## Agent integration (Hermes / Claude Code)

The full task lifecycle has four mandatory MCP calls:

**1. Start — `record_task`**

```json
{
  "prompt": "Add rate limiting to the /api/users endpoint",
  "cwd": "/path/to/repo",
  "files_mentioned": ["src/api/users.py", "src/middleware/rate_limit.py"],
  "agent_type": "hermes"
}
```

Response includes a `TaskStandard` with benchmarks and a completion checklist, file recency warnings (files touched in the last 24 hours), and recent `prompt_quality` scores for this task type so the agent can see its own track record. The checklist is a **gate** — every item must be satisfied before calling `report_outcome`.

**2. Local model work — `run_local_task` + `report_local_result`**

For any subtask delegated to a local Ollama model (commit messages, code review, summarization, structured output, etc.):

```json
{
  "task_description": "Write a conventional commit message for this diff",
  "user_input": "<the diff>",
  "session_id": "abc-123",
  "attempt_number": 1
}
```

The server classifies the task type, checks the historical pass rate to determine `local_capable`, picks the appropriate model, runs it, and logs the attempt. If `local_capable` is `false`, handle the subtask yourself — do not retry.

After evaluating the output, record the verdict:

```json
{ "run_id": "def-456", "verdict": "pass" }
```

On a failed attempt, retry with `attempt_number` incremented and `previous_run_id` set to the last `run_id`. Max 5 attempts.

**3. Complete — `report_outcome`**

```json
{
  "session_id": "abc-123",
  "sent_back": false,
  "already_existed": false,
  "is_rewrite": false,
  "turn_count": 12,
  "checklist_passed": [0, 1, 2, 3, 4]
}
```

`checklist_passed` is a list of 0-based indices of completion checklist items the agent satisfied. Used to compute a `checklist_compliance` rate stored alongside the session.

The response includes:

| Field | Description |
|-------|-------------|
| `verdict` | `pass` or `fail` |
| `hermes_assessment` | Computed score (0–1), stored separately from `outcome_quality` |
| `score_breakdown` | Per-signal contribution list |
| `checklist_compliance` | Fraction of checklist items self-reported satisfied |
| `guidance_adherence` | Fraction of Learngentic guidance tools actually called this session |
| `objective_signals` | Hook-observed turn count, divergence, scope expansion, edited files |
| `objective_warnings` | Alerts when divergence or scope expansion exceed thresholds |
| `next_steps` | Concrete remediation steps when verdict is `fail` |

`report_outcome` also triggers an async git ingest for commits made after the session started, computing a preliminary durability score without waiting for the CLI scorer.

Git signals arriving after a push may later override the self-rating with a `git_grounded` score.

## Score lifecycle

```
record_task       → sessions.assessment_source: 'pending'
report_outcome    → sessions.assessment_source: 'hermes_self'   (self-rated)
learngentic score → sessions.assessment_source: 'git_grounded'  (externally validated)

run_local_task + report_local_result → logged in local_model_runs table
                                        verdict ('pass'/'fail'/'gave_up') drives
                                        per-task-type pass rate and model selection

PostToolUse hook  → tool_event_buffer.jsonl → synced to tool_events table
                                               at report_outcome time
```

Self-rated scores are stored in `hermes_assessment`, separate from `outcome_quality`. If git history contradicts the self-rating (file reverted, rapidly reworked), the git signal wins.

## Project structure

```
src/learngentic/
├── mcp/
│   └── server.py          # MCP server — all seven tools + objective signal computation
├── local_tasks/
│   └── runner.py          # local model execution: classify, select model, run, log
├── scoring/
│   ├── bayesian.py        # durability from git history
│   ├── efficiency.py      # turn count → efficiency score
│   ├── prompt_grader.py   # LLM-based prompt quality grader
│   └── signals.py         # git signal collection
├── standards/
│   └── generator.py       # TaskStandard from global_patterns
├── store/
│   ├── db.py              # Turso (libSQL) connection layer
│   └── vector_index.py    # semantic similarity search
├── pipeline/
│   ├── git_parser.py      # parse git commits into change events
│   ├── git_scorer.py      # score sessions from git data
│   └── joiner.py          # join sessions with change events
├── classifier.py          # LLM-based prompt → (change_type, region) classifier
└── cli.py                 # learngentic CLI

.claude/hooks/
├── check_rlt_gate.py      # PreToolUse: block Edit/Write until run_local_task is called
├── set_rlt_gate.py        # PostToolUse: open the gate after run_local_task
└── check_open_sessions.py # PreToolUse: warn on unclosed sessions
```

## Hard rules for Claude Code

These are enforced via CLAUDE.md and hooks — not optional:

1. Call `record_task` as the **first action** before any file edits or tool calls.
2. Use `run_local_task` for **all** local model work — never call Ollama via `curl` or raw HTTP.
3. Call `report_local_result` immediately after evaluating every `run_local_task` output.
4. Call `report_outcome` before telling the user a task is done.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
