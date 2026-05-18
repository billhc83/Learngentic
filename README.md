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
│  2. → record_task (MCP)        ─────┼──► Turso DB (creates session row)
│  3. Works through checklist         │         │
│  4. → run_local_task (MCP)     ─────┼──► Turso DB (logs local model attempt)
│  5. → report_local_result (MCP)─────┼──► Turso DB (records pass/fail verdict)
│  6. → report_outcome (MCP)     ─────┼──► Turso DB (writes hermes_assessment)
│  7. Returns result                  │         │
└─────────────────────────────────────┘         │
                                                 │
┌─────────────────────────────────────┐         │
│         Git Signal Collector        │         │
│  (learngentic score - CLI)          │         │
│  Runs after git pushes              │         │
│  Updates durability scores     ─────┼──► Turso DB (flips source to
└─────────────────────────────────────┘          'git_grounded')

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
| `record_task` | Before starting any non-trivial task | `session_id`, `TaskStandard`, file recency warnings |
| `run_local_task` | To delegate any subtask to a local Ollama model | `run_id`, `task_type`, `local_capable` flag, `output`, `model_used` |
| `report_local_result` | After evaluating every `run_local_task` output | Confirmation the verdict was stored |
| `report_outcome` | Before returning a completed result to the user | Pass/fail verdict, score breakdown, next steps |
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

Optionally set per-task-type model overrides:

```json
{
  "task_models": {
    "code_task":         "qwen2.5-coder:14b",
    "analytical":        "qwen3:14b",
    "structured_output": "qwen3-14b-nothink",
    "summarization":     "qwen3-14b-nothink"
  }
}
```

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
learngentic sessions          # list recent sessions
learngentic sessions --limit 25
learngentic status            # check config, Turso, and Ollama connectivity
```

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

Response includes a `TaskStandard` with benchmarks and a completion checklist. The checklist is a **gate** — every item must be satisfied before calling `report_outcome`.

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

The server classifies the task, checks historical pass rates for that task type, selects the appropriate model, runs it, and logs the attempt. If `local_capable` is `false`, handle the subtask yourself.

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
  "turn_count": 12
}
```

The verdict and any `next_steps` are returned. Git signals arriving after a push may later override the self-rating with a `git_grounded` score.

## Score lifecycle

```
record_task       → sessions.assessment_source: 'pending'
report_outcome    → sessions.assessment_source: 'hermes_self'   (self-rated)
learngentic score → sessions.assessment_source: 'git_grounded'  (externally validated)

run_local_task + report_local_result → logged separately in local_model_runs table
                                        verdict ('pass'/'fail'/'gave_up') drives
                                        per-task-type pass rate and model selection
```

Self-rated scores are stored in `hermes_assessment`, separate from `outcome_quality`. If git history contradicts the self-rating (file reverted, rapidly reworked), the git signal wins.

## Project structure

```
src/learngentic/
├── mcp/
│   └── server.py          # MCP server — all seven tools
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
