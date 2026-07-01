# TensorZero stack for the glossa eval (Track-B capture)

Brings up the TensorZero gateway + ClickHouse for agent eval and prompt optimization. The eval harness (`kb-eval`) calls the gateway for **`answer_hotpot`** (full agent) and posts `em`/`f1`/`recall`/`judge` feedback. **`kb-train`** uses additional TZ functions for GEPA micro-task scoring.

## Prerequisites

- Docker Desktop (Windows) or Docker on Linux/macOS.
- LM Studio on the host, serving `qwen3.5-4b` on `http://localhost:1234`.
- OpenRouter API key in `.env` for GEPA reflect (`gepa_mutator` / DeepSeek-R1).

## Bring it up

```powershell
cd eval/tensorzero
Copy-Item .env.example .env          # set LMSTUDIO_API_KEY, OPENROUTER_API_KEY
docker compose up -d --wait
```

Or from repo root: `just up`, `just health`.

## Endpoints

| Service | URL |
|---------|-----|
| Gateway | `http://localhost:3000` — `POST /inference`, `POST /feedback` |
| ClickHouse | `http://localhost:8123` (user `chuser`, db `tensorzero`) |

## TZ functions (config/)

| Function | Role |
|----------|------|
| `answer_hotpot` | Full agent: search, grep, glob, read, glossary, neighbors, … |
| `search`, `grep`, `glob`, `read` | GEPA micro-tasks (one tool each); prod prompt passed via `input.system` |
| `gepa_reflect` | Mutator proposes improved system prompt |
| `enrich` | Graph enrichment from solved cases |

Prompts live under `config/*/system.minijinja`. Prod agent prompt: `config/answer_hotpot/system.minijinja`.

After editing config or tool JSON schemas:

```bash
just tools          # regenerate tool schemas from kb MCP definitions
just gw-restart     # reload gateway
```

## Smoke test

```powershell
curl http://localhost:3000/inference -H "Content-Type: application/json" -d '{
  "function_name": "answer_hotpot",
  "input": { "messages": [ { "role": "user", "content": "Question: test" } ] }
}'
```

A tool-call response confirms wiring. The eval harness executes tools in-process against the local glossa index.

## Dev tooling note

This stack is **not** part of the shipped `kb` release binary. See [docs/eval-and-training.md](../../docs/eval-and-training.md) for the full pipeline (eval → export-tz → GEPA).
