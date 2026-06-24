# TensorZero stack for the glossa eval (Track-B capture)

Brings up the TensorZero gateway + ClickHouse, with the `answer_hotpot` function (prompt +
tools) configured and LM Studio (Qwen3.5-4B) wired as the model provider. The eval harness
(the future `tensorzero` backend) calls the gateway, executes glossa's `search`/`read` in
process, and POSTs `em`/`f1`/`retrieved` feedback — so one eval run banks an optimizable
dataset in ClickHouse.

## Prerequisites
- Docker Desktop (Windows).
- LM Studio running on the host, serving `qwen3.5-4b` on `http://localhost:1234` (the same
  endpoint the `openai` backend uses).

## Bring it up
```powershell
cd eval/tensorzero
Copy-Item .env.example .env          # then edit .env, set the real LMSTUDIO_API_KEY
docker compose up -d --wait
```

## Endpoints (after `up`)
- Gateway: `http://localhost:3000`
  - Native inference: `POST /inference`  (call function `answer_hotpot` with `{ "input": { "messages": [...] } }`)
  - Feedback:         `POST /feedback`   (attach `em`/`f1`/`retrieved` to an episode/inference id)
  - OpenAI-compatible: `POST /openai/v1/chat/completions`
- ClickHouse: `http://localhost:8123`  (user `chuser` / db `tensorzero`)

## Smoke test (model reachable through the gateway)
```powershell
curl http://localhost:3000/inference -H "Content-Type: application/json" -d '{
  "function_name": "answer_hotpot",
  "input": { "messages": [ { "role": "user", "content": "Question: Who wrote Animorphs?" } ] }
}'
```
A tool-call response (the model asking to `search`) confirms the function + tools are wired;
the eval harness is what actually executes the tool against glossa and continues the episode.

## Notes
- The PROMPT lives in `config/answer_hotpot/system.minijinja` (so TensorZero owns/optimizes it),
  seeded identical to the harness's `prompt.rs`. Add variants in `config/tensorzero.toml` to A/B.
- `config/` is mounted read-only into the gateway; edit + `docker compose restart gateway` to apply.
- This is dev tooling; it is not part of the shipped `glossa` binary and has no C-free constraint.
