# glossa dev pipeline — run `just <recipe>` (bare `just` lists all recipes).
# Install just: `cargo install just`  or  `winget install casey.just`.
# Recipes run under bash (git-bash on Windows); `./target/debug/<bin>` resolves the .exe there.
set shell := ["bash", "-cu"]

work    := "kb-test"                              # corpus root: index + reasoning graph live here
train   := "kb-val/derived/synthetic-train.json"  # solved cases the enricher reverse-traces
tzcfg   := "eval/tensorzero/config"               # TensorZero gateway config + generated tool schemas
compose := "eval/tensorzero"                       # dir holding docker-compose.yml + .env
gateway := "http://localhost:3000"
out     := "gepa-out"                              # dump/GEPA artifacts (git-ignored — derived from corpus)

# list recipes
default:
    @just --list

# ── build & test ────────────────────────────────────────────────────────────
build:
    cargo build --workspace
build-kb:
    cargo build --bin kb
build-train:
    cargo build -p kb-eval --bin kb-train
build-eval:
    cargo build -p kb-eval --bin kb-eval
test:
    cargo test --workspace
# fast check without producing binaries
check:
    cargo check --workspace

# ── TensorZero stack (gateway + clickhouse + ui) ────────────────────────────
# bring the stack up (reads eval/tensorzero/.env for the LMSTUDIO/OPENROUTER keys)
up:
    cd {{compose}} && docker compose up -d --wait
down:
    cd {{compose}} && docker compose down
# reload config (tool schemas / prompts / models) WITHOUT touching clickhouse
gw-restart:
    docker restart tensorzero-gateway-1
health:
    @curl -s -o /dev/null -w 'gateway %{http_code}\n' {{gateway}}/health
gw-logs:
    docker logs -f --tail 100 tensorzero-gateway-1

# regenerate the TZ tool schemas from the live MCP tool defs (run after changing MCP tools),
# then reload the gateway so the new schemas take effect
tools: build-kb
    ./target/debug/kb mcp dump-tz-tools --config-dir {{tzcfg}}
    @echo "regenerated — run 'just gw-restart' to load the new schemas"

# ── enrich → dump → GEPA (against {{work}}) ─────────────────────────────────
# Build the reasoning graph from solved cases (limit=0 → all 185; ~hours).
# Long-running: launch detached, e.g.  nohup just enrich > enrich.log 2>&1 &
enrich limit="0": build-train
    ./target/debug/kb-train enrich --train {{train}} --work {{work}} --limit {{limit}}

# GPU-free: dump the gold-supervision retrieval datasets (query.jsonl + select.jsonl) into {{out}}/
dump: build-train
    @mkdir -p {{out}}
    ./target/debug/kb-train dump --work {{work}} --out {{out}} --once

# GEPA-optimize the select prompt against the dumped select.jsonl
# (per-case select → local qwen; mutator → DeepSeek-R1 through the gateway)
gepa budget="6" minibatch="5":
    ./target/debug/kb-train optimize --select {{out}}/select.jsonl --out {{out}}/select.prompt.txt --budget {{budget}} --minibatch {{minibatch}}

# fresh snapshot, then GEPA on it
gepa-all budget="6" minibatch="5": dump
    @just gepa {{budget}} {{minibatch}}

# ── eval (measure the agent end-to-end) ─────────────────────────────────────
# Run a benchmark through the TZ agent and score it. `dataset` is required, e.g.
#   just eval kb-val/derived/test.json
eval dataset func="answer_hotpot" corpus="eval-corpus": build-eval
    ./target/debug/kb-eval run --dataset {{dataset}} --backend tensorzero --work {{corpus}} --tensorzero-function {{func}}

# ── inspect ─────────────────────────────────────────────────────────────────
# reasoning-graph node/edge counts
graph-stats: build-kb
    ./target/debug/kb graph stats {{work}}
# run a ClickHouse SQL against the TZ observability store, e.g.
#   just ch "SELECT count() FROM tensorzero.ChatInference WHERE function_name='enrich'"
ch sql:
    @curl -s "http://localhost:8123/?user=chuser&password=chpassword" --data "{{sql}}"
# the most recent enrich episode id (feed into your own ClickHouse queries)
last-episode:
    @just ch "SELECT episode_id FROM tensorzero.ChatInference WHERE function_name='enrich' ORDER BY timestamp DESC LIMIT 1 FORMAT TabSeparatedRaw"
