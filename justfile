# glossa dev pipeline — run `just <recipe>` (bare `just` lists all recipes).
# Install just: `cargo install just`  or  `winget install casey.just`.
# Recipes run under bash (git-bash on Windows); `./target/release/<bin>` resolves the .exe there.
# Login shell + preface below: rustup puts cargo in ~/.bashrc, which non-login `-cu` skips on git-bash.
set shell := ["bash", "-lc"]

# Source rustup env / prepend ~/.cargo/bin before any cargo invocation (Windows git-bash).
preface := '[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"; [[ -d "$HOME/.cargo/bin" ]] && export PATH="$HOME/.cargo/bin:$PATH"; '
release := "--release"
bin     := "target/release"

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
    {{preface}}cargo build --workspace {{release}}
build-kb:
    {{preface}}cargo build {{release}} --bin kb
build-train:
    {{preface}}cargo build {{release}} -p kb-eval --bin kb-train
build-eval:
    {{preface}}cargo build {{release}} -p kb-eval --bin kb-eval
test:
    {{preface}}cargo test --workspace {{release}}
# fast check without producing binaries
check:
    {{preface}}cargo check --workspace {{release}}

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
    ./{{bin}}/kb mcp dump-tz-tools --config-dir {{tzcfg}}
    @echo "regenerated — run 'just gw-restart' to load the new schemas"

# ── enrich → dump → GEPA (against {{work}}) ─────────────────────────────────
# Build the reasoning graph from solved cases (limit=0 → all 185; ~hours).
# Long-running: launch detached, e.g.  nohup just enrich > enrich.log 2>&1 &
enrich limit="0": build-train
    ./{{bin}}/kb-train enrich --train {{train}} --work {{work}} --limit {{limit}}

# GPU-free: dump the gold-supervision retrieval datasets (query.jsonl + select.jsonl) into {{out}}/
dump: build-train
    @mkdir -p {{out}}
    ./{{bin}}/kb-train dump --work {{work}} --out {{out}} --once

# GEPA-optimize the select prompt (logged in TZ as function `select`; metrics → episode feedback)
#   just gepa run=gepa-v1 variant=baseline
# After changing tensorzero.toml: just gw-restart
gepa budget="6" minibatch="4" variant="baseline" run="": build-train
    #!/usr/bin/env bash
    set -euo pipefail
    args=(optimize --select {{out}}/select.jsonl --out {{out}}/select.prompt.txt --budget {{budget}} --minibatch {{minibatch}} --variant {{variant}})
    if [[ -n "{{run}}" ]]; then args+=(--run "{{run}}"); fi
    ./{{bin}}/kb-train "${args[@]}"

# fresh snapshot, then GEPA on it
gepa-all budget="6" minibatch="4" variant="baseline" run="": dump
    @just gepa {{budget}} {{minibatch}} {{variant}} {{run}}

# compare GEPA runs by TZ tag `run` (ClickHouse; UI filters function + variant + time)
gepa-metrics:
    @just ch "SELECT t.value AS run, round(avgIf(f.value, f.metric_name='select_acc'), 3) AS select_acc, round(avgIf(f.value, f.metric_name='select_baseline_acc'), 3) AS baseline, maxIf(f.value, f.metric_name='gepa_candidates') AS candidates FROM tensorzero.FloatMetricFeedback f JOIN tensorzero.FloatMetricFeedbackTagView t ON f.id = t.feedback_id AND t.key = 'run' WHERE f.metric_name IN ('select_acc','select_baseline_acc','gepa_candidates') GROUP BY run ORDER BY run DESC LIMIT 20 FORMAT PrettyCompact"

# ── eval (measure the agent end-to-end) ─────────────────────────────────────
# Run a benchmark through the TZ agent and score it. `dataset` is required, e.g.
#   just eval kb-val/derived/test.json
eval dataset func="answer_hotpot" corpus="eval-corpus": build-eval
    ./{{bin}}/kb-eval run --dataset {{dataset}} --backend tensorzero --work {{corpus}} --tensorzero-function {{func}}

# ── inspect ─────────────────────────────────────────────────────────────────
# reasoning-graph node/edge counts
graph-stats: build-kb
    ./{{bin}}/kb graph stats {{work}}
# run a ClickHouse SQL against the TZ observability store, e.g.
#   just ch "SELECT count() FROM tensorzero.ChatInference WHERE function_name='enrich'"
ch sql:
    @curl -s "http://localhost:8123/?user=chuser&password=chpassword" --data "{{sql}}"
# the most recent enrich episode id (feed into your own ClickHouse queries)
last-episode:
    @just ch "SELECT episode_id FROM tensorzero.ChatInference WHERE function_name='enrich' ORDER BY timestamp DESC LIMIT 1 FORMAT TabSeparatedRaw"
