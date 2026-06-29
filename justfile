# glossa dev pipeline — run `just <recipe>` (bare `just` lists all recipes).
# Install just: `cargo install just`  or  `winget install casey.just`.
# Recipes run under bash (git-bash on Windows); use {{exe}} suffix — bash does not auto-resolve .exe.
# Login shell + preface below: rustup puts cargo in ~/.bashrc, which non-login `-cu` skips on git-bash.
set shell := ["bash", "-lc"]

# Source rustup env / prepend ~/.cargo/bin before any cargo invocation (Windows git-bash).
preface := '[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"; [[ -d "$HOME/.cargo/bin" ]] && export PATH="$HOME/.cargo/bin:$PATH"; '
release := "--release"
bin     := "target/release"
exe     := if os() == 'windows' { '.exe' } else { '' }
kb_bin       := "./" + bin + "/kb" + exe
kb_train_bin := "./" + bin + "/kb-train" + exe
kb_eval_bin  := "./" + bin + "/kb-eval" + exe

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
# Pipeline recipes depend on build-*; skip cargo when binary exists. FORCE_BUILD=1 to rebuild.
build:
    {{preface}}cargo build --workspace {{release}} --locked
build-offline:
    {{preface}}cargo build --workspace {{release}} --locked --offline
build-kb:
    {{preface}}if [[ "${FORCE_BUILD:-}" != "1" ]] && { [[ -x ./target/release/kb ]] || [[ -x ./target/release/kb.exe ]]; }; then echo "kb: already built"; else cargo build {{release}} --bin kb --locked; fi
build-train:
    {{preface}}if [[ "${FORCE_BUILD:-}" != "1" ]] && { [[ -x ./target/release/kb-train ]] || [[ -x ./target/release/kb-train.exe ]]; }; then echo "kb-train: already built"; else cargo build {{release}} -p kb-eval --bin kb-train --locked; fi
build-eval:
    {{preface}}if [[ "${FORCE_BUILD:-}" != "1" ]] && { [[ -x ./target/release/kb-eval ]] || [[ -x ./target/release/kb-eval.exe ]]; }; then echo "kb-eval: already built"; else cargo build {{release}} -p kb-eval --bin kb-eval --locked; fi
test:
    {{preface}}cargo test --workspace {{release}} --locked
check:
    {{preface}}cargo check --workspace {{release}} --locked

# ── TensorZero stack (gateway + clickhouse + ui) ────────────────────────────
up:
    cd {{compose}} && docker compose up -d --wait
down:
    cd {{compose}} && docker compose down
gw-restart:
    docker restart tensorzero-gateway-1
health:
    @curl -s -o /dev/null -w 'gateway %{http_code}\n' {{gateway}}/health
gw-logs:
    docker logs -f --tail 100 tensorzero-gateway-1

tools: build-kb
    {{kb_bin}} mcp dump-tz-tools --config-dir {{tzcfg}}
    @echo "regenerated — run 'just gw-restart' to load the new schemas"

# ── enrich → dump → GEPA (against {{work}}) ─────────────────────────────────
enrich limit="0": build-train
    {{kb_train_bin}} enrich --train {{train}} --work {{work}} --limit {{limit}}

dump: build-train
    @mkdir -p {{out}}
    {{kb_train_bin}} dump --work {{work}} --out {{out}} --once

gepa budget="6" minibatch="4" variant="baseline" run="": build-train
    {{preface}}extra=''; [[ -n "{{run}}" ]] && extra=' --run {{run}}'; {{kb_train_bin}} optimize --select {{out}}/select.jsonl --out {{out}}/select.prompt.txt --budget {{budget}} --minibatch {{minibatch}} --variant {{variant}} $extra

gepa-all budget="6" minibatch="4" variant="baseline" run="": dump
    {{preface}}extra=''; [[ -n "{{run}}" ]] && extra=' --run {{run}}'; {{kb_train_bin}} optimize --select {{out}}/select.jsonl --out {{out}}/select.prompt.txt --budget {{budget}} --minibatch {{minibatch}} --variant {{variant}} $extra

gepa-metrics:
    @just ch "SELECT t.value AS run, round(avgIf(f.value, f.metric_name='select_acc'), 3) AS select_acc, round(avgIf(f.value, f.metric_name='select_baseline_acc'), 3) AS baseline, maxIf(f.value, f.metric_name='gepa_candidates') AS candidates FROM tensorzero.FloatMetricFeedback f JOIN tensorzero.FloatMetricFeedbackTagView t ON f.id = t.feedback_id AND t.key = 'run' WHERE f.metric_name IN ('select_acc','select_baseline_acc','gepa_candidates') GROUP BY run ORDER BY run DESC LIMIT 20 FORMAT PrettyCompact"

# ── eval ────────────────────────────────────────────────────────────────────
eval dataset func="answer_hotpot" corpus="eval-corpus": build-eval
    {{kb_eval_bin}} run --dataset {{dataset}} --backend tensorzero --work {{corpus}} --tensorzero-function {{func}}

eval-fixture: build-eval
    {{kb_eval_bin}} run --dataset eval/fixtures/sample-hotpot-distractor.json --backend mock

# ── inspect ─────────────────────────────────────────────────────────────────
graph-stats: build-kb
    {{kb_bin}} graph stats {{work}}

ch sql:
    @curl -s "http://localhost:8123/?user=chuser&password=chpassword" --data "{{sql}}"

last-episode:
    @just ch "SELECT episode_id FROM tensorzero.ChatInference WHERE function_name='enrich' ORDER BY timestamp DESC LIMIT 1 FORMAT TabSeparatedRaw"
