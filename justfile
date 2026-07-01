# glossa dev pipeline — run `just <recipe>` (bare `just` lists all recipes).
# Install just: `cargo install just`  or  `winget install casey.just`.
# Recipes run under bash (git-bash on Windows). Cargo emits `kb-train` (GNU/WSL) or `kb-train.exe` (MSVC).
set shell := ["bash", "-lc"]

preface := '[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"; [[ -d "$HOME/.cargo/bin" ]] && export PATH="$HOME/.cargo/bin:$PATH"; '
release := "--release"
bin     := "target/release"
kb_bin       := if os() == "windows" { "./" + bin + "/kb.exe" } else { "./" + bin + "/kb" }
kb_train_bin := if os() == "windows" { "./" + bin + "/kb-train.exe" } else { "./" + bin + "/kb-train" }
kb_eval_bin  := if os() == "windows" { "./" + bin + "/kb-eval.exe" } else { "./" + bin + "/kb-eval" }

work    := "kb-test"                              # corpus root: index + reasoning graph live here
train   := "kb-val/derived/synthetic-train.json"  # solved cases the enricher reverse-traces
tzcfg   := "eval/tensorzero/config"               # TensorZero gateway config + generated tool schemas
compose := "eval/tensorzero"                       # dir holding docker-compose.yml + .env
gateway := "http://127.0.0.1:3000"
out     := "gepa-out"                              # dump/GEPA artifacts (git-ignored — derived from corpus)

# list recipes
default:
    @just --list

# ── build & test ────────────────────────────────────────────────────────────
# Pipeline recipes depend on build-*; skip cargo when binary exists.
# PowerShell: `just build-train force`
build:
    {{preface}}cargo build --workspace {{release}} --locked
build-offline:
    {{preface}}cargo build --workspace {{release}} --locked --offline
build-kb force="":
    {{preface}}b='{{kb_bin}}'; if [[ -z "{{force}}" && "${FORCE_BUILD:-}" != "1" ]] && [[ -x "$b" ]]; then echo "kb: already built"; else cargo build {{release}} --bin kb --locked; fi
build-train force="":
    {{preface}}b='{{kb_train_bin}}'; if [[ -z "{{force}}" && "${FORCE_BUILD:-}" != "1" ]] && [[ -x "$b" ]]; then echo "kb-train: already built"; else cargo build {{release}} -p kb-eval --bin kb-train --locked; fi
build-train-offline force="":
    {{preface}}b='{{kb_train_bin}}'; if [[ -z "{{force}}" && "${FORCE_BUILD:-}" != "1" ]] && [[ -x "$b" ]]; then echo "kb-train: already built"; else cargo build {{release}} -p kb-eval --bin kb-train --locked --offline; fi
build-eval force="":
    {{preface}}b='{{kb_eval_bin}}'; if [[ -z "{{force}}" && "${FORCE_BUILD:-}" != "1" ]] && [[ -x "$b" ]]; then echo "kb-eval: already built"; else cargo build {{release}} -p kb-eval --bin kb-eval --locked; fi
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
    {{preface}}{{kb_bin}} mcp dump-tz-tools --config-dir {{tzcfg}}
    @echo "regenerated — run 'just gw-restart' to load the new schemas"

# ── enrich → export-tz → GEPA (against {{work}}) ───────────────────────────
enrich limit="0": build-train
    {{preface}}{{kb_train_bin}} enrich --train {{train}} --work {{work}} --limit {{limit}}

export-tz run="" k="10": build-train
    @mkdir -p {{out}}
    {{preface}}extra=''; [[ -n "{{run}}" ]] && extra=' --run {{run}}'; \
    {{kb_train_bin}} export-tz --work {{work}} --out {{out}} \
      --train kb-val/derived/synthetic-train.json \
      --train kb-val/derived/train.json \
      --k {{k}} $extra

dump: build-train
    @mkdir -p {{out}}
    {{preface}}{{kb_train_bin}} dump --work {{work}} --out {{out}} --once

# GEPA long-run optimize — continues from {{out}}/answer_hotpot.prompt.txt when present.
# Override: `just gepa budget=12 run=my-run`
gepa budget="40" minibatch="12" variant="baseline" run="" w_search="0.35" w_read="0.40" w_grep="0.15" w_glob="0.10": build-train
    @mkdir -p {{out}}
    {{preface}}run_tag='{{run}}'; [[ -z "$run_tag" ]] && run_tag="gepa-long-$(date +%Y%m%d-%H%M)"; \
    if [[ -f {{out}}/answer_hotpot.prompt.txt ]]; then \
      seed_arg="--seed {{out}}/answer_hotpot.prompt.txt"; \
    else \
      seed_arg="--seed {{tzcfg}}/answer_hotpot/system.minijinja"; \
    fi; \
    {{kb_train_bin}} optimize \
      --search {{out}}/search.jsonl --grep {{out}}/grep.jsonl \
      --glob {{out}}/glob.jsonl --read {{out}}/read.jsonl \
      --out {{out}}/answer_hotpot.prompt.txt \
      $seed_arg \
      --work {{work}} --budget {{budget}} --minibatch {{minibatch}} \
      --w-search {{w_search}} --w-read {{w_read}} --w-grep {{w_grep}} --w-glob {{w_glob}} \
      --pareto-size 20 \
      --variant {{variant}} --run "$run_tag"

gepa-all budget="40" minibatch="12" variant="baseline" run="" w_search="0.35" w_read="0.40" w_grep="0.15" w_glob="0.10": export-tz
    @mkdir -p {{out}}
    {{preface}}run_tag='{{run}}'; [[ -z "$run_tag" ]] && run_tag="gepa-long-$(date +%Y%m%d-%H%M)"; \
    if [[ -f {{out}}/answer_hotpot.prompt.txt ]]; then \
      seed_arg="--seed {{out}}/answer_hotpot.prompt.txt"; \
    else \
      seed_arg="--seed {{tzcfg}}/answer_hotpot/system.minijinja"; \
    fi; \
    {{kb_train_bin}} optimize \
      --search {{out}}/search.jsonl --grep {{out}}/grep.jsonl \
      --glob {{out}}/glob.jsonl --read {{out}}/read.jsonl \
      --out {{out}}/answer_hotpot.prompt.txt \
      $seed_arg \
      --work {{work}} --budget {{budget}} --minibatch {{minibatch}} \
      --w-search {{w_search}} --w-read {{w_read}} --w-grep {{w_grep}} --w-glob {{w_glob}} \
      --pareto-size 20 \
      --variant {{variant}} --run "$run_tag"

# Apply optimized prompt to prod template (backs up seed first).
gepa-apply:
    @test -f {{out}}/answer_hotpot.prompt.txt || (echo "missing {{out}}/answer_hotpot.prompt.txt — run just gepa first" && exit 1)
    cp {{tzcfg}}/answer_hotpot/system.minijinja {{tzcfg}}/answer_hotpot/system.minijinja.bak
    cp {{out}}/answer_hotpot.prompt.txt {{tzcfg}}/answer_hotpot/system.minijinja
    @echo "applied {{out}}/answer_hotpot.prompt.txt -> {{tzcfg}}/answer_hotpot/system.minijinja (backup: {{tzcfg}}/answer_hotpot/system.minijinja.bak)"
    @echo "run: just gw-restart && just eval … run=after-gepa"

gepa-metrics:
    @just ch "SELECT t.value AS run, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_baseline_combined'), 3) AS baseline, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_combined_acc'), 3) AS final, round(avgIf(f.value, f.metric_name='gepa_iter_combined'), 3) AS iter_avg, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_final_search'), 3) AS search, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_final_grep'), 3) AS grep, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_final_glob'), 3) AS glob, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_final_read'), 3) AS read, maxIf(f.value, f.metric_name='gepa_candidates') AS candidates FROM tensorzero.FloatMetricFeedback f JOIN tensorzero.FloatMetricFeedbackTagView t ON f.id = t.feedback_id AND t.key = 'run' WHERE f.metric_name IN ('gepa_baseline_combined','gepa_combined_acc','gepa_iter_combined','gepa_final_search','gepa_final_grep','gepa_final_glob','gepa_final_read','gepa_candidates') GROUP BY run ORDER BY run DESC LIMIT 20 FORMAT PrettyCompact"

# Wipe GEPA run history in ClickHouse (search/read/gepa_reflect + episode metrics). Does not touch enrich/eval/coding.
gepa-reset:
    @just ch "ALTER TABLE tensorzero.ModelInference DELETE WHERE inference_id IN (SELECT id FROM tensorzero.ChatInference WHERE function_name IN ('search', 'grep', 'glob', 'read', 'gepa_reflect'))"
    @just ch "ALTER TABLE tensorzero.InferenceTag DELETE WHERE inference_id IN (SELECT id FROM tensorzero.ChatInference WHERE function_name IN ('search', 'grep', 'glob', 'read', 'gepa_reflect'))"
    @just ch "ALTER TABLE tensorzero.ChatInference DELETE WHERE function_name IN ('search', 'grep', 'glob', 'read', 'gepa_reflect')"
    @just ch "ALTER TABLE tensorzero.FeedbackTag DELETE WHERE feedback_id IN (SELECT id FROM tensorzero.FloatMetricFeedback WHERE metric_name IN ('gepa_baseline_search','gepa_baseline_grep','gepa_baseline_glob','gepa_baseline_read','gepa_baseline_combined','gepa_iter_search','gepa_iter_grep','gepa_iter_glob','gepa_iter_read','gepa_iter_combined','gepa_iter_candidates','gepa_final_search','gepa_final_grep','gepa_final_glob','gepa_final_read','gepa_final_combined','gepa_combined_acc','gepa_candidates','gepa_examples_train','gepa_examples_val','select_acc','select_baseline_acc','search_acc','search_baseline_acc','read_acc','gepa_baseline_query','gepa_iter_query','gepa_final_query','query_acc','query_baseline_acc'))"
    @just ch "ALTER TABLE tensorzero.FloatMetricFeedback DELETE WHERE metric_name IN ('gepa_baseline_search','gepa_baseline_grep','gepa_baseline_glob','gepa_baseline_read','gepa_baseline_combined','gepa_iter_search','gepa_iter_grep','gepa_iter_glob','gepa_iter_read','gepa_iter_combined','gepa_iter_candidates','gepa_final_search','gepa_final_grep','gepa_final_glob','gepa_final_read','gepa_final_combined','gepa_combined_acc','gepa_candidates','gepa_examples_train','gepa_examples_val','select_acc','select_baseline_acc','search_acc','search_baseline_acc','read_acc','gepa_baseline_query','gepa_iter_query','gepa_final_query','query_acc','query_baseline_acc')"
    @echo "gepa-reset: mutations queued — wait ~5s then: just gepa-metrics"

# ── eval ────────────────────────────────────────────────────────────────────
eval dataset func="answer_hotpot" corpus="eval-corpus" run="": build-eval
    {{preface}}run_tag='{{run}}'; run_tag="${run_tag#run=}"; extra=''; [[ -n "$run_tag" ]] && extra=" --tag run=$run_tag"; \
    {{kb_eval_bin}} run --dataset {{dataset}} --backend tensorzero --work {{corpus}} --tensorzero-function {{func}} \
      --judge-endpoint http://localhost:1234 --judge-model qwen3.5-4b $extra

eval-fixture: build-eval
    {{preface}}{{kb_eval_bin}} run --dataset eval/fixtures/sample-hotpot-distractor.json --backend mock

eval-metrics:
    @just ch "SELECT tr.value AS run, ta.value AS arm, countIf(f.metric_name='f1') AS n, round(avgIf(f.value, f.metric_name='f1'), 3) AS f1, round(avgIf(f.value, f.metric_name='recall_at_10'), 3) AS r10, round(avgIf(f.value, f.metric_name='judge'), 3) AS judge FROM tensorzero.FloatMetricFeedback f JOIN tensorzero.FloatMetricFeedbackTagView tr ON f.id = tr.feedback_id AND tr.key = 'run' LEFT JOIN tensorzero.FloatMetricFeedbackTagView ta ON f.id = ta.feedback_id AND ta.key = 'arm' WHERE f.metric_name IN ('f1','recall_at_10','judge','mrr','recall_at_5','recall_at_20') GROUP BY run, arm ORDER BY run DESC, arm FORMAT PrettyCompact"

# Wipe HotpotQA eval history in ClickHouse (answer_hotpot* inferences + episode metrics). Does not touch enrich/GEPA/coding.
eval-reset:
    @just ch "ALTER TABLE tensorzero.ModelInference DELETE WHERE inference_id IN (SELECT id FROM tensorzero.ChatInference WHERE function_name IN ('answer_hotpot', 'answer_hotpot_nograph'))"
    @just ch "ALTER TABLE tensorzero.InferenceTag DELETE WHERE inference_id IN (SELECT id FROM tensorzero.ChatInference WHERE function_name IN ('answer_hotpot', 'answer_hotpot_nograph'))"
    @just ch "ALTER TABLE tensorzero.ChatInference DELETE WHERE function_name IN ('answer_hotpot', 'answer_hotpot_nograph')"
    @just ch "ALTER TABLE tensorzero.FeedbackTag DELETE WHERE feedback_id IN (SELECT id FROM tensorzero.FloatMetricFeedback WHERE metric_name IN ('f1', 'recall_at_5', 'recall_at_10', 'recall_at_20', 'mrr', 'judge') UNION ALL SELECT id FROM tensorzero.BooleanMetricFeedback WHERE metric_name IN ('em', 'retrieved'))"
    @just ch "ALTER TABLE tensorzero.FloatMetricFeedback DELETE WHERE metric_name IN ('f1', 'recall_at_5', 'recall_at_10', 'recall_at_20', 'mrr', 'judge')"
    @just ch "ALTER TABLE tensorzero.BooleanMetricFeedback DELETE WHERE metric_name IN ('em', 'retrieved')"
    @echo "eval-reset: mutations queued — wait ~5s then: just eval-metrics"

# ── inspect ─────────────────────────────────────────────────────────────────
graph-stats: build-kb
    {{preface}}{{kb_bin}} graph stats {{work}}

ch sql:
    @curl -s "http://localhost:8123/?user=chuser&password=chpassword" --data "{{sql}}"

last-episode:
    @just ch "SELECT episode_id FROM tensorzero.ChatInference WHERE function_name='enrich' ORDER BY timestamp DESC LIMIT 1 FORMAT TabSeparatedRaw"
