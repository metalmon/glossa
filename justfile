# glossa dev pipeline — run `just <recipe>` (bare `just` lists all recipes).
# Install just: `cargo install just`  or  `winget install casey.just`.
# Recipes run under bash (git-bash on Windows). Cargo emits `kb` (GNU/WSL) or `kb.exe` (MSVC).
#
# IMPORTANT (Windows): `windows-shell` is pinned to git-bash's own absolute path — NOT the bare
# `bash` command. On a machine with WSL installed, plain `bash` on PATH often resolves to the WSL
# launcher (C:\Windows\system32\bash.exe) *ahead of* git-bash, since git-bash's install dir is
# normally not added to the Windows PATH at all. `just` itself always runs as a native Windows
# binary here (`os() == "windows"`), so recipes must run in a native-Windows shell too — otherwise
# `cargo build` silently runs inside WSL and produces a *Linux* ELF binary next to (but distinct
# from) the native `kb.exe` you actually invoke, and the two drift out of sync.
# `set windows-shell` only accepts a constant value (no path_exists() etc. — just rejects function
# calls there), so this is a plain hardcoded path. If your git-bash lives elsewhere (32-bit install,
# custom location, portable git, …), edit this one line to match.
set shell := ["bash", "-lc"]
set windows-shell := ["C:/Program Files/Git/bin/bash.exe", "-lc"]

preface := '[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"; [[ -d "$HOME/.cargo/bin" ]] && export PATH="$HOME/.cargo/bin:$PATH"; '
release := "--release"
bin     := "target/release"
kb_bin       := if os() == "windows" { "./" + bin + "/kb.exe" } else { "./" + bin + "/kb" }
kb_dump_episode_bin := if os() == "windows" { "./" + bin + "/kb-dump-episode.exe" } else { "./" + bin + "/kb-dump-episode" }
kbx_bin      := if os() == "windows" { "./" + bin + "/kbx.exe" } else { "./" + bin + "/kbx" }

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
# PowerShell: `just build-kb force`
build:
    {{preface}}cargo build --workspace {{release}} --locked
build-offline:
    {{preface}}cargo build --workspace {{release}} --locked --offline
build-kb force="":
    {{preface}}b='{{kb_bin}}'; if [[ -z "{{force}}" && "${FORCE_BUILD:-}" != "1" ]] && [[ -x "$b" ]]; then echo "kb: already built"; else cargo build {{release}} --bin kb --locked; fi
build-kbx force="":
    {{preface}}b='{{kbx_bin}}'; if [[ -z "{{force}}" && "${FORCE_BUILD:-}" != "1" ]] && [[ -x "$b" ]]; then echo "kbx: already built"; else cargo build {{release}} -p kb-eval --bin kbx --locked; fi

# Opt-in: builds `kb` WITH the CSP/marking-schema solver (glossa-constraint). The default
# `build-kb` is lean (no constraint_solve MCP tool) — use this only for constraint work.
build-kb-constraint force="":
    {{preface}}cargo build {{release}} --bin kb --locked --features constraint
test:
    {{preface}}cargo test --workspace {{release}} --locked
    {{preface}}cargo test -p kb-eval {{release}} --locked --features constraint
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

# The `verify` gate is fail-closed at runtime (hidden until a corpus enables+calibrates it), but the
# generated TZ catalog advertises the FULL tool set — so enable it here purely so `dump-tz-tools`
# emits verify's schema/description alongside the rest.
verify_gen_env := "GLOSSA_VERIFY_ENABLED=true GLOSSA_VERIFY_THRESHOLD_SINGLE=0.5 GLOSSA_VERIFY_THRESHOLD_MULTI=0.5"

tools: (build-kb "force")
    {{preface}}{{verify_gen_env}} {{kb_bin}} mcp dump-tz-tools --config-dir {{tzcfg}}
    @echo "regenerated — run 'just gw-restart' to load the new schemas"

# TZ gateway runs from the primary checkout (docker compose cwd); sync generated tool schemas there.
gw_root := env_var_or_default("GLOSSA_GW_ROOT", "../glossa")

tools-glossa: (build-kb "force")
    {{preface}}{{verify_gen_env}} {{kb_bin}} mcp dump-tz-tools --config-dir {{gw_root}}/eval/tensorzero/config
    @echo "regenerated {{gw_root}}/eval/tensorzero/config — run 'just gw-restart' from {{gw_root}}/eval/tensorzero"


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

constraint_out := "gepa-constraint-out"
sop_dir        := "eval/sops/example"              # SOP pack driving the constraint agent


gepa-constraint-metrics:
    @just ch "SELECT t.value AS run, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_c_baseline_discover'), 3) AS baseline_d, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_c_baseline_materialize'), 3) AS baseline_m, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_c_baseline_compile'), 3) AS baseline_c, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_c_baseline_coverage'), 3) AS baseline_cov, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_c_baseline_validate'), 3) AS baseline_v, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_c_combined_acc'), 3) AS final, round(avgIf(f.value, f.metric_name='gepa_c_iter_materialize'), 3) AS iter_m, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_c_final_discover'), 3) AS discover, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_c_final_materialize'), 3) AS materialize, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_c_final_compile'), 3) AS compile, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_c_final_coverage'), 3) AS coverage, round(argMaxIf(f.value, f.timestamp, f.metric_name='gepa_c_final_validate'), 3) AS validate FROM tensorzero.FloatMetricFeedback f JOIN tensorzero.FloatMetricFeedbackTagView t ON f.id = t.feedback_id AND t.key = 'run' WHERE f.metric_name IN ('gepa_c_baseline_discover','gepa_c_baseline_materialize','gepa_c_baseline_compile','gepa_c_baseline_coverage','gepa_c_baseline_validate','gepa_c_baseline_combined','gepa_c_iter_materialize','gepa_c_iter_combined','gepa_c_combined_acc','gepa_c_final_discover','gepa_c_final_materialize','gepa_c_final_compile','gepa_c_final_coverage','gepa_c_final_validate','gepa_c_final_combined','gepa_c_candidates') GROUP BY run ORDER BY run DESC LIMIT 20 FORMAT PrettyCompact"

gepa-constraint-reset:
    @just ch "ALTER TABLE tensorzero.ModelInference DELETE WHERE inference_id IN (SELECT id FROM tensorzero.ChatInference WHERE function_name IN ('cdiscover', 'cmaterialize', 'ccompile', 'ccoverage', 'cvalidate', 'cresearch', 'ccompile_fix'))"
    @just ch "ALTER TABLE tensorzero.InferenceTag DELETE WHERE inference_id IN (SELECT id FROM tensorzero.ChatInference WHERE function_name IN ('cdiscover', 'cmaterialize', 'ccompile', 'ccoverage', 'cvalidate', 'cresearch', 'ccompile_fix'))"
    @just ch "ALTER TABLE tensorzero.ChatInference DELETE WHERE function_name IN ('cdiscover', 'cmaterialize', 'ccompile', 'ccoverage', 'cvalidate', 'cresearch', 'ccompile_fix')"
    @just ch "ALTER TABLE tensorzero.FeedbackTag DELETE WHERE feedback_id IN (SELECT id FROM tensorzero.FloatMetricFeedback WHERE metric_name IN ('gepa_c_baseline_discover','gepa_c_baseline_materialize','gepa_c_baseline_compile','gepa_c_baseline_coverage','gepa_c_baseline_validate','gepa_c_baseline_combined','gepa_c_iter_materialize','gepa_c_iter_combined','gepa_c_combined_acc','gepa_c_final_discover','gepa_c_final_materialize','gepa_c_final_compile','gepa_c_final_coverage','gepa_c_final_validate','gepa_c_final_combined','gepa_c_candidates'))"
    @just ch "ALTER TABLE tensorzero.FloatMetricFeedback DELETE WHERE metric_name IN ('gepa_c_baseline_discover','gepa_c_baseline_materialize','gepa_c_baseline_compile','gepa_c_baseline_coverage','gepa_c_baseline_validate','gepa_c_baseline_combined','gepa_c_iter_materialize','gepa_c_iter_combined','gepa_c_combined_acc','gepa_c_final_discover','gepa_c_final_materialize','gepa_c_final_compile','gepa_c_final_coverage','gepa_c_final_validate','gepa_c_final_combined','gepa_c_candidates')"
    @echo "gepa-constraint-reset: mutations queued — wait ~5s then: just gepa-constraint-metrics"


# ── eval ────────────────────────────────────────────────────────────────────


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

# Average constraint-eval metrics grouped by the `run` tag ('-' = untagged runs).
constraint-metrics:
    @just ch "SELECT coalesce(tr.value, '-') AS run, countIf(f.metric_name='field_coverage') AS n, round(avgIf(f.value, f.metric_name='field_coverage'), 3) AS fields, round(avgIf(f.value, f.metric_name='constraint_coverage'), 3) AS constraints, round(avgIf(f.value, f.metric_name='literal_coverage'), 3) AS literals, round(avgIf(f.value, f.metric_name='tools_used'), 1) AS rounds, round(avgIf(f.value, f.metric_name='agent_graph_node_count'), 1) AS nodes, round(avgIf(f.value, f.metric_name='agent_graph_edge_count'), 1) AS edges FROM tensorzero.FloatMetricFeedback f LEFT JOIN tensorzero.FloatMetricFeedbackTagView tr ON f.id = tr.feedback_id AND tr.key = 'run' WHERE f.metric_name IN ('field_coverage','constraint_coverage','literal_coverage','tools_used','agent_graph_node_count','agent_graph_edge_count') GROUP BY run ORDER BY run DESC FORMAT PrettyCompact"
    @just ch "SELECT coalesce(tr.value, '-') AS run, countIf(f.metric_name='csp_agreement') AS n, round(avgIf(toFloat64(f.value), f.metric_name='csp_agreement'), 3) AS csp_agree, round(avgIf(toFloat64(f.value), f.metric_name='llm_correct'), 3) AS llm_correct FROM tensorzero.BooleanMetricFeedback f LEFT JOIN tensorzero.BooleanMetricFeedbackTagView tr ON f.id = tr.feedback_id AND tr.key = 'run' WHERE f.metric_name IN ('csp_agreement','llm_correct') GROUP BY run ORDER BY run DESC FORMAT PrettyCompact"
    @just ch "SELECT coalesce(tr.value, '-') AS run, countIf(f.metric_name='ref_param_coverage') AS n, round(avgIf(f.value, f.metric_name='ref_param_coverage'), 3) AS params, round(avgIf(f.value, f.metric_name='ref_value_coverage'), 3) AS vals, round(avgIf(f.value, f.metric_name='table_csp_count'), 1) AS csp FROM tensorzero.FloatMetricFeedback f LEFT JOIN tensorzero.FloatMetricFeedbackTagView tr ON f.id = tr.feedback_id AND tr.key = 'run' WHERE f.metric_name IN ('ref_param_coverage','ref_value_coverage','table_csp_count') GROUP BY run ORDER BY run DESC FORMAT PrettyCompact"

# Reset constraint-eval METRICS in ClickHouse (episode feedback only). Keeps the
# constraint_validate inference transcripts so a grind can be investigated afterwards.
# Does not touch enrich/GEPA/HotpotQA/coding.
constraint-reset:
    @just ch "ALTER TABLE tensorzero.FeedbackTag DELETE WHERE feedback_id IN (SELECT id FROM tensorzero.FloatMetricFeedback WHERE metric_name IN ('field_coverage', 'constraint_coverage', 'literal_coverage', 'tools_used', 'agent_graph_node_count', 'agent_graph_edge_count', 'ref_param_coverage', 'ref_value_coverage', 'table_csp_count') UNION ALL SELECT id FROM tensorzero.BooleanMetricFeedback WHERE metric_name IN ('csp_agreement', 'llm_correct'))"
    @just ch "ALTER TABLE tensorzero.FloatMetricFeedback DELETE WHERE metric_name IN ('field_coverage', 'constraint_coverage', 'literal_coverage', 'tools_used', 'agent_graph_node_count', 'agent_graph_edge_count', 'ref_param_coverage', 'ref_value_coverage', 'table_csp_count')"
    @just ch "ALTER TABLE tensorzero.BooleanMetricFeedback DELETE WHERE metric_name IN ('csp_agreement', 'llm_correct')"
    @echo "constraint-reset: mutations queued — wait ~5s then: just constraint-metrics"

# Readable transcript of a constraint_validate episode (ClickHouse → text file).
# Examples: just constraint-dump run=deploy-35b
#           just constraint-dump episode=019f343a-7c0c-7b33-ad14-75bf26c50cc0
constraint-dump episode="" run="" latest="false" out="eval/results/episode-dump.txt": build-dump-episode
    {{preface}}args=''; \
    [[ -n "{{episode}}" ]] && args="$args --episode-id {{episode}}"; \
    [[ -n "{{run}}" ]] && args="$args --run {{run}}"; \
    [[ "{{latest}}" == "true" ]] && args="$args --latest"; \
    {{kb_dump_episode_bin}} $args --out {{out}}; \
    echo "wrote {{out}}"


build-dump-episode force="":
    {{preface}}b='{{kb_dump_episode_bin}}'; if [[ -z "{{force}}" && "${FORCE_BUILD:-}" != "1" ]] && [[ -x "$b" ]]; then echo "kb-dump-episode: already built"; else cargo build {{release}} -p kb-eval --bin kb-dump-episode --locked; fi

# ── inspect ─────────────────────────────────────────────────────────────────
graph-stats: build-kb
    {{preface}}{{kb_bin}} graph stats {{work}}

ch sql:
    @curl -s "http://localhost:8123/?user=chuser&password=chpassword" --data "{{sql}}"

last-episode:
    @just ch "SELECT episode_id FROM tensorzero.ChatInference WHERE function_name='enrich' ORDER BY timestamp DESC LIMIT 1 FORMAT TabSeparatedRaw"

# ── measure ─────────────────────────────────────────────────────────────────
# Extraction baseline — phase 1 of docs/superpowers/specs/2026-11-10-omnivorous-extraction-design.md.
#
# The MEASURED `kb` is passed in and never built here: a baseline taken on a local build measures a
# binary nobody ships. Get one from CI — `gh workflow run release.yml --ref <branch>`, then
# `gh run download` — and pass its path. `kbx` itself is a tool, not the subject, so it builds
# locally like every other tool in this file.
#
# Corpora are measured as COPIES. A `--force` pass rebuilds a corpus's index, and on a corpus whose
# manifest predates the current schema it migrates it too — a corpus an eval baseline rests on would
# move underneath that baseline if measured in place.
bench_dir := env_var_or_default("GLOSSA_BENCH_DIR", "../glossa-bench")

# Copy one corpus for measuring (index and graph come along: the point is to measure a real one).
bench-copy src dst:
    {{preface}}mkdir -p '{{bench_dir}}'; rm -rf '{{bench_dir}}/{{dst}}'; cp -r '{{src}}' '{{bench_dir}}/{{dst}}'; echo "copied {{src}} -> {{bench_dir}}/{{dst}}"

# Measure one corpus. `kb` is the path to the CI-built binary.
bench-extract kb corpus *extra: build-kbx
    {{preface}}{{kbx_bin}} bench extract '{{corpus}}' --bin '{{kb}}' --out '{{corpus}}.bench.json' {{extra}}

# The whole baseline: two real corpora as copies, plus a generated corpus with the size ladder.
# Corpus directories are arguments, not constants — a corpus name says who the corpus belongs to.
# Rows go to stdout (the column header goes to stderr), so append them where you want them.
bench-baseline kb corpus_a corpus_b: build-kbx
    {{preface}}set -e;       just bench-copy '{{corpus_a}}' a-copy;       just bench-copy '{{corpus_b}}' b-copy;       rm -rf '{{bench_dir}}/synth';       just bench-extract '{{kb}}' '{{bench_dir}}/a-copy';       just bench-extract '{{kb}}' '{{bench_dir}}/b-copy';       just bench-extract '{{kb}}' '{{bench_dir}}/synth' --generate 2000 --seed 1 --ladder
