# Fullwiki retrieval eval — design

**Status:** approved (brainstorm 2026-06-25)
**Goal:** measure glossa's retrieval at Wikipedia scale — Recall@k of the gold supporting-fact articles
over the full ~5M-article HotpotQA corpus — the discriminating engine metric that the distractor
setting can't provide (distractor recall saturates at ~1.0 over only 10 candidate docs).

## Why this matters / what's different from distractor

Distractor mode ships 10 paragraphs per question (2 gold + 8 distractors); the harness writes those 10
as files and indexes them per question, so retrieval is trivial (recall ~1.0). Fullwiki retrieves the
gold articles from one **shared** index built over all ~5M Wikipedia intro paragraphs — ranking,
Recall@k, and the multi-hop bridge become the real levers.

## Corpus

The **fullwiki intro-paragraph** archive (first intro paragraph per article, ~5M articles, **1.55 GB**
bz2; CC BY-SA 4.0). Canonical, comparable to leaderboard fullwiki. Sources: Stanford NLP
(`nlp.stanford.edu/projects/hotpotqa/…-withlinks-processed.tar.bz2`) and HF mirrors
(`ParthMandaliya/hotpotqa-wiki`). NOT the 7.4 GB full-text version. Each record is JSON with
`{id, title, text: [sentences], …}`.

## Eval arm — Qwen only (deliberate)

The fullwiki run uses **only** the `openai`/Qwen-3.5-4B backend (harness-side function-calling loop,
glossa `search`/`read` executed **in-process**). Rationale: this is a **retrieval** measurement, so the
agent must depend on glossa's tools and nothing else. The `cli`/`claude -p` arm is excluded here — we
can't guarantee it answers only from glossa tools (it may use parametric knowledge), which would
inflate EM while saying nothing about retrieval. The 4B is weak enough to genuinely depend on
retrieval, and the harness controls its tool loop end-to-end (glossa-only). Claude's reader ceiling is
already established on distractor (EM 0.80) — no need to repeat it here.

## Components

### 1. Corpus prep — `kb-eval prep-fullwiki`

`kb-eval prep-fullwiki --archive <path-to.tar.bz2> --out <wiki-corpus-dir>`:
- Decompress + walk the tar.bz2 (the `eval` crate is dev-only, so C-backed crates like `bzip2`/`tar`
  are fine — the C-free invariant binds only `glossa`).
- For each shard file, write **one markdown file per shard** (`<out>/AA/wiki_00.md`, …) with each
  article rendered as `# <title>\n<intro sentences joined by space/newline>\n`.
- Result: ~tens of thousands of md files holding ~5M article-sections. (Writing 5M individual files
  would kill the filesystem and the walk; shard-grouping keeps file count sane while reusing the real
  glossa extract→index pipeline we are benchmarking — each article becomes a chunk whose `location`
  is its title.)
- Then the operator (or a follow-up step) runs `kb index <wiki-corpus-dir>` once to build the shared
  tantivy index (`<out>/.glossa/index`).

### 2. Harness fullwiki mode — `kb-eval run --fullwiki <wiki-corpus-dir>`

When `--fullwiki` is set:
- Skip per-question `corpus::write_corpus` + `corpus::index` + the `.glossa` clear. The corpus is
  pre-built and shared/read-only.
- Backends receive `work = <wiki-corpus-dir>`, so the in-process `DocIndex::open_or_create(work)`
  opens the **shared** pre-built index; nothing is cleared between questions.
- Recall is still measured from the per-question trace time-window (`read_window [t0,t1]`) — the shared
  `work/.glossa/traces` accumulates across questions, but timestamps isolate each question (runs are
  sequential). Search over 5M docs is tantivy BM25 (milliseconds); the "search is slow" ROADMAP note
  is about the default ripgrep *scan*, not the index path the eval uses.
- `--limit` samples the dev set (default first run: 200 questions).

### 3. Recall@k scoring

New pure scorer over the captured transcript (already in `Row.transcript`):
- For each question, walk its `search` tool calls; each result carries ranked hits with `location`
  (= article title) and `score`. Build the ranked list of distinct titles surfaced (dedup by first
  occurrence / best rank across the question's searches).
- A gold supporting **title** is retrieved@k if it appears within the top-k of that merged ranked
  list. Report **Recall@5, Recall@10, Recall@20** (fraction of a question's gold titles found within
  top-k, averaged over questions) and **MRR** (1/rank of the first gold title found).
- Title match: normalized exact match (`score::normalize`-style: lowercase, trim, collapse spaces)
  between gold title and hit `location`.
- Keep EM/F1 too (answer quality), but Recall@k is the headline fullwiki number.

## Components / files

- `eval/src/prep.rs` (new) — `prep_fullwiki(archive, out)`: bz2/tar walk + JSON-lines → md-shard writer.
- `eval/src/main.rs` — add the `PrepFullwiki` subcommand and the `--fullwiki <dir>` flag on `Run`.
- `eval/src/run.rs` — `run_eval` honors fullwiki mode (skip per-question corpus build; `work` = corpus);
  compute and include Recall@k/MRR per row and in the report aggregate.
- `eval/src/score.rs` — `recall_at_k(ranked_titles, gold_titles, k) -> f32` and `mrr(ranked_titles, gold_titles) -> f32` (pure, unit-tested).
- `eval/Cargo.toml` — add `bzip2` + `tar` (dev crate; C allowed).
- `.gitignore` — add `/wiki-corpus/` and the downloaded archive (large generated artifacts).

## Feasibility spike (first plan task, before the full build)

Indexing 5M docs stress-tests glossa's indexer (a known perf concern). First: run `prep-fullwiki` on a
**single shard** (or a small `--max-shards N`), `kb index` it, and measure wall-time + index size;
extrapolate to 5M. If it projects to many hours, decide (with the operator) whether to parallelize the
indexer first (ROADMAP item, blocked on the PdfExtractor panic-hook race — but markdown-only corpus
sidesteps PDFs) or accept the one-time cost. Gate the full build on this number.

## Error handling

- Missing/corrupt archive or shard: log and skip the shard, continue (never abort the whole prep).
- A question whose gold titles never appear in any search: Recall@k = 0 for it (correct).
- Empty transcript (model issued no search): Recall@k = 0, EM/F1 still scored.

## Testing (TDD)

- `prep`: a tiny synthetic tar.bz2 (or a JSON-lines fixture) → asserts md-shard output has `# Title`
  sections with the intro text; multiple articles per shard preserved.
- `score::recall_at_k`: ranked titles `[A,B,C,D]`, gold `[C,E]`, k=2 → 0.0 (C at rank 3); k=3 → 0.5;
  `mrr` → 1/3. Normalization (case/space) covered.
- `run` fullwiki wiring: a mock backend over a tiny pre-built corpus → asserts no per-question clear,
  shared index used, Recall@k computed. Deterministic, no model/network (the mock_e2e gate extended).

## Backlog (deferred)

Full 7,405-dev run (canonical leaderboard number); graph-on/off A/B over fullwiki (multihop claim);
parallel indexer for faster builds; `text_with_links` (use hyperlinks for a graph layer); dense-retrieval
baseline for context. Claude/cli arm on fullwiki only if a tool-use audit confirms glossa-only behavior.
