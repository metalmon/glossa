# glossa — roadmap & backlog

Status as of 2026-06-24: Milestones 1–5 + real `graph_upsert` are merged to `master`.
Pure-Rust single offline binary (`kb`); ~54 tests green; no C compiled on shipping targets.

## What works today
- Extraction: md, docx/doc/xlsx/xls/pptx/ppt (office_oxide), pdf (oxidize-pdf **lenient/xref-recovery**, per-page chunks `p.N`, scans indexed by filename).
- Search: ripgrep-syntax scan (default) **and** BM25 ranked (`--rank`) with RU/EN stemming.
- Knowledge graph (redb): provenance-stamped nodes/edges, `ontology.toml` validation, bounded
  traversal (`neighbors`/`path`), deterministic auto layer-1 (Document/Section/CONTAINS) during index.
- MCP server (`kb mcp --profile reader|editor|full`): `search`, `read` (text + embedded images as
  vision content), `glossary`, `neighbors`, `index`, `reindex`, `resolve`, `graph_upsert` (validated,
  provenance-stamped), `purge`. gitignore-aware indexing.

## Performance (search/index speed)
- **Search is slow on large bases.** The default scan re-walks the whole tree and regex-scans every
  file on *each* query (no persistent state); `index`/PDF extraction is also heavy. Optimize the
  walker + per-file IO: parallel traversal (rayon), fewer syscalls, mmap/streaming reads, skip-by-size,
  and consider a persistent file list. **Reference for ideas:** `fff` — https://github.com/dmtrKovalenko/fff
  (blazingly-fast parallel file finder in Rust) — mine its traversal/IO approach.
  - **Blocker for parallel extraction:** `PdfExtractor` uses the process-global panic hook
    (`take_hook`/`set_hook`) to silence pdf backtraces; that races if `extract` runs on multiple
    threads. Before parallelizing indexing, drop the hook swap (rely on `catch_unwind` alone) or guard
    it. Correctness is unaffected today (single-threaded).

## Technical backlog (carry-forward, non-blocking)
- **Graph crash-atomicity**: fold one file's node/edge writes into a single redb write txn so a
  mid-file crash can't leave a partial graph the manifest then skips as "unchanged". (`reindex` recovers today.)
- **`--expand`**: glossary query expansion — needs the layer-2 `Term`/co-occurrence layer (not built).
- **HTTP/streamable transport** for the MCP server — stdio only today.
- **Image-only PDF pages (scans).** PDFs are chunked per page (`p.N`); pages with no text layer are
  skipped, and a PDF with *no* extractable text is now indexed **by filename** (location `(no-text)`)
  so it is never dropped and is findable by name. Remaining work: extract/render the page *image* so
  the connected agent can *vision-read* it (like `read` already returns embedded office images as
  vision content). Pure-Rust offline OCR is hard (tesseract is C), so bet on vision-read, not OCR.
- **Indexing progress UX**: show per-file progress on slow/large bases (in flight).
- **PDF robustness**: `pdf-extract` can *panic* on malformed PDFs — must be caught so indexing never aborts (in flight).
- `type_of` in `upsert` swallows `get_node` errors via `.ok()` (fail-closed) — propagate.
- `read_region` returns empty string for unknown extensions silently; `join("\n")` may double newlines.
- Bounded-traversal depth doc-comments; secondary indexes for `resolve`/`delete_by_source` (O(n) today).
- Cosmetic: `Mcp` enum-variant vs match-arm ordering.

## Product roadmap

The work splits into **two distinct tracks**. They share one model-agnostic agent-eval harness, but
the "traces → patterns → domain skills" loop belongs ONLY to Track B.

### Stage 1 — local bring-up (IN PROGRESS)
- Build the release binary; connect glossa as an MCP server to the controlling agent (Claude Code).
- Smoke-test on a real knowledge base (`kb-test/`, git-ignored): `index` → `search`/`--rank` → `graph`.
- Operator runs CLI commands in the terminal.

### Track A — public benchmarks (engine positioning)
Goal: measure where the **engine** sits vs the field. NOT a domain — no skill loop here.
- Build a **model-agnostic agent-eval harness**: an MCP *client* that drives a model, lets it call
  glossa tools, and scores. Swap the model backend: the smart controller (Claude) vs a weak local
  model (Qwen3.5-4B via LM Studio's OpenAI-compatible API).
- Datasets + scoring come WITH the public benchmark — use their gold + **their official metric**
  (Exact Match / F1 against the gold answer string). Do NOT use our LLM-judge here.
- Gold **supporting-fact spans are provided** by multihop benches → retrieval/groundedness measurable
  out of the box.
- Benches: multihop reasoning + exact-fact (HotpotQA, 2WikiMultihopQA, MuSiQue).
- **Multihop A/B (the key claim):** same questions / same model, graph OFF vs graph ON
  (no-saturation, code layers 1–2 only → after the agent saturates layers 3–4 via `graph_upsert`).
  Needs a "graph-off" mode for a clean control.
- Honest caveat: these benches are English-Wikipedia, clean text — they do NOT exercise our
  differentiators (office/pdf, Russian, offline, agentic graph) and on paraphrase recall will favor
  dense-embedding RAG (we are lexical+graph). They validate the **multihop/retrieval engine**, not
  the product edge. Expect to trail on pure-semantic benches; our bet is multihop+graph+offline.
- Traces here are for **debugging failures only** — not pattern-mining.

### Track B — domain refinement (the product moat)
Goal: tune glossa to a real domain via your own knowledge base. THIS is where the loop lives.
- **Curate a domain Q/A set** (laborious, first-class deliverable): question → gold answer → **gold
  source spans** (which file + location). *Bootstrap:* the smart agent answers each question against
  the base and records the sources it used → draft gold-spans; human verifies. Include **negative /
  no-answer cases** to measure false-positive / hallucination rate.
- **Scoring split:** retrieval metrics (was the right source surfaced?) via **regex/span-match**
  (works well here) — Recall@k, MRR; answer correctness (free-form) via **LLM-as-judge**; plus
  **groundedness/citation accuracy** (every answer traces to a real span — our differentiator).
- **Tracing is external, not in the server:** the eval harness is the MCP *client* mediating
  model↔glossa, so it already sees every tool call + result (search results carry
  path:location:line:snippet+score). Log the trace in the harness; keep the server clean. **Reverse-trace:**
  given a question's gold spans, which queries/hops reach them (optimal-path oracle for diagnosing failures).
- **Heatmap = static PNG** (pure-Rust plotter, e.g. `plotters`, C-free; open/share anywhere): per
  question × per step hit/miss, so the operator sees, visually, how retrieval is doing.
- **Patterns → domain skills:** start trace-assisted **manual** skill authoring (operator + agent read
  traces); auto pattern-mining is a later nicety, not promised now.
- **Skill packaging:** a **base ontology + base skill**, refined per domain (legal/medical/eng/…).
- **CI regression:** once the harness exists, run it on a fixed mini-set so quality can't silently regress.
- Possible small product adds for this track: a `list` tool (enumerate indexed docs, for systematic
  saturation), a "graph-off" query mode (for the A/B), richer tool-result metadata.

### Ordering
1. Finish Stage 1 (bring-up) — in flight.
2. Build the shared agent-eval harness; run **Track A** (engine numbers, Claude then Qwen-4B).
3. In parallel/after: **Track B** — curate domain Q/A (+ bootstrap gold-spans), external traces,
   PNG heatmap, patterns → domain skills.

## Notes
- Everything stays pure Rust / offline / single binary. The graph is a disposable overlay over files
  (File-First) — deleting `.glossa/` loses nothing; it rebuilds from the index.
- Profiles are tool-visibility gating, not RBAC: `reader` (query), `editor` (+ build/index), `full` (+ admin).
