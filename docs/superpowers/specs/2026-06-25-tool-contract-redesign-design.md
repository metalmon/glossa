# glossa Tool-Contract Redesign — read-by-number + dual-mode search (design)

- **Date:** 2026-06-25
- **Status:** Draft (awaiting review)
- **Scope:** the agent-facing MCP/CLI tool contract (`search`, `read`, new `grep`) + the index
  change they require, mirrored in the `kb-eval` harness so the eval measures the prod contract.
- **Research basis:** [`glossa-grep-trigram-research.md`](../plans/glossa-grep-trigram-research.md)
  (ugrep, ripgrep, Russ Cox trigram method, Zoekt/livegrep, fzf).

---

## 1. Problem & motivation

Observed during real-domain eval runs:

1. **The model mis-addresses `read`.** Search shows `…pdf:p.21: snippet`; the model calls
   `read(path, location="страница 21")` — it paraphrases the location token. `read` then misses
   and falls through to opening the source file, which (mid-reindex) was absent → `os error 2`.
   Root cause: `location` is a **free-form string** the model is free to reword.
2. **The `search` contract contradicts itself.** It is advertised three ways in `src/mcp.rs`:
   the param schema says `"ripgrep-syntax query"` (mcp.rs:72), the server instructions say
   `"Use ripgrep syntax for search"` (mcp.rs:202), but the tool description says
   `"natural-language keywords … NOT a regex … BM25-ranked"` (mcp.rs:114). The implementation is
   BM25; the "ripgrep" labels are stale and **lie**. The model is told two opposite things.

### Design principles (binding)
- **Self-evident tools.** The contract must be clear from the tool/param schema alone — never
  propped up by prompt text. A weak model must get it right with no instructions.
- **One interpretation.** No parameter should admit a second reasonable reading.
- **File-First (HARD, per the v1 spec §12.2).** Files are the source of truth; the index is a
  disposable accelerator/pointer. The agent contract must not make the index authoritative.
- **Two query styles, model chooses.** Exact/regex (`grep`) and ranked keywords (`search`) are
  separate tools; each model uses whichever suits it. Tool *names* document the style (`grep` ⇒
  ripgrep semantics every model already knows; `search` ⇒ keyword/BM25). A mode-parameter is
  rejected — it reintroduces "read the description and guess."

## 2. Goals / non-goals

**Goals**
- `read(path, n)` where `n` is a **typed integer** chunk number — un-mis-formattable.
- Honest `search` (BM25 keywords); drop every "ripgrep" claim.
- New `grep` tool: ripgrep literal/regex, exact, File-First. **v1 needs no new index.**
- **Trigram accelerator as step 2** (after step 1 is debugged), per the research.
- Mirror the contract in `kb-eval` so the eval measures the prod agent contract.
- Stay **C-free** (pure-Rust single binary).

**Non-goals**
- Semantic/vector search (unchanged; out of scope).
- Positional trigrams (Zoekt) / suffix array (livegrep) — only if profiling demands.
- Fuzzy ranking — optional step 3 via pure-Rust `nucleo`, off the correctness path.

## 3. The contract (three tools)

### 3.1 `read(path, n: integer)` — read a chunk by number
- `n` = the chunk's **single canonical number within the document**: the **page number** for
  PDFs (so `n` equals the `p.N` the result already shows), the **1-based section index** for
  headings, the **1-based sheet index** for spreadsheets. Exactly one number per chunk — never a
  synthetic ordinal competing with a page number. It is an **integer in the JSON schema**, so the
  model *cannot* send `"страница 21"` — the protocol forces a number. Self-evident by type, not
  prompt.
- Returns the chunk's **stored body from the index** (the File-First accelerator already stores
  chunk text, v1 spec §4.3) — so `read` works even when source files are mid-reindex/absent.
  When the source file is present and `include_images` is set, also returns its image blocks
  (unchanged behavior) by opening the file at that chunk.
- **Context expansion / "next chunk":** the result footer states the chunk's position and that
  the neighbors are `n-1` / `n+1` (omitted at document edges, mirroring the structural
  `NEXT`/`PREV` edges, v1 spec §11 Layer 1). To widen context the model calls `read(path, n+1)`.
- **Robustness fallback:** if a client still sends a string, strip to digits (`"p.21"`→`21`).
  Belt-and-suspenders only; the integer schema is the real guarantee.
- The legacy `read_region(path, location)` stays as the **CLI/human** surface (`kb read <path>`)
  for direct file reads; it is no longer the agent contract.

### 3.2 `search(query)` — BM25 keywords (honest)
- Description, param schema, and server instructions all say the truth: **natural-language
  keywords, morphology-aware, BM25-ranked.** Remove every "ripgrep syntax" string.
- Output lines carry exactly one read key: `[#n] <path> · <label> · <snippet>  (score)`, where
  `n` is the chunk's canonical number (§3.1) and `<label>` is a **non-numeric** hint only — the
  heading text, or `pdf`/`sheet` — so it never competes with `#n`. `read(path, n)` is the
  obvious, un-guessable follow-up.

### 3.3 `grep(pattern, …flags)` — NEW: ripgrep literal/regex, File-First
- Exact/regex search via the **`regex` crate (the same engine ripgrep uses)** — `grep` is
  self-documenting to any model. Output `…:#n: matched line` (same `[#n]` read key).
- **v1 = no new index** (research §2.A): parse `pattern` → HIR; extract required literals with
  `regex_syntax::hir::literal::Extractor`; **metadata-prefilter** candidate chunks by `path`/
  `file_type` (`-g`/`-t`); **BM25 word-token prefilter** over the existing tantivy `body` *only
  when extracted literals are whole tokens* (else **full-scan the stored bodies**); **confirm**
  with the compiled `regex` over each candidate's stored `body`. File re-read only for
  out-of-chunk `-A/-B/-C` context (File-First).
- **Flags (ripgrep parity):** `-i` smart-case, `-F` fixed-string, `-w` word, `-A/-B/-C` context,
  `-g/--glob`, `-t/--type`.
- Worst case is a scan of in-index stored bodies (fast); best case BM25 prefilter slashes
  candidates. Correct and File-First on day one; the accelerator (step 2) is purely optional.

## 4. Index change required by `read(path, n)` and `search`

Add a **per-document canonical chunk number** `ord` (the `n` of §3.1):
- During the streaming index build (`index_dir`), chunks arrive in document order. Set `ord` =
  the **page number** for PDF chunks (parsed from the existing `p.N` location), and the **1-based
  section/sheet index** for other formats. Store it as a tantivy fast/stored field.
- `read(path, n)` resolves to the chunk with `(path, ord = n)` via the index (extends the
  existing `read_chunk` term lookup, keyed on `ord` instead of a `location` string).
- The human `location` (`p.350` / heading) stays available internally; the agent only ever sees
  and types `ord`, and for PDFs `ord` **is** the page number, so the two never disagree.
- **Stability:** `ord` is stable for a static (eval) index; an incremental rechunk of a non-paged
  document can renumber its sections — documented, acceptable (the index is disposable, §12.2).
  PDF `ord` = page number is stable across reindex.

## 5. Architecture & data flow

```
index_dir(streaming) ── assign ord per doc ──▶ tantivy {path, location, file_type, body, ord}
                                                   │
agent ──MCP──▶ search(query)  ── BM25 ──▶ [#n] path · location · snippet
        ──MCP──▶ read(path,n)  ── term (path, ord=n) ──▶ stored body + (n-1/n+1)
        ──MCP──▶ grep(pattern) ── regex_syntax literals → prefilter → regex confirm ──▶ path:#n:line
```

- **Product:** `src/mcp.rs` (3 tool descriptions + `read`/`search`/new `grep`), `src/read.rs`
  (read-by-ordinal from index; `read_region` stays for CLI), `src/index/store.rs` (`ord` field +
  resolve-by-ordinal), `src/main.rs` (CLI `kb grep`).
- **Eval mirror:** `eval/src/backend/glossa_tools.rs` (numbered `search`, `read(path,n)`, `grep`),
  `eval/tensorzero/config/answer_hotpot/system.minijinja` (describe the 3 tools minimally — the
  schemas carry the contract), `eval/tensorzero/config/tools/*.json` (schemas).

## 6. Phasing

- **Step 1 — debug first (this spec's core):**
  `ord` field in the index; `read(path, n)` typed-integer (+ digit-strip fallback); honest
  numbered `search`; `grep` **v1 (no new index)**. Mirror in the eval. Validate on `kb-test`:
  `read` never mis-addresses; `grep` finds exact Cyrillic tokens / codes (`maxTsdr`, part
  numbers) that BM25 morphology blurs.
- **Step 2 — trigram accelerator (right after step 1, per request):**
  Cox regex→`TrigramQuery` translator (`enum {Any, Trigram, And, Or}`, `Any`→full-scan
  fallback); **byte-trigrams over UTF-8** (load-bearing for short Cyrillic terms — research §2.D);
  posting store **starts as a tantivy `NgramTokenizer` field**, escalates to **`redb` + `roaring`
  byte-trigram postings** only if byte-level Cyrillic selectivity needs it; **ugrep monotonic
  rule** — the accelerator may only *skip*; any newer/unindexed chunk is always scanned, never a
  missed match.
- **Step 3 — optional:** fuzzy result ranking via `nucleo` (off the grep correctness path).

## 7. C-free invariant

Every crate is pure-Rust: `regex`, `regex-syntax`, `aho-corasick`, `memchr`, `tantivy`
(existing), and step-2 `redb`, `roaring`, optional `ignore`/`grep-searcher`/`nucleo`.
`cargo tree -p glossa -i cc` must stay empty.

## 8. Errors & File-First invariants

- `read` serves from the disposable index; deleting the index loses nothing (rebuild from files).
- `grep` accelerator (step 2) is **skip-only**: staleness degrades to full scan, never to a
  missed match (ugrep property).
- Un-accelerable patterns (`.`, `.*`, `\d+`, short alternations, anchors-only, huge classes)
  detect → full scan. The accelerator is always optional; the scan is always correct.

## 9. Testing

- **read:** `read(path, n)` returns the n-th chunk; `n+1` is the next; out-of-range → clear
  error; integer schema rejects strings; digit-strip fallback maps `"p.21"`→21.
- **search:** numbered `[#n]` output; **no** "ripgrep"/"regex" claim anywhere in its contract.
- **grep v1:** literal + regex find an exact **Cyrillic token and a code** that BM25 misses;
  `-i/-F/-w/-A/-B/-C/-g/-t` behave; un-accelerable pattern falls back to full scan and still
  matches; output is `path:#n:line`.
- **index:** `ord` assigned 1-based per document, contiguous, resets per document.
- **C-free:** `cargo tree -p glossa -i cc` empty.
- **eval:** mock-backed test that the numbered contract round-trips (search #n → read n).

## 10. Open questions

- Non-paged `ord` = section/sheet **index** means search shows `#3` for the 3rd section while the
  heading text is the label. Acceptable, or should non-paged formats keep a different key? (PDFs —
  all of kb-test — are unaffected: `ord` = page number.)
- Expose `grep` in the eval immediately (to measure its retrieval lift on exact tokens) or after
  the `read`/`search` fix is validated alone.
