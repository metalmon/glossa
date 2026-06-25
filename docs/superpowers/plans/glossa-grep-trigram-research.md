# glossa `grep` mode — research synthesis (literal/regex search + trigram acceleration)

Research basis for a pure-Rust, C-free, offline, File-First grep tool alongside the existing tantivy BM25 `search`. Date: 2026-06.

## Sources (cited)
- Russ Cox, *Regular Expression Matching with a Trigram Index* — https://swtch.com/~rsc/regexp/regexp4.html (canonical regex→trigram method; `csearch`/`cindex`: https://github.com/google/codesearch)
- BurntSushi (Andrew Gallant), *ripgrep is faster than {grep, ...}* — https://blog.burntsushi.net/ripgrep/
- ugrep-indexer — https://github.com/Genivia/ugrep-indexer (now folded into ugrep 6.0+, `ugrep --index`)
- Zoekt design (positional trigrams) — https://github.com/sourcegraph/zoekt/blob/main/doc/design.md
- livegrep (suffix-array regex search) — https://github.com/livegrep/livegrep ; blog https://blog.nelhage.com/2015/02/regular-expression-search-with-suffix-arrays/
- fzf — https://github.com/junegunn/fzf
- `regex_syntax::hir::literal::Extractor` — https://docs.rs/regex-syntax/latest/regex_syntax/hir/literal/struct.Extractor.html

---

## 1. Core technique per source

### ugrep-indexer ("index accelerates, files are truth")
- **Granularity = per-file.** Writes a hidden `._UG#_Store` index file *into each directory* (one entry per file). Files are never modified; they remain the source of truth.
- **What it stores:** NOT posting lists. Per file it stores a set of **n-gram Bloom-filter hash tables** (bit tiers for 1-,2-,3-grams; separate bit space per n so 3-grams never collide with 2-grams → no cross-n false positives). Effectively a compact "which n-grams *can* occur in this file" sketch.
- **Accuracy knob `-0`..`-9`** (default `-4`/`-5`): trades index size vs false-positive rate. E.g. `-0` ≈ 490 B/file, default ≈ 4256 B/file. "Noise" = entropy measure; high-entropy files index poorly.
- **Query path:** regex pattern → derive required n-grams → for each file, test its Bloom filter; if the filter says "cannot contain required grams," **skip the file entirely**; else fall back to real grep on the file. Reported: e.g. "Skipped 1301 of 1317 files with non-matching indexes," 12x speedup, 15 false positives.
- **Monotonic / File-First safety:** search compares file/dir mtimes to the index timestamp; **any file newer than the index is always searched** (never wrongly skipped). This is the key correctness property glossa must replicate: *the index may only allow skipping; staleness must degrade to full scan, never to a missed match.*
- **Permissive patterns are costly:** big Unicode classes and unbounded `*`/`+` blow up the gram set → high start-up cost and low selectivity.

### ripgrep (fast, NO index)
Reusable pure-Rust building blocks (all C-free, already in glossa's likely dep set):
- **`regex`** crate — finite-automata engine, linear time, full Unicode, no catastrophic backtracking. The real-match confirmer.
- **`aho-corasick`** — multi-literal search (for alternations `foo|bar`); includes the SIMD **Teddy** algorithm for short literal sets.
- **`memchr`** — SIMD single/multi-byte scan; underlies Boyer-Moore candidate finding.
- **Literal optimization:** extract literal(s) from the regex and run a fast substring/multi-substring prescan; only invoke the full automaton near candidates. This is the *same idea* as trigram prefiltering but at the byte-scan level rather than the index level.
- **`grep-searcher` / `grep-matcher` / `grep-printer`** crates expose ripgrep's line search + context (`-A/-B/-C`) + printing; **`ignore`** crate gives the parallel gitignore-aware directory walk, `-g/--glob`, and `-t/--type` file-type filtering. glossa can reuse `ignore` and `grep-*` rather than re-implementing.
- Lesson: memory-maps are often *slower* than buffered reads for many small files.

### Russ Cox — regex → trigram boolean query (THE method)
The analysis computes, by structural recursion over the regex, four pieces per subexpression `e`: `emptyable`, `prefix(e)` (set of leading strings), `suffix(e)` (set of trailing strings), and `match(e)` (a boolean trigram query that any matching document must satisfy).
- **`trigrams(s)`** of a string: `ANY` if `len(s)<3`, else AND of all length-3 windows. For a *set* of strings: OR over members. (If any member is <3 chars → whole set collapses to `ANY` = "match everything".)
- **Combining rules (sketch):**
  - *Literal char/string:* prefix=suffix={s}, match=trigrams(s).
  - *Concatenation `xy`:* match = match(x) AND match(y) AND trigrams(suffix(x)×prefix(y)) (cross-boundary grams); prefix/suffix propagate, collapsing when a side is emptyable/too long.
  - *Alternation `x|y`:* match = match(x) OR match(y); prefix = prefix(x)∪prefix(y).
  - *Repetition `x*`:* emptyable, match = ANY (a `*` can match zero times → contributes nothing); `x+` keeps match(x).
- **Simplifications to stay tractable:** at any step may set `match(e) = match(e) AND trigrams(prefix(e))` / `... AND trigrams(suffix(e))`; cap set sizes and drop to `ANY` when sets explode. Final query is an AND/OR tree of trigrams evaluated against posting lists; documents that pass are then confirmed with the real regex.
- **Degrades to ANY (no acceleration) for:** patterns with no 3+ char literal run — `.`, `.*`, `a.b`, short alternations, `\d{3}`, anchors-only, huge char classes. These MUST fall back to full scan.

### Zoekt & livegrep (refinements at scale)
- **Zoekt: positional trigrams.** Stores trigram → list of (file, offset). To match a string it intersects only the *first* and *last* trigram posting lists and checks they sit at the right distance — so it touches few posting lists, can pick the *rarest* pair of trigrams in the pattern (e.g. "qui" over "the"), and the index need not be RAM-resident. Index ≈ 3× corpus (2× offsets + 1× content). Case handling: case-folded ngrams with a separate case bitmask.
- **livegrep: suffix array** over the whole corpus (concatenated). Regex search walks the regex's NFA against the sorted suffixes, turning regex match into range queries on the suffix array. Pro: handles *any* substring length incl. <3 chars; Con: index build is heavier, less incremental than per-file ngram sketches. For glossa's File-First incremental model, **per-file/per-chunk trigram posting lists (Cox/Zoekt style) fit better than a global suffix array.**

### fzf (and "fff")
- **fzf is a fuzzy *substring* finder/ranker, not regex and not an inverted index.** It scores subsequence matches (gap penalties, prefix/word-boundary bonuses) over a stream of candidate lines. It is **complementary, not part of grep**: useful as an optional `--fuzzy` interactive ranking mode over candidate lines/paths, orthogonal to literal/regex correctness.
- **"fff"** most likely = the user mis-typed **fzf**. Other things named "fff": a minimal bash file manager (`fff`), and "FFF" fast file find scripts — none are search-index tech. Treat as fzf. Recommendation: optionally expose fuzzy ranking via the **`nucleo`** crate (pure-Rust fzf-style matcher by Helix authors) for interactive result ordering; keep it out of the grep correctness path.

---

## 2. Concrete recommendations for glossa

### A. v1 grep — NO new index (reuse tantivy + regex)
Pipeline: **candidate select (existing BM25 index) → regex confirm (stored bodies)**.
1. Parse user pattern → HIR via `regex_syntax::parse`. Build the matcher with the `regex` crate (set `unicode`, smart-case as needed).
2. **Extract required literals** from the HIR with `regex_syntax::hir::literal::Extractor`:
   - Get prefix Seq and suffix Seq. Inspect `Seq::is_exact()`/literals. Set `limit_total` / `limit_len` to bound blow-up (e.g. `[A-Z]+`, `[ab]{3}{3}` → cap, else treat as no-literal).
   - If the Seq is `infinite`/too short/empty → **no useful prefilter** → mark "full scan" (iterate all stored chunk bodies).
3. **Use literals to prefilter via the EXISTING tantivy index** (no new index): the chunk `body` is already tokenized for BM25. For each extracted literal that is a whole word (or word-prefix), issue a tantivy term/phrase query on `body` to get a *candidate chunk set* (their stored `{path, location, body}`). Intersect (AND) across required literals; union (OR) across alternation branches — mirroring Cox's AND/OR structure but at the *word-token* level that BM25 already supports.
   - Caveat: BM25 tokenization is word/stem based, so this prefilter is **approximate and only sound as a *superset* if** the literal aligns with token boundaries. For substrings *inside* tokens (e.g. regex `oba` inside "foobar"), BM25 cannot find them → **fall back to full scan**. So: use the BM25 prefilter *only* when extracted literals are whole tokens of length ≥ 1 token; otherwise scan all bodies.
4. **Confirm:** run the compiled `regex` against each candidate chunk's STORED `body`. Emit matches with `{path, location}`. Because bodies are stored, v1 needs **zero file re-reads** for the index-backed path; for `-A/-B/-C` context that spans beyond the chunk, optionally re-open the file (File-First) at `path`+`location`.
5. Crates: `regex`, `regex-syntax`, `tantivy` (already present), `aho-corasick`/`memchr` (transitively via `regex`). Optional `ignore`/`grep-searcher` only if also scanning raw files.

**Selling point:** v1 is correct and File-First today — worst case it scans stored bodies (which are in-index, fast), best case BM25 word-prefilter slashes candidates. No new on-disk structure.

### B. Phase-2 — trigram accelerator (for large bases)
Build a trigram posting store; translate regex→trigram boolean query (Cox) → confirm with `regex`.

Two options evaluated:

**(i) Trigram tokens as an ngram field inside tantivy.**
- Add a field `body_trigrams` using tantivy's `NgramTokenizer { min:3, max:3, prefix_only:false }` (pure-Rust, already C-free). Query = AND/OR of 3-char terms produced by the Cox analysis, run as a tantivy BooleanQuery returning candidate chunks; confirm with `regex`.
- Pros: reuses tantivy storage/mer/commit/incremental segment machinery and the existing File-First rebuild path; one index to manage; postings compression for free.
- Cons: tantivy term dict overhead per trigram; no *positional* distance check (it's set-membership, like Cox per-file, not Zoekt positional) — fine, confirmation step handles precision; ngram field roughly doubles index size.

**(ii) Separate pure-Rust `redb` posting store.**
- `redb` (pure-Rust embedded KV, C-free) table `trigram (`[u8;3]`) → roaring bitmap of chunk-ids` (use `roaring` crate for compressed postings). Query intersects/unions bitmaps per Cox tree; map chunk-ids → stored bodies (in tantivy) → confirm regex.
- Pros: tiny, fast AND/OR over `roaring` bitmaps; full control of byte-trigram semantics (critical for Cyrillic, see pitfalls); can store positional offsets later (Zoekt-style) if needed.
- Cons: a second store to keep in sync with the chunk set / File-First invalidation; you reimplement build+merge.

**Recommendation:** start phase-2 with **(i) tantivy ngram field** — least new code, inherits File-First incremental indexing and the disposable-accelerator property. Move to **(ii) redb+roaring** only if profiling shows the tantivy term-dict/word-tokenizer semantics hurt (esp. for byte-level Cyrillic trigrams, which an ngram *character* tokenizer on codepoints handles, but where you may want raw byte trigrams).

Cox translation module (shared by both): recursive HIR walk producing an `enum TrigramQuery { Any, Trigram([char;3]), And(Vec), Or(Vec) }`; `Any` short-circuits to full-scan fallback. Reuse `regex_syntax` HIR; do NOT hand-roll the parser.

### C. Flags and how each touches prefiltering
- **`-i` / smart-case:** if pattern has no uppercase → case-insensitive. For trigram gen, **case-fold the corpus AND the query trigrams identically** (index lowercased trigrams; lowercase the pattern's literals before gram extraction). With `regex`, set `(?i)`; with BM25 prefilter, rely on tantivy's lowercasing tokenizer. Unicode case-folding for Cyrillic must use `char::to_lowercase` (Rust handles Cyrillic).
- **`-F` fixed-string:** skip regex parse; the literal IS the gram source → best-case prefilter (always selective if ≥3 chars). Use `aho-corasick`/`memchr` directly for confirm.
- **`-w` word boundary:** wrap as `\b…\b`; doesn't change trigram set (boundaries aren't grams) → prefilter unchanged, confirm with `\b` regex.
- **`-A/-B/-C` context:** post-confirm concern. If context exceeds the stored chunk, re-read the file at `path` (File-First). Use `grep-searcher` for clean context handling if scanning files.
- **`-g/--glob`, `-t/--type`:** pre-filter the *candidate chunk set by `path`/`file_type` fields* (both already stored per chunk) BEFORE trigram/BM25 selection — cheap AND on metadata, shrinks the search space first. Reuse `ignore`'s glob/types crate definitions for parity with ripgrep.

### D. Pitfalls
- **Un-accelerable regexes** → must detect and full-scan: anything where Cox analysis yields `ANY` — `.`, `.*`, `a.b`, `\d+`, short alternations, anchors-only, `^$`, very short patterns (<3 literal chars), `.`-heavy patterns, and huge Unicode classes (which explode the gram set — cap via `Extractor::limit_total`/`limit_len` and bail to scan). Always have a correct full-scan fallback; the accelerator is *optional*.
- **Unicode / Cyrillic trigrams (corpus is heavily Russian):** a Cyrillic char is 2 UTF-8 bytes, so **byte-trigrams cut across codepoints** and a 3-Cyrillic-char substring spans 6 bytes = 4 byte-trigrams (fine, but more grams) while a 1–2 char Cyrillic literal still yields ≥3 bytes (good — byte-trigrams stay selective even for short Cyrillic words, *better* than codepoint-trigrams which would collapse to ANY at <3 chars). **Decision: index BYTE-trigrams (n=3 over UTF-8 bytes), not codepoint-trigrams**, for selectivity on short Cyrillic terms — but then the regex→gram translation must also operate on the UTF-8 byte encoding of literals (encode each literal to bytes, take 3-byte windows). This argues for option (ii) redb+roaring with raw byte trigrams over tantivy's char-ngram tokenizer if byte-level selectivity matters. Keep consistent: index grams and query grams from the SAME byte encoding + SAME case-folding.
- **Case-insensitivity × trigrams:** case-fold both sides identically. For `(?i)` over Cyrillic, fold via Unicode lowercase before gram extraction; do not mix folded index with unfolded query.
- **Anchors `^ $ \b`:** contribute no grams; strip them for gram extraction, keep them for confirm. `^foo` still yields trigram `foo`.
- **Staleness / File-First correctness:** like ugrep, the accelerator may only *allow skipping*. Track chunk/file mtime vs index time; any newer-or-unindexed file is always full-scanned. Index is disposable: deletable, rebuildable from files.

---

## 3. Recommended v1 vs phase-2 split

**v1 (no new index, ship first):**
- Pattern → `regex` + `regex_syntax` literal `Extractor`.
- Metadata pre-filter on stored `path`/`file_type` (`-g`,`-t`).
- Optional BM25 word-token prefilter over existing tantivy `body` *only when extracted literals are whole tokens*; else full scan of stored bodies.
- Confirm with real `regex` on stored bodies; File-First file re-read only for out-of-chunk `-A/-B/-C` context.
- Flags: `-i`(smart-case), `-F`, `-w`, `-A/-B/-C`, `-g`, `-t`.
- Crates: `regex`, `regex-syntax`, `tantivy`(existing); optional `ignore`,`grep-searcher`,`nucleo`(fuzzy).

**Phase-2 (trigram accelerator for large bases):**
- Cox regex→`TrigramQuery` translator (shared module) with `Any`→fallback.
- Posting store: **start with tantivy `NgramTokenizer` field**; escalate to **`redb` + `roaring` byte-trigram store** if byte-level Cyrillic selectivity or positional checks are needed.
- Byte-trigrams, case-folded, File-First mtime-guarded skipping (ugrep monotonic property).
- Confirm step unchanged (real `regex` over stored bodies).

All recommended crates (`regex`, `regex-syntax`, `aho-corasick`, `memchr`, `tantivy`, `redb`, `roaring`, `ignore`, `grep-searcher`, `nucleo`) are **pure Rust / C-free**.
