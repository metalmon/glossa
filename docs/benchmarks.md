# glossa — benchmark log & progress toward SOTA

Curated record of agent-eval runs (the `kbx eval` harness). Raw `eval-*.json` reports are git-ignored;
this file is the durable, human-readable history. One row per meaningful run. Append, don't rewrite.

Two kinds of number live here, and they answer different questions. **Retrieval quality** (below) is
what the engine finds. **Extraction cost** (at the end) is what indexing and reading a corpus costs
in time and memory — added 2026-11-10, because the omnivorous-extraction work needs every phase to
report a delta against a measured baseline rather than a guess.

## Reference targets — HotpotQA (distractor setting), Answer EM / F1

| Reference | Regime | EM | F1 |
|---|---|---|---|
| Baseline (Yang et al. 2018) | supervised reader | ~44 | ~58 |
| DFGN / QFE | supervised + graph | ~55 | ~69 |
| **HGN / SAE (RoBERTa-large)** | supervised SOTA-class | **~69** | **~82** |
| Leaderboard top | supervised | ~70–72 | ~83–84 |

Notes: supervised numbers are models fine-tuned on HotpotQA reading the (gold+distractor) paragraphs
directly. glossa runs a **zero-shot agent** over a search tool — a different regime; the EM gap to
supervised models reflects the reader model, not retrieval. Distractor retrieval (10 docs) is easy by
design — the discriminating retrieval test is **fullwiki** (not yet run). Strict EM also penalizes
correct-but-differently-bounded spans, so true correctness runs higher than EM.

## Runs

| Date | Dataset | N | Backend / model | EM | F1 | Recall | eval@ | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-06-24 | hotpot_dev_distractor | 50 | openai / qwen3.5-4b (LM Studio) | **0.680** | **0.807** | **0.990** | 938f2e3 | First real run. Harness-side function-calling tool loop; terse answer prompt. recall=1.0 on 49/50. 9 near-misses are span-boundary (e.g. `3,677` vs `3,677 seated`) → factual correctness ≈ 0.85. 1 retrieval miss (`Fujioka, Gunma`→`Japan`). |
| 2026-06-25 | hotpot_dev_distractor | 50 | cli / claude (`claude -p`, server-side MCP loop) | **0.800** | **0.899** | **0.930** | ff5b8b3 | A/B vs the Qwen row: SAME 50 questions, SAME glossa engine, reader swapped 4B→Claude. recall=1.0 on 46/50, 0 failed, 7 span-boundary near-misses (`Lee Hazlewood` vs `Barton Lee Hazlewood`) → factual correctness ≈ 0.94. ~3 genuine reader errors (`4,000` vs `3,677 seated`). |

### A/B read (reader contribution, engine held constant)
Same 50 questions, identical glossa retrieval; only the reader changes:

| Arm | EM | F1 | Recall |
|---|---|---|---|
| Qwen3.5-4B | 0.68 | 0.81 | 0.99 |
| Claude | **0.80** | **0.90** | 0.93 |

- Swapping only the reader lifts EM +0.12 / F1 +0.09 — **into supervised-SOTA territory** (~0.69 EM / ~0.82 F1).
  The 4B already had recall 0.99 (it *held* the right docs), so its lower score was **reader-limited, not
  retrieval-limited** — the engine claim.
- **Recall caveat:** Claude's recall is *lower* (0.93 vs 0.99) not because the engine found less, but because
  Claude is a more **targeted** agent — it answers in fewer searches, touching fewer gold docs. Our recall
  metric counts docs-surfaced-during-retrieval, so it partly measures agent search *behavior*, not pure
  engine quality. For a clean engine-retrieval number, use a fixed-search-budget protocol or fullwiki Recall@k.

## Where we are / where to go

- **Now:** A/B done on 50q. Claude reader hits **EM 0.80 / F1 0.90** (supervised-SOTA class); the 4B reaches
  0.68/0.81 on the same retrieval. Engine claim validated: with the right docs surfaced (recall ~0.99), a
  strong reader scores at SOTA level — the gap is the reader, not glossa.
- **Next milestones:**
  1. ~~Claude arm A/B~~ **DONE** (2026-06-25): EM 0.80 / F1 0.90.
  2. **Larger N** (200–500) for a stable point estimate; confirm 50-q numbers hold (both arms).
  3. **fullwiki** — the real retrieval test (Recall@k over all of Wikipedia, not 10 distractors). This is
     where the recall metric becomes discriminating (distractor recall is near-saturated/agent-behaviour-bound).
  4. **Graph A/B** (graph OFF vs ON) once a graph-off query mode exists — the multihop claim.
- Caveat: these benches are English-Wikipedia clean text; they validate the multihop/retrieval engine,
  not glossa's product edge (office/pdf/legacy encodings/offline/agentic graph).

## Extraction cost — indexing and read, per corpus

Produced by `just bench-baseline <path-to-ci-built-kb>` (`kbx bench extract`). One row per corpus
per run; append, don't rewrite.

The two real corpora are private, so they appear as **corpus A** (38 mixed documents — 33 pdf,
4 docx, 1 odt — largest 39.6 MiB) and **corpus B** (11 PDFs, largest 0.6 MiB). Their shapes are what
makes the numbers readable; their names are not.

**How to read a row, and what it does not cover.**

- The measured binary is a **CI artifact**, never a local build — a baseline taken on a binary
  nobody ships measures the wrong thing. Its `kb --version` is in the row.
- Corpora are measured as **copies**. `--force` rebuilds a corpus's index, and on a corpus whose
  manifest predates the current schema it migrates it as well, so the `Migration` column says which
  side of that the row is. corpus A sits at schema 3 against a current 5: a row reading
  `schema 3→5` measured migrated bodies; one reading `schema 5→5` measured a corpus already there.
- **Peak RSS is Linux-only.** On Windows the cell reads `not measured`, not a zero: the Win32 route
  needs a new direct dependency, and polling a shell through a multi-minute run would steal CPU
  from the process being measured. On that platform the memory evidence is only indirect: the
  **Largest file** column and the generated corpus's size ladder. The ladder has now been run to
  16 MiB with per-file cost staying linear, so it is not evidence of a per-file memory ceiling — it
  is evidence that there is none below 16 MiB.
- **Per-file p50/p95 come from timestamping `kb index`'s own progress lines**, so they include that
  file's extraction, chunking, index write and graph write — not extraction alone.
- **The last file walked gets no per-file sample.** A progress line says when a file *started*, so
  the last one has no closing tick; running its interval to process exit charges it the whole
  finalize, and that is what produced this log's retracted "16 MiB cliff" (below). Generated ladder
  corpora now end with a tiny file under `zz-tail-sentinel/` that sorts after every other path, so
  the ladder's largest file gets a real duration.
- **Tail is everything after the last file started**: that file's own work plus the finalize —
  cross-document link resolution, the tantivy commit, the manifest, the DF sidecar. `(df N)` in the
  cell is the DF sidecar write, which `kb index` brackets with its own stderr lines. Read the tail
  against **Chunks**, never against the last file's size.
- **read p50/p95 are CLI invocations**, so each carries process startup. They are a floor for the
  serving path, not a measurement of in-process MCP reads.
- **No network mount has been measured.** Every row here is local disk, and the extraction design's
  cost reasoning about SMB/NFS is therefore unverified by these numbers.
- **Load med/peak says how busy the machine was**, as a factor against a fixed integer workload
  timed before the run started: `x1.00` means the machine stayed as quiet as it was at the start,
  `x3.00` that something else was taking the CPU. Rows reading `not measured` were taken before the
  probe existed, which is why none of them can be told apart from a load spike.
- There is **no regression gate**. A slower row is a finding to explain, not a build failure.

| Corpus | kb | Platform | Files | Wall | per-file p50/p95 | Tail | Peak RSS | Chunks | Chunks/s | Largest file | Migration | read p50/p95 | Load med/peak | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| corpus A (copy) | kb 0.5.7 (none) | windows | 38 | 28451 ms | 49 / 5270 ms | — | not measured | 2349 | 83 | 39.6 MiB | schema 3→5 | 12 / 13 ms | not measured | ok |
| corpus A (copy, repeat) | kb 0.5.7 (none) | windows | 38 | 26751 ms | 33 / 4881 ms | — | not measured | 2349 | 88 | 39.6 MiB | schema 5→5 | 11 / 14 ms | not measured | ok |
| corpus B (copy) | kb 0.5.7 (none) | windows | 11 | 1924 ms | 138 / 397 ms | — | not measured | 173 | 90 | 0.6 MiB | schema 2→5 | 12 / 13 ms | not measured | ok |
| generated, 2000 docs + 1→16 MiB ladder | kb 0.5.7 (none) | windows | 2005 | 191651 ms | 6 / 23 ms | — | not measured | 29205 | 152 | 16.0 MiB | schema 1→5 | 11 / 13 ms | not measured | ok |
| ladder only, 1→16 MiB | kb 0.5.7 (none) | windows | 5 | 73792 ms | 1618 / 67764 ms | — | not measured | 19205 | 260 | 16.0 MiB | schema 1→5 | 11 / 12 ms | x1.00 / x1.27 | ok |
| ladder only, 1→16 MiB (repeat) | kb 0.5.7 (none) | windows | 5 | 79665 ms | 1696 / 72502 ms | — | not measured | 19205 | 241 | 16.0 MiB | schema 5→5 | 13 / 14 ms | x1.01 / x2.60 | ok |
| ladder only, 1→16 MiB (repeat) | kb 0.5.7 (none) | windows | 5 | 88302 ms | 1608 / 81211 ms | — | not measured | 19205 | 217 | 16.0 MiB | schema 5→5 | 11 / 14 ms | x0.98 / x3.06 | ok |

| 16 MiB alone + sentinel | kb 0.5.7 (none) | windows | 2 | 25167 ms | 6236 / 6236 ms | 18892 ms | not measured | 9911 | 394 | 16.0 MiB | schema 1→5 | 12 / 12 ms | x1.00 / x1.08 | ok |
| 16 × 1 MiB + sentinel | kb 0.5.7 (none) | windows | 17 | 15643 ms | 387 / 447 ms | 9368 ms | not measured | 9921 | 634 | 1.0 MiB | schema 1→5 | 11 / 15 ms | x1.00 / x1.05 | ok |
| ladder 1→16 MiB + sentinel | kb 0.5.7 (none) | windows | 6 | 75741 ms | 1562 / 8402 ms | 61561 ms | not measured | 19206 | 254 | 16.0 MiB | schema 1→5 | 11 / 12 ms | x1.00 / x1.80 | ok |
| bisect 8/10/12/14/16 MiB + sentinel | kb 0.5.7 (none) | windows | 6 | 271374 ms | 4683 / 9152 ms | 244537 ms | not measured | 37161 | 137 | 16.0 MiB | schema 1→5 | 11 / 13 ms | x1.00 / x3.41 | ok |
| corpus A (copy) | kb 0.5.7 (none) | windows | 38 | 26769 ms | 35 / 4250 ms | 7145 ms (df 11) | not measured | 2349 | 88 | 39.6 MiB | schema 5→5 | 12 / 15 ms | x1.00 / x1.07 | ok |
| corpus B (copy) | kb 0.5.7 (none) | windows | 11 | 1955 ms | 100 / 368 ms | 208 ms (df 4) | not measured | 173 | 88 | 0.6 MiB | schema 5→5 | 12 / 14 ms | x1.00 / x1.18 | ok |
| ladder 1→16 MiB + sentinel (repeat) | kb 0.5.7 (none) | windows | 6 | 78168 ms | 1562 / 11981 ms | 59209 ms (df 1) | not measured | 19206 | 246 | 16.0 MiB | schema 5→5 | 11 / 15 ms | x1.00 / x1.08 | ok |
| 16 MiB alone, 50k-word vocab | kb 0.5.7 (none) | windows | 2 | 20546 ms | 6404 / 6404 ms | 14057 ms (df 15) | not measured | 8991 | 438 | 16.0 MiB | schema 1→5 | 15 / 16 ms | x0.98 / x1.55 | ok |
| ladder 1→16 MiB, 50k-word vocab | kb 0.5.7 (none) | windows | 6 | 56614 ms | 1566 / 6604 ms | 44009 ms (df 20) | not measured | 17426 | 308 | 16.0 MiB | schema 1→5 | 11 / 12 ms | x1.00 / x1.29 | ok |
| bisect 8/10/12/14/16 MiB, 50k-word vocab | kb 0.5.7 (none) | windows | 6 | 219316 ms | 6458 / 10625 ms | 188292 ms (df 17) | not measured | 33726 | 154 | 16.0 MiB | schema 1→5 | 11 / 12 ms | x1.00 / x1.25 | ok |
| corpus A (copy, + sentinel) | kb 0.5.7 (none) | windows | 39 | 27339 ms | 36 / 4024 ms | 7126 ms (df 13) | not measured | 2350 | 86 | 39.6 MiB | schema 5→5 | 13 / 17 ms | x1.00 / x1.11 | ok |
| corpus B (copy, + sentinel) | kb 0.5.7 (none) | windows | 12 | 1888 ms | 103 / 368 ms | 138 ms (df 4) | not measured | 174 | 92 | 0.6 MiB | schema 5→5 | 11 / 14 ms | x1.01 / x1.01 | ok |
| 16 × 1 MiB, SIMILAR fix | kb 0.5.7 (none) | windows | 17 | 8919 ms | 453 / 566 ms | 372 ms (commit 60, links 0, df 0) | not measured | 9921 | 1112 | 1.0 MiB | schema 5→5 | 12 / 19 ms | x1.04 / x1.53 | ok |
| ladder 1→16 MiB, SIMILAR fix | kb 0.5.7 (none) | windows | 6 | 25455 ms | 1877 / 15791 ms | 837 ms (commit 69, links 0, df 1) | not measured | 19206 | 755 | 16.0 MiB | schema 5→5 | 14 / 20 ms | x0.96 / x1.44 | ok |
| bisect 8→16 MiB, SIMILAR fix | kb 0.5.7 (none) | windows | 6 | 43330 ms | 7464 / 9719 ms | 1352 ms (commit 74, links 0, df 1) | not measured | 37161 | 858 | 16.0 MiB | schema 5→5 | 15 / 18 ms | x1.04 / x2.69 | ok |
| corpus A (copy, + sentinel), SIMILAR fix | kb 0.5.7 (none) | windows | 39 | 34086 ms | 58 / 5046 ms | 6098 ms (commit 113, links 0, df 20) | not measured | 2350 | 69 | 39.6 MiB | schema 5→5 | 13 / 16 ms | x1.01 / x1.35 | ok |
| corpus A (copy, + sentinel), **directml** | kb 0.5.7 (ort-directml) | windows | 39 | 28580 ms | 34 / 5139 ms | 5245 ms (commit 110, links 0, df 14) | not measured | 2350 | 82 | 39.6 MiB | schema 5→5 | 22 / 31 ms | x1.00 / x3.24 | ok |
| corpus B (copy, + sentinel), **directml** | kb 0.5.7 (ort-directml) | windows | 12 | 2250 ms | 126 / 460 ms | 125 ms (commit 74, links 0, df 3) | not measured | 174 | 77 | 0.6 MiB | schema 5→5 | 21 / 25 ms | x1.01 / x1.06 | ok |
**2026-11-10 — first baseline, taken before any omnivorous-extraction change.** Measured with the
CI artifact `glossa-x86_64-pc-windows-msvc-none` from the `release.yml` dispatch on
`fix/full-coverage-pass-and-one-search-depth`. The `none` engine is `--no-default-features`: no ONNX
runtime, no model discovery, so a later delta is attributable to extraction rather than to provider
start-up. That is **not** the feature set a Windows deployment ships — a `directml` row for the
shipped set is still owed.

### What these first rows establish

**RETRACTED — the "16 MiB cliff" below was this harness, not the code.** The rows above charged
the last file walked with everything the run does after it: the tantivy commit, the manifest, the DF
sidecar. A progress line says when a file *started*, so the last file has no closing tick, and in
every corpus measured `ladder/big-04-16MiB.md` sorted last. Measured with a closing tick, that file
costs 8.4 s in the ladder and 6.2 s alone. The retracted text is kept below as written, because a
log that quietly deletes its wrong findings teaches nobody which instrument to distrust; the
corrected figures are in the next block.

**Lead (RETRACTED): indexing one large file may be sharply superlinear, with a step between 8 and 16 MiB.** From
the generated corpus's ladder, each step one doubling of the one before:

| File | Time | vs previous step |
|---|---|---|
| `big-01` 2 MiB | 773 ms | — |
| `big-02` 4 MiB | 1 555 ms | ×2.0 |
| `big-03` 8 MiB | 3 151 ms | ×2.0 |
| **`big-04` 16 MiB** | **164 749 ms** | **×52** |

Linear to 8 MiB, then 52× for one more doubling: that single 16 MiB Markdown file took 165 s, 86% of
the whole 191 s run over 2005 documents. Neither the cause nor the effect is established — the cause
is not guessed at, and the effect needs repeating on a quiet machine before it counts. What the
three lower steps do support, being mutually consistent and cheap enough to be load-insensitive, is
that per-file cost scales with size at least linearly, so a corpus's largest documents dominate its
indexing time. corpus A's largest document is 39.6 MiB, past the top of this ladder; the ladder
needs extending past 16 MiB either way.

**Cost concentrates in a few documents on a real corpus.** corpus A's per-file p95 is ~100× its
p50 (33 ms against 4 881 ms), and the five costliest are all PDFs. corpus B, 11 PDFs with nothing
above 0.6 MiB, has a p95/p50 of only 2.9×. So "how long does indexing take" is answered by which
large documents a corpus holds, not by its document count. (This survives the retraction: the p95
here is a real document, and the corrected run puts it at 4.1 s against a 37 ms p50. Only the
costliest file's figure moved, because that one was the last walked.)

**Repeatability, from the two corpus A rows:** chunk count identical (2349 both times, so
extraction is deterministic), wall within 6%, p95 within 8%, but **p50 within 33%** — a median over
38 samples is noisy. A later phase showing under ~10% movement on this corpus is therefore not
showing a signal. This is what the design wanted instead of a regression gate invented in advance.

**Migration is visible per row, and the first run of each corpus did one.** corpus A came in at
schema 3 and corpus B at **2**, against a current 5 — both working corpora were serving bodies
produced by older extractors, corpus B by three versions. The repeat row reads `schema 5→5`: the
same corpus, different input conditions, which is exactly why the column exists.

**corpus A indexed 38 documents against the 34 in its manifest.** The four uppercase `.PDF`
certificates that commit `3fafe9f` made walkable are in the index for the first time.

### 2026-11-10, second pass — the same corpora with a fixed instrument

Same CI artifact and same machine, with two changes to the measurement: the last file walked no
longer absorbs the finalize (it is reported as **Tail**), and generated ladders end with a
`zz-tail-sentinel/` file so the ladder's largest document gets a real duration. Every row was taken
with the machine quiet — median load x0.96–x1.02.

**Per-file cost is linear in file size from 1 to 16 MiB.** One run, one corpus, each file with a
closing tick:

| File | 1 MiB | 2 MiB | 4 MiB | 8 MiB | 16 MiB |
|---|---|---|---|---|---|
| Cost | 380 ms | 715 ms | 1 562 ms | 3 080 ms | 8 402 ms |
| vs previous | — | ×1.9 | ×2.2 | ×2.0 | ×2.7 |

And inside the doubling the retracted finding put a cliff in, from the bisect row: 8 MiB 3 178 ms,
10 MiB 3 873 ms, 12 MiB 4 683 ms, 14 MiB 5 908 ms, 16 MiB 9 152 ms — 397, 387, 390, 422, 572 µs per
MiB. There is a mild uptick in the last step and no cliff anywhere.

**A single large document still costs more than the same bytes spread out, but by a factor, not an
order.** Same generator, same ~9 900 chunks: sixteen 1 MiB files take 15.6 s wall with a 9.4 s tail;
one 16 MiB file takes 25.2 s with an 18.9 s tail. So one big document costs ~1.6× the wall and ~2×
the finalize of the same bytes in sixteen documents.

**The cost that grows with the corpus is the finalize, not any file.** Tail against index size, all
four synthetic rows:

| Chunks | 9 921 | 9 911 | 19 206 | 37 161 |
|---|---|---|---|---|
| Tail | 9 368 ms | 18 892 ms | 61 561 ms | 244 537 ms |

Each doubling of the corpus multiplies the finalize by ~3.3–4.0, and in the bisect row it is 245 s
of a 271 s run. **What the tail is, is NOT established.** It is not the DF sidecar: that phase is now
timed separately and comes in at 1 ms here and 11 ms on corpus A. The remaining candidates — the
tantivy commit and its merges, cross-document link resolution, the last file's own work — are named
here only so the next measurement knows where to look, not as an explanation.

**And the real corpora do not reproduce it.** Corpus A: 2 349 chunks, tail 7.1 s of a 26.8 s run.
Corpus B: 173 chunks, tail 0.2 s of 2.0 s. Corpus A has a quarter of the chunks of the cheapest
synthetic row and about the same tail, so the synthetic curve is not a statement about corpus size
alone — the generated corpus has a 32-word vocabulary and near-duplicate text, which is a
pathological shape for an inverted index. **Do not plan from the quadratic.** The next measurement
that would settle it is a generated corpus with a realistic vocabulary at several sizes.

**A single per-file sample moves ±40%.** The ladder was measured twice: wall 75.7 s and 78.2 s (3%),
tail 61.6 s and 59.2 s (4%), but the 16 MiB file itself 8 402 ms and 11 981 ms. Chunk counts were
identical (19 206). So a whole-corpus figure here is repeatable and a single file's figure is not —
read one file's cost as a factor, never as a number.

### 2026-11-10, third pass — is the finalize about corpus size or corpus shape?

The second pass left the near-quadratic finalize with an obvious suspect: the generator drew from 32
words, so a few terms carried enormous posting lists over near-duplicate text — nothing like a
document corpus. `--vocab 50000` draws Zipf-weighted from generated pseudo-words instead, leaving
the byte count and the ladder identical, and the real corpora were re-run with `--tail-sentinel` so
their finalize is separated from the last document's own cost.

One caveat about re-running these three: they were measured before a fix to the rank-to-index shift
in the Zipf draw, so their corpora used a 50 000-word vocabulary whose commonest word was never
drawn. The distribution's shape was unaffected and the rows stand, but the same seed now generates a
slightly different corpus — re-measure the pair, not just one side of it.

**The vocabulary was not it.** Three matched pairs:

| Chunks | Tail @ 32 words | Chunks | Tail @ 50k words | ratio |
|---|---|---|---|---|
| 9 911 | 18 892 ms | 8 991 | 14 057 ms | 0.74 |
| 19 206 | 61 561 ms | 17 426 | 44 009 ms | 0.71 |
| 37 161 | 244 537 ms | 33 726 | 188 292 ms | 0.77 |

A realistic vocabulary is a flat ~0.74x constant and leaves the growth alone: the exponent against
chunk count is 1.79 / 2.09 at 32 words and 1.72 / 2.20 at 50 000. Per-file cost drops with it too
(the 16 MiB file: 8 402 → 6 604 ms), which fits — fewer repeated terms, shorter posting lists per
term. **So the finalize is ~quadratic in index size within one corpus family, and the synthetic
vocabulary did not cause that.**

**But chunk count does not predict its level, so do not extrapolate it.** With a sentinel in the
copy, corpus A's finalize is 7 126 ms at 2 350 chunks — the generated curve at that chunk count
predicts ~960 ms, a factor of 7 out. The sentinel barely moved it (7 145 → 7 126 ms), so that tail
was already the finalize and not the last PDF. The 16×1 MiB versus 1×16 MiB control says the same
from the other side: identical chunk counts, finalize 9 368 ms against 18 892 ms. Something other
than chunk count is in there, and a figure like "N minutes at 100k chunks" cannot be read off these
rows in either direction.

**What is ruled out:** the DF sidecar. It is timed separately now and costs 1–20 ms everywhere,
including 13 ms on the real corpus. The remaining candidates inside the tail are cross-document link
resolution and the tantivy commit with its merges — which is why the next step is to have `kb index`
bracket those two phases on stderr the way it already brackets the sidecar, so a row names the
quadratic phase instead of leaving two suspects.

### 2026-11-10, fourth pass — the finalize is the graph, not the index

`kb index` now brackets each finalize phase on stderr and the harness times them, so the row's tail
reads `(commit N, links N, df N)`. The three named phases turned out to account for almost none of
it:

| Corpus | Chunks | Tail | commit | links | df |
|---|---|---|---|---|---|
| 16 × 1 MiB | 9 921 | 11 402 ms | 58 ms | 0 ms | 1 ms |
| 16 MiB, one file | 9 911 | 22 739 ms | 69 ms | 0 ms | 1 ms |
| ladder 1→16 MiB | 19 206 | 67 225 ms | 76 ms | 0 ms | 1 ms |
| bisect 8→16 MiB | 37 161 | 259 722 ms | 52 ms | 0 ms | 1 ms |
| corpus A | 2 350 | 7 054 ms | 80 ms | 0 ms | 11 ms |

**The tantivy commit is 52–80 ms and does not grow with the corpus at all.** Reference resolution is
0 ms. The DF sidecar is 1–11 ms. Everything else in the tail — 11 s to 260 s of it — is the one
thing left between the last file and process exit: the **graph generalization pass that `--force`
runs by itself** after indexing (`src/main.rs:2016`: transitive closure, SIMILAR edges, communities,
centrality).

Timed standalone with `kb graph generalize` (non-merge, exactly as `--force` invokes it):

| Corpus | Tail measured | `graph generalize` alone | What it produced |
|---|---|---|---|
| 16 × 1 MiB | 11 402 ms | 11 772 ms | 64 SIMILAR edges, 0 inferred, 0 communities |
| 16 MiB, one file | 22 739 ms | 23 151 ms | 4 SIMILAR edges |
| ladder 1→16 MiB | 67 225 ms | 67 479 ms | 20 SIMILAR edges |

Within 0.4–3%. **So the "near-quadratic finalize" of the earlier passes is the graph pass, and it is
not extraction, not chunking, not the index.** Two consequences worth stating plainly:

- **The cost is ~quadratic in node count and the work is nearly wasted here.** Eleven seconds for 64
  edges at 10k nodes, 23 s for 4 edges, 67 s for 20. A document corpus's `Section` labels are mostly
  unique, so a pairwise SIMILAR search finds almost nothing — while still paying for every pair.
  The same shape explains the one-file-versus-many gap that looked like an indexing effect earlier
  (23 s against 11.8 s at equal node counts): it is in the graph pass too.
- **It only happens on `--force`.** An incremental `kb index` does not run it, which is why a
  freshen is quick and why this never showed up in serving.

What this does NOT say: that the generalize pass is wrong to exist (it keeps derived edges in sync,
and on a corpus with a real semantic graph it has something to find), nor which part of it is the
quadratic one — closure, SIMILAR and community detection were timed as one pass. Both are questions
for the graph layer, not for extraction, and they are now the first thing to look at before any
format work makes corpora bigger.

### 2026-11-10, fifth pass — the fix, measured against the rows above

`linkpred::jaccard_pairs` now pairs reasoning nodes only (the ontology's structural types are
excluded from BOTH SIMILAR arms, where before the filter reached only the label-BM25 one).
Neighbourhoods are still built from every edge. Same corpora, same machine, same binary family:

| Corpus | Chunks | Tail before | Tail after | Wall before → after |
|---|---|---|---|---|
| 16 × 1 MiB | 9 921 | 11 402 ms | **372 ms** | 18.7 s → 8.9 s |
| ladder 1→16 MiB | 19 206 | 67 225 ms | **837 ms** | 82.1 s → 25.5 s |
| bisect 8→16 MiB | 37 161 | 259 722 ms | **1 352 ms** | 294.3 s → 43.3 s |
| corpus A | 2 350 | 7 054 ms | 6 098 ms | 30.3 s → 34.1 s |

**The quadratic is gone from the structural case.** 372 / 837 / 1 352 ms over 9.9k / 19.2k / 37.2k
chunks is flat per chunk (~0.037 ms). Full reindex throughput on the largest synthetic corpus went
126 → 858 chunks/s.

**Nothing semantic was lost.** Standalone `kb graph generalize`, before → after:

| Corpus | Time | SIMILAR edges | Communities |
|---|---|---|---|
| ladder 1→16 MiB | 67 479 → 849 ms | 20 → **0** | 0 → 0 |
| bisect 8→16 MiB | 288 667 → 1 602 ms | 20 → **0** | 0 → 0 |
| corpus A | 12 712 → 10 245 ms | 9 356 → **9 151** | 91 → 91 |

The synthetic corpora's SIMILAR edges were all section-to-section — the noise retrieval de-weights —
and they are now simply not derived. On the corpus with a real reasoning graph, 205 edges of 9 356
went (2.2%), every community survived, and the pass still costs 10 s: **the pairing loop is still
quadratic, now in reasoning nodes, which is the set it was written for.** A corpus whose graph is
genuinely rich still pays for that, and this fix does not claim otherwise.

**What did not improve, and one caution.** Corpus A's wall went slightly UP (30.3 → 34.1 s) and its
costliest file moved 4 552 → 5 046 ms; both are inside the noise this log already documents for
single-file samples. That noise turned out to be wider than the ±40% recorded in the second pass:
the ladder's 16 MiB file has now been sampled at 8 402, 11 981 and 15 791 ms across runs on a quiet
machine, a spread of 1.9x. **Read a single file's cost as an order of magnitude, never as a
number** — only whole-corpus wall and tail have been repeatable here (3–4%).

### 2026-11-10 — the `directml` row the baseline owed

Every row above was taken on `--no-default-features` (`none`), which is not what a Windows
deployment ships. The two rows measured on the shipped feature set, same corpora and same commit:

| Corpus | Engine | Wall | per-file p50/p95 | Tail | read p50/p95 |
|---|---|---|---|---|---|
| corpus A | none | 34 086 ms | 58 / 5 046 ms | 6 098 ms | 13 / 16 ms |
| corpus A | **directml** | 28 580 ms | 34 / 5 139 ms | 5 245 ms | **22 / 31 ms** |
| corpus B | none | 1 888 ms | 103 / 368 ms | 138 ms | 11 / 14 ms |
| corpus B | **directml** | 2 250 ms | 126 / 460 ms | 125 ms | **21 / 25 ms** |

**Indexing does not care which engine is compiled in** — the wall difference goes both ways and
sits inside this log's documented noise, which is what you would expect: nothing in the index path
scores anything. **The read path roughly doubles**, 13→22 and 11→21 ms at p50. These are CLI
invocations, so that is ONNX-runtime process start-up, not per-query inference: a `kb read` on the
shipped build pays about 10 ms more before it does anything. In-process MCP reads pay that once per
server, not once per call — so this number is a floor for the CLI and says nothing about serving.

### DF ratios (`df(token)/n_chunks`, what the `verify` gate compares against)

| Corpus | Sample |
|---|---|
| corpus A | `plc` 0.036 · `modbus` 0.063 · `iec` 0.049 |
| corpus B | five sampled domain terms: 0.185 · 0.179 · 0.121 · 0.087 · 0.000 |
| generated | `zylathon` 0.166 · `qorvex` 0.034 · `brimtalve` 0.135 · `dunmorrow` 0.003 |

Two things to know before trusting a ratio here. A token must occur **as a whole token**: one
sampled term came back 0.000 because it only ever appears inside a longer token (a hostname), and a
substring search is no evidence about a token-keyed table. And another came back 0.000 because the
corpus's language inflects and the corpus carries a declined form, not the base one. Pick the form
that occurs; the echoed ratio is the verification, which is why the tool prints every token beside
its number.
