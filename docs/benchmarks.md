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

**How to read a row, and what it does not cover.**

- The measured binary is a **CI artifact**, never a local build — a baseline taken on a binary
  nobody ships measures the wrong thing. Its `kb --version` is in the row.
- Corpora are measured as **copies**. `--force` rebuilds a corpus's index, and on a corpus whose
  manifest predates the current schema it migrates it as well, so the `Migration` column says which
  side of that the row is. `kb-abac` sits at schema 3 against a current 5: a row reading
  `schema 3→5` measured migrated bodies; one reading `schema 5→5` measured a corpus already there.
- **Peak RSS is Linux-only.** On Windows the cell reads `not measured`, not a zero: the Win32 route
  needs a new direct dependency, and polling a shell through a multi-minute run would steal CPU
  from the process being measured. On that platform the memory evidence is the **Largest file**
  column together with the generated corpus's size ladder (1→16 MiB), which is where a per-file
  ceiling shows up as degradation.
- **Per-file p50/p95 come from timestamping `kb index`'s own progress lines**, so they include that
  file's extraction, chunking, index write and graph write — not extraction alone.
- **read p50/p95 are CLI invocations**, so each carries process startup. They are a floor for the
  serving path, not a measurement of in-process MCP reads.
- **No network mount has been measured.** Every row here is local disk, and the extraction design's
  cost reasoning about SMB/NFS is therefore unverified by these numbers.
- **Load med/peak says how busy the machine was**, as a factor against a fixed integer workload
  timed before the run started: `x1.00` means the machine stayed as quiet as it was at the start,
  `x3.00` that something else was taking the CPU. Rows reading `not measured` were taken before the
  probe existed, which is why none of them can be told apart from a load spike.
- There is **no regression gate**. A slower row is a finding to explain, not a build failure.

| Corpus | kb | Platform | Files | Wall | per-file p50/p95 | Peak RSS | Chunks | Chunks/s | Largest file | Migration | read p50/p95 | Load med/peak | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kb-abac` (copy) | kb 0.5.7 (none) | windows | 38 | 28451 ms | 49 / 5270 ms | not measured | 2349 | 83 | 39.6 MiB | schema 3→5 | 12 / 13 ms | not measured | ok |
| `kb-abac` (copy, repeat) | kb 0.5.7 (none) | windows | 38 | 26751 ms | 33 / 4881 ms | not measured | 2349 | 88 | 39.6 MiB | schema 5→5 | 11 / 14 ms | not measured | ok |
| `kb-gost` (copy) | kb 0.5.7 (none) | windows | 11 | 1924 ms | 138 / 397 ms | not measured | 173 | 90 | 0.6 MiB | schema 2→5 | 12 / 13 ms | not measured | ok |
| generated, 2000 docs + 1→16 MiB ladder | kb 0.5.7 (none) | windows | 2005 | 191651 ms | 6 / 23 ms | not measured | 29205 | 152 | 16.0 MiB | schema 1→5 | 11 / 13 ms | not measured | ok |

**2026-11-10 — first baseline, taken before any omnivorous-extraction change.** Measured with the
CI artifact `glossa-x86_64-pc-windows-msvc-none` from the `release.yml` dispatch on
`fix/full-coverage-pass-and-one-search-depth`. The `none` engine is `--no-default-features`: no ONNX
runtime, no model discovery, so a later delta is attributable to extraction rather than to provider
start-up. That is **not** the feature set a Windows deployment ships — a `directml` row for the
shipped set is still owed.

### What these first rows establish

**UNREPLICATED, AND THE MACHINE WAS NOT IDLE — read the next finding as a lead, not a fact.** Every
row above was taken on a developer workstation with other work running on it (Docker among others),
and each number below is a **single sample of a single file**. The repeatability check further down
measures ~6-8% noise on a whole-corpus wall time; it says nothing about how much a load spike can
move one 165-second file. A 52x step is far too large to dismiss as noise, and far too
single-sampled to assert as behaviour. Replication and a load probe are the next task; until they
land, this is the strongest lead the baseline produced and not more than that.

**Lead: indexing one large file may be sharply superlinear, with a step between 8 and 16 MiB.** From
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
indexing time. `kb-abac`'s largest document is 39.6 MiB, past the top of this ladder; the ladder
needs extending past 16 MiB either way.

**Cost concentrates in a few documents on a real corpus.** `kb-abac`'s per-file p95 is ~100× its
p50 (33 ms against 4 881 ms), and the five costliest are all PDFs — the top one a 7.2 s certificate.
`kb-gost`, 11 PDFs with nothing above 0.6 MiB, has a p95/p50 of only 2.9×. So "how long does
indexing take" is answered by which large documents a corpus holds, not by its document count.

**Repeatability, from the two `kb-abac` rows:** chunk count identical (2349 both times, so
extraction is deterministic), wall within 6%, p95 within 8%, but **p50 within 33%** — a median over
38 samples is noisy. A later phase showing under ~10% movement on this corpus is therefore not
showing a signal. This is what the design wanted instead of a regression gate invented in advance.

**Migration is visible per row, and the first run of each corpus did one.** `kb-abac` came in at
schema 3 and `kb-gost` at **2**, against a current 5 — both working corpora were serving bodies
produced by older extractors, `kb-gost` by three versions. The repeat row reads `schema 5→5`: the
same corpus, different input conditions, which is exactly why the column exists.

**`kb-abac` indexed 38 documents against the 34 in its manifest.** The four uppercase `.PDF`
certificates that commit `3fafe9f` made walkable are in the index for the first time.

### DF ratios (`df(token)/n_chunks`, what the `verify` gate compares against)

| Corpus | Sample |
|---|---|
| `kb-abac` | `plc` 0.036 · `modbus` 0.063 · `iec` 0.049 |
| `kb-gost` | `стандарт` 0.185 · `метод` 0.121 · `требования` 0.179 · `испытания` 0.087 · `температура` 0.000 |
| generated | `zylathon` 0.166 · `qorvex` 0.034 · `brimtalve` 0.135 · `dunmorrow` 0.003 |

Two things to know before trusting a ratio here. A token must occur **as a whole token**: `gost`
sampled 0.000 on `kb-gost` because it appears only inside `www.gost.ru`-style tokens, and a
substring search is no evidence about a token-keyed table. And `температура` sampled 0.000 because
Russian inflects — the corpus carries `температуры`, not the base form. Pick the form that occurs;
the echoed ratio is the verification.
