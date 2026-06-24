# glossa — benchmark log & progress toward SOTA

Curated record of agent-eval runs (the `kb-eval` harness). Raw `eval-*.json` reports are git-ignored;
this file is the durable, human-readable history. One row per meaningful run. Append, don't rewrite.

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
  not glossa's product edge (office/pdf/Russian/offline/agentic graph).
