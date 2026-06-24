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

## Where we are / where to go

- **Now:** glossa + Qwen3.5-4B at EM 0.68 / F1 0.81 — within reach of supervised SOTA-class (~69/~82),
  driven by near-perfect retrieval (0.99). Engine claim validated on the distractor sample.
- **Next milestones:**
  1. **Claude arm** (`--backend cli`) on the same questions — isolate reader vs engine (A/B).
  2. **Larger N** (200–500) for a stable point estimate; confirm 50-q numbers hold.
  3. **fullwiki** — the real retrieval test (Recall@k over all of Wikipedia, not 10 distractors).
  4. **Graph A/B** (graph OFF vs ON) once a graph-off query mode exists — the multihop claim.
- Caveat: these benches are English-Wikipedia clean text; they validate the multihop/retrieval engine,
  not glossa's product edge (office/pdf/Russian/offline/agentic graph).
