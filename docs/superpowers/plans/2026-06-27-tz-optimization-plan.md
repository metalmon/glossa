# TensorZero Optimization — Plan (Dev-Optimize, Prod-Without-TZ)

**Decision (2026-06-27):** TensorZero is a **dev-time** optimization/observability tool here;
**prod will NOT run TZ** (closed contour, local/offline model serving). Therefore we only
adopt optimization methods whose OUTPUT is a **portable static artifact** (a prompt or a model),
never methods that require TZ (or equivalent machinery) in the serving path. This mirrors the
graph-generalization decision ([2026-06-27-graph-generalization-no-embeddings.md](2026-06-27-graph-generalization-no-embeddings.md)):
**no runtime infra dependency in prod, offline, portable.**

## What TensorZero actually offers (nothing was removed — the UI just shows one class)
The dashboard exposes only **Fine Tuning** (cloud providers). The rest live in the **SDK /
recipes / config**, not the UI. Three classes:

1. **Model optimization (weights)** — UI "Fine Tuning":
   - SFT (supervised), DPO (preference), RLHF. Output: a model artifact/endpoint.
2. **Prompt optimization (automated prompt engineering)** — SDK/recipes, NOT UI:
   - MIPRO / MIPROv2, DSPy integration, GEPA. Output: a **static optimized system prompt**
     (instructions + selected few-shot demos).
3. **Inference-time optimization** — config variant types + SDK, NOT UI:
   - DICL (dynamic in-context learning), Best-of-N, Mixture-of-N, Chain-of-Thought. These are
     **runtime machinery**, not artifacts.

## Portability ranking (the only criterion that matters for no-TZ prod)
| Method | Output | Port to prod without TZ |
|---|---|---|
| Fine-tuning (SFT/DPO) | model | ✅ deploy the model |
| **MIPRO / GEPA** | **static prompt** | ✅ paste the prompt — zero runtime deps |
| DICL / Best-of-N / Mixture-of-N | runtime pipeline | ❌ needs TZ or a reimplemented serving layer |

DICL specifically = (example store of embeddings) + (embed → kNN retrieve → inject few-shot →
call). It does **not** export as an artifact, and it **requires an embeddings endpoint** —
which directly violates the repo's no-embeddings portability principle.

## Hard constraints discovered (scope the effort honestly)
- **Optimization needs a feedback metric.** We have one for the **`answer_hotpot`** eval
  (the harness POSTs `em` / `f1` / `retrieved` / `recall@k` / `mrr` / `judge`) — that is the
  optimization target. The whole plan is about `answer_hotpot` (KB-QA).
- **`answer_hotpot` is a multi-turn tool-using agent.** MIPRO/DICL are cleanest on single-call
  functions. For an agent, optimizing the **system prompt (MIPRO/GEPA)** is tractable; DICL
  injecting whole multi-turn **trajectories** as few-shot is awkward and token-heavy.

## Recommendation
1. **Primary: MIPRO / GEPA → static system prompt.** Best fit for "optimize in dev, deploy
   without TZ": the artifact is a string. No prod infra, deterministic, cheap at inference,
   no embeddings — consistent with the graph plan's portability stance. Works for the
   multi-turn agent (optimize instructions + optional fixed few-shot).
2. **Fallback: fine-tuning** — only if/when prompt-opt plateaus AND a local training path is
   acceptable (cloud SFT is out for the closed contour; local SFT is a separate effort).
3. **DICL — dev experiments only, NOT the prod path.** Reconsider only if (a) the mid-proxy
   already runs in prod, (b) fixed few-shot demonstrably underperforms on diverse inputs, and
   (c) an embedder + vector store in the closed contour is acceptable. Then DICL "ports" as
   *the example dataset + a retrieve-and-inject layer hosted in the proxy*, not as TZ.

## Data prerequisites (ClickHouse) to run any recipe
1. **Inferences** for `answer_hotpot` — captured automatically by the gateway. Nothing to do.
2. **Episode feedback** — the harness already POSTs it. Need it tied to enough episodes.
3. **Volume** — dozens–hundreds of episodes with feedback; current capture is likely too small.
   Run the eval over a larger slice (~100–300 questions) first.
4. **One optimization metric** — pick `judge` (best for free-form answers), or `em`/`f1`.
5. **(DICL only)** a "good example" threshold (e.g. `em=true` or `judge≥0.8`) + the embedding
   model (`lmstudio_embed`, already configured).

## Sequencing
1. Run `answer_hotpot` eval over a larger slice → accumulate (inference + feedback) in ClickHouse.
2. Launch **MIPRO** via SDK (`experimental_launch_optimization`) on `answer_hotpot`, metric `judge`.
3. Export the optimized prompt → this is the prod artifact (drop into the system prompt; no TZ).
4. Measure the delta vs `baseline` on held-out questions; keep only if it moves the metric.
5. Only if MIPRO plateaus: evaluate DICL-in-proxy (dev) or fine-tuning — per the constraints above.

## Constraints (summary)
- Prod runs **no TZ**; adopt only artifact-producing methods (prompt / model).
- **No embeddings dependency in prod** (portability) → prefer MIPRO over DICL.
- Optimization is gated on a feedback metric → applies to `answer_hotpot`.
- Measure each step; do not optimize blind.
