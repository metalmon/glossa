# Build fine-tuning datasets from your knowledge graph

Turn your reasoning graph into **SFT** and **DPO** datasets for [Unsloth](https://unsloth.ai/), so you can fine-tune a
small local model to answer questions over your corpus. Two complementary sources:

- **Teacher distillation** — a strong model invents questions from the graph and answers them; you train the small
  model to imitate those high-quality trajectories. Best source of **SFT** data.
- **On-policy capture** — run the small model you intend to fine-tune, capture its own trajectories, and label them by
  the grader. Best source of **DPO** preference pairs (its correct vs wrong answers to the same question).

The reward signal for both is the graded **judge** verdict (Correct / Partial / Wrong) — see
[evidence-grounded judging](eval-and-training.md). Both paths write the same Unsloth-ready JSONL.

> Prerequisites: a built, cleaned reasoning graph (see [graph-lifecycle.md](graph-lifecycle.md)), a `dataset.toml`, and
> endpoints configured in `lab.toml`.

---

## Path A — teacher distillation (strong model → SFT)

Point both `[distil]` and `[model]` at a **strong** model in `lab.toml` for this pass (in a local-first setup they
often default to the small model — switch them to your strong/cloud endpoint here).

```bash
# 1. Generate synthetic (question, answer) golds FROM the graph. The strong [distil] model walks
#    grounded seeds, proposes each Q/A via `propose_gold`, self-gates, and drops any question that
#    is answerable without the reasoning chain (adversarial leak check).
kbx distil --emit-golds teacher-qa.toml

# 2. Let the strong model SOLVE those questions over the graph, capturing its full trajectories.
kbx eval --dataset teacher-qa.toml --tag teacher --capture

# 3. Export the correct teacher trajectories as an SFT dataset.
kbx export --from teacher --format sft --out teacher-sft.jsonl
```

`teacher-sft.jsonl` is now a supervised set of strong-model demonstrations grounded in your own documents.

---

## Path B — on-policy capture (small model → SFT + DPO)

Point `[model]` at the **small** model you will fine-tune. Run each case several times so the (stochastic) reader
produces varied trajectories — some correct, some wrong — which is what DPO needs.

```bash
# Capture N trajectories per question from the small reader.
kbx eval --tag onpolicy --capture --samples 4

# SFT from the trajectories the judge marked correct:
kbx export --from onpolicy --format sft --out onpolicy-sft.jsonl

# DPO pairs — a correct trajectory (chosen) vs a wrong one (rejected) for the same question:
kbx export --from onpolicy --format dpo --out onpolicy-dpo.jsonl
```

You can pass several runs to `--from` (comma-separated) to pool more samples.

---

## Dataset formats (Unsloth-ready)

- **SFT** (`--format sft`) — one JSON line per kept trajectory. Default `--shape messages` (ChatML / OpenAI, the
  Hugging Face default):

  ```json
  {"messages": [
    {"role": "system", "content": "…"},
    {"role": "user", "content": "…"},
    {"role": "assistant", "tool_calls": [ … ]},
    {"role": "tool", "content": "…"},
    {"role": "assistant", "content": "<final answer>"}
  ]}
  ```

  Use `--shape sharegpt` for the `{"conversations": [{"from": "human", "value": …}]}` form (Unsloth
  `standardize_sharegpt`). `--include-partial` also keeps Partial trajectories.

- **DPO** (`--format dpo`) — one JSON line per preference pair, TRL/Unsloth `DPOTrainer` columns:

  ```json
  {"prompt":   [{"role": "system", "content": "…"}, {"role": "user", "content": "…"}],
   "chosen":   [{"role": "assistant", "content": "<correct answer>"}],
   "rejected": [{"role": "assistant", "content": "<wrong answer>"}]}
  ```

  One best-correct-vs-wrong pair per question by default; `--max-pairs N` raises the cap.

---

## Fine-tune in Unsloth

Load the JSONL as a Hugging Face dataset and train — the messages/ShareGPT form goes through `apply_chat_template`
(or `standardize_sharegpt`), and the DPO form through `DPOTrainer`. See the
[Unsloth datasets guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide.md) and
[TRL DPO trainer](https://huggingface.co/docs/trl/dpo_trainer). Unsloth Studio (desktop) accepts the same files.

**Rule of thumb:** SFT teaches the model *how* to answer (train mostly on the teacher set); DPO sharpens *which* answer
it prefers (train on the small model's own correct-vs-wrong pairs). Combining both — SFT first, then DPO — usually
beats either alone.

---

## See also

- [graph-lifecycle.md](graph-lifecycle.md) — build and clean the reasoning graph the questions are grounded in.
- [eval-and-training.md](eval-and-training.md) — the `kbx` pipeline, the judge, and GEPA prompt optimization.
