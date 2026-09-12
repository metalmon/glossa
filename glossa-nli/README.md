# glossa-nli

In-process NLI (Natural Language Inference) entailment scorer over an ONNX
3-way sequence-classification model, used by glossa's answer-grounding verifier:
given a retrieved chunk (premise) and an answer claim (hypothesis), it scores
`P(entailment)` to check whether the answer is supported by its sources.

Standalone crate — it does **not** depend on `glossa`. `glossa` owns the
`NliScorer` trait and implements it for this crate's `InProcessNli` (see
glossa's `src/gate/nli_engine.rs`), which keeps the dependency graph acyclic.

## Enabling it

The engine is behind glossa's opt-in `nli` feature (off by default; the default
build compiles zero ML dependencies):

```bash
cargo build -p kb-eval --features nli      # builds kbx with the NLI engine
```

Inference uses the ONNX Runtime loaded **at runtime** from a shared library
(`ort` `load-dynamic`), so point `ORT_DYLIB_PATH` at an `onnxruntime` dylib whose
major version is ABI-compatible with the pinned `ort` release:

```bash
export ORT_DYLIB_PATH=/path/to/onnxruntime.dll   # or libonnxruntime.so
```

## Bring your own model

Any Hugging Face `*ForSequenceClassification` model with a **3-way NLI head**
works, exported to ONNX. The graph must expose the standard BERT-family I/O:

| tensor | dtype | shape |
|:--|:--|:--|
| `input_ids` / `attention_mask` / `token_type_ids` | int64 | `[batch, seq]` |
| `logits` | float32 | `[batch, 3]` |

**Entailment index.** The three logits are `[entailment, neutral, contradiction]`
in *some* order that depends on the model. Read `id2label` from the model's
`config.json` and pass the entailment index to `[verify.nli].entail_index`
(via `kbx nli set --entail-index N`). It is **not** always 0 — MNLI models often
put entailment last.

### Exporting a model to ONNX (one-time, local tooling)

Either `optimum` (when its ONNX exporter is available for your Python), or a
minimal `torch.onnx` script — the point is a graph with the I/O above and
dynamic `batch`/`seq` axes:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SRC = "<hf-model-id-or-local-snapshot>"
OUT = "model.onnx"
tok = AutoTokenizer.from_pretrained(SRC)
model = AutoModelForSequenceClassification.from_pretrained(SRC).eval()

class LogitsOnly(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids).logits

enc = tok("a", "b", return_tensors="pt")
ids, am = enc["input_ids"].long(), enc["attention_mask"].long()
tt = enc.get("token_type_ids", torch.zeros_like(ids)).long()
dyn = {n: {0: "batch", 1: "seq"} for n in ("input_ids", "attention_mask", "token_type_ids")}
dyn["logits"] = {0: "batch"}
torch.onnx.export(
    LogitsOnly(model).eval(), (ids, am, tt), OUT,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"], dynamic_axes=dyn, opset_version=14,
)
```

Copy `tokenizer.json` (+ `config.json`) from the source snapshot next to
`model.onnx`. This is local tooling, not shipped code.

## Runtime workflow (`kbx nli`)

```bash
# 1. fetch a model dir (model.onnx + tokenizer.json + config.json) from a HF repo
kbx nli download --repo <org/model-onnx> --to <model_dir>

# 2. register it in the corpus config ([verify.nli] in ontology.toml)
kbx nli set <corpus> --model-dir <model_dir> --entail-index <N>

# 3. check readiness (feature built? model present? ORT dylib? loads + runs?)
kbx nli check <corpus>

# 4. calibrate the verify mode/thresholds from a scored run (ac vs nli vs combined)
kbx eval calibrate --run <tag> --max-error 0.2 --write
```

`kbx nli check` is the diagnostic for the fail-open design: if anything is
missing (feature not built, no model, dylib unset, load failure) the verifier
silently falls back to lexical-only (AC) grounding, and `check` tells you which.

## Batching (`GLOSSA_NLI_BATCH_TOKENS`)

`entail` can batch its `(window × hypothesis)` rows into fewer `session.run`
calls. On the **CPU** execution provider this is a net loss — the rows carry the
same total FLOPs and the larger tensor costs more in cache/allocation than the
per-call overhead it saves — so the default budget is one row per batch
(effectively sequential, no regression). Raise it for a GPU build where a single
kernel over the batch is a real win:

```bash
export GLOSSA_NLI_BATCH_TOKENS=16384
```

The batched path is parity-verified (identical scores to sequential) at any
budget; only the default is tuned for CPU.
