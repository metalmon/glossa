"""Local DICL optimization cycle for the `answer_hotpot` TensorZero function.

Fully local: LM Studio embeddings (lmstudio_embed) + Qwen (qwen). DICL builds a `dicl` variant that, at
inference time, retrieves k nearest GOOD past examples (curated by feedback) and injects them as context —
no model fine-tuning, no external provider.

PREREQS:
  1. `answer_hotpot` episodes already banked in ClickHouse (run `kb-eval ... --backend tensorzero ...`).
  2. The gateway restarted AFTER `lmstudio_embed` was added to tensorzero.toml:
        docker compose -f eval/tensorzero/docker-compose.yml restart gateway
  3. `pip install tensorzero` (done).

NOTE: list_inferences / train_samples API shapes are pinned at runtime — the script prints diagnostics
and degrades gracefully so a first run reveals any signature mismatch to fix.
"""
import sys, time, inspect

GATEWAY = "http://localhost:3000"
FUNCTION = "answer_hotpot"

try:
    from tensorzero import TensorZeroGateway, DICLOptimizationConfig
except Exception as e:
    print("import error:", e); sys.exit(1)


def main():
    with TensorZeroGateway.build_http(gateway_url=GATEWAY) as t0:
        # 1) Pull banked inferences for the function (the example pool). Print the API shape first.
        print("list_inferences signature:", list(inspect.signature(t0.list_inferences).parameters))
        try:
            samples = t0.list_inferences(function_name=FUNCTION)
        except TypeError:
            samples = t0.list_inferences(FUNCTION)
        print(f"banked inferences for {FUNCTION}: {len(samples)}")
        if not samples:
            print("no banked inferences — run a `--backend tensorzero` eval first"); return

        # 2) Launch DICL → creates/append a `dicl` variant from the good examples.
        cfg = DICLOptimizationConfig(
            function_name=FUNCTION,
            variant_name="dicl",
            embedding_model="lmstudio_embed",
            model="qwen",
            k=5,
            append_to_existing_variants=True,
        )
        print("launching DICL optimization...")
        handle = t0.experimental_launch_optimization(train_samples=samples, optimization_config=cfg)

        # 3) Poll to completion.
        for _ in range(360):  # up to ~1h
            st = t0.experimental_poll_optimization(handle)
            print("poll:", st)
            s = str(st).lower()
            if "complet" in s or "fail" in s or "error" in s:
                break
            time.sleep(10)
        print("DICL cycle finished. If completed, the `dicl` variant is now in the gateway "
              "(verify in the UI / config) — then A/B it vs `baseline` by re-running the eval.")


if __name__ == "__main__":
    main()
