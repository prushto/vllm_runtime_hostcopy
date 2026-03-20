#!/usr/bin/env python3
"""LDA correctness: same HF id for main and dormant, c=1, strict logits/KL checks.

Loads the checkpoint twice (two modules + split KV). With
``validate_same_checkpoint``, each forward must satisfy:
  - dormant logits ~= base logits
  - KL(dormant || base) below a small threshold

Run inside the vLLM dev container (GPU):

  EXPERIMENTS_DIR=$HOME/vllm_runtime_hostcopy ./docker/run_vllm_dev.sh

  # Qwen2.5-0.5B (default preset)
  python3 test/basic_func/test_lda_same_model_validate.py

  # Qwen2.5-7B (higher mem util, capped context for KV)
  python3 test/basic_func/test_lda_same_model_validate.py --preset 7b
  # or:
  python3 test/basic_func/test_lda_same_model_validate_7b.py

Overrides (apply on top of preset): ``--model``, ``--gpu-mem-util``, ``--max-model-len``.

Both presets default to ``gpu_memory_utilization=0.4`` so the engine leaves headroom for
**two** weight loads and a **50/50 KV split** (see ``test_lda_generate.py`` / ``test/speed_checks/config.yaml``).
Raising it (e.g. 0.9) often OOMs on LDA when allocating the dormant KV pool.
"""
from __future__ import annotations

import argparse
from typing import Any

from vllm import LLM, SamplingParams

# Preset: model id, gpu_memory_utilization, max_model_len (None = engine default),
# and extra lda validation tuning (bf16 / two independent loads).
_PRESETS: dict[str, dict[str, Any]] = {
    "0.5b": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        # Match LDA smoke tests (split KV for main + dormant).
        "gpu_mem_util": 0.4,
        "max_model_len": None,
        "lda_validation": {
            "validation_kl_max": 5e-3,
            "validation_logits_rtol": 0.05,
            "validation_logits_atol": 0.5,
        },
    },
    "7b": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        # Same as test_lda_generate.py / speed_checks: cap vLLM pool so two
        # weight sets + split KV + cudagraphs fit (high util OOMs on ~120GB).
        "gpu_mem_util": 0.4,
        # Cuts KV footprint for same-checkpoint smoke on single-GPU setups.
        "max_model_len": 8192,
        "lda_validation": {
            "validation_kl_max": 1e-2,
            "validation_logits_rtol": 0.08,
            "validation_logits_atol": 1.0,
        },
    },
}

TEST_MESSAGES = [{"role": "user", "content": "Say hello in one word."}]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preset",
        choices=sorted(_PRESETS.keys()),
        default="0.5b",
        help="0.5b = Qwen2.5-0.5B-Instruct; 7b = Qwen2.5-7B-Instruct",
    )
    p.add_argument("--model", default=None, help="Override preset model id")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument(
        "--gpu-mem-util",
        type=float,
        default=None,
        help="Override preset gpu_memory_utilization",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override preset max_model_len (e.g. 4096 if 8192 OOMs)",
    )
    p.add_argument("--max-tokens", type=int, default=32)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = dict(_PRESETS[args.preset])
    model = args.model or cfg["model"]
    gpu_mem = args.gpu_mem_util if args.gpu_mem_util is not None else cfg["gpu_mem_util"]
    max_model_len = (
        args.max_model_len if args.max_model_len is not None else cfg["max_model_len"]
    )
    lda_validation = dict(cfg["lda_validation"])

    print(
        "LDA same-checkpoint validation:",
        {
            "preset": args.preset,
            "model": model,
            "dormant_model": model,
            "alpha": args.alpha,
            "gpu_memory_utilization": gpu_mem,
            "max_model_len": max_model_len,
        },
    )

    llm_kwargs: dict[str, Any] = dict(
        model=model,
        dormant_model=model,
        lda_alpha=args.alpha,
        gpu_memory_utilization=gpu_mem,
        additional_config={
            "lda": {
                "validate_same_checkpoint": True,
                **lda_validation,
            }
        },
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outputs = llm.chat(TEST_MESSAGES, sampling_params=sampling_params)
    text = outputs[0].outputs[0].text
    print(f"Generated: {text!r}")
    print("OK: same-checkpoint LDA validation passed for all forward steps.")


if __name__ == "__main__":
    main()
