#!/usr/bin/env python3
"""Minimal LDA test: same model as main and dormant, alpha=0.5, one chat completion.

Validates that the LDA runner loads two models, blends logits, and produces output.
Uses the same checkpoint for both (main and dormant) so no second download.
Recommend gpu_memory_utilization=0.4 so the single KV pool is split 50/50.

Run inside the vLLM dev container (with repo mounted as workspace):

  EXPERIMENTS_DIR=$HOME/vllm_runtime_hostcopy ./docker/run_vllm_dev.sh
  python test/test_lda_generate.py
"""
from __future__ import annotations

import argparse

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DORMANT = "jane-street/dormant-model-warmup"
TEST_MESSAGES = [{"role": "user", "content": "What's 2 + 2?"}]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dormant-model", default=DORMANT)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--gpu-mem-util", type=float, default=0.4)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional override for faster debugging (e.g. 2048/4096).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        "Loading LDA with settings:",
        {
            "model": args.model,
            "dormant_model": args.dormant_model,
            "alpha": args.alpha,
            "gpu_mem_util": args.gpu_mem_util,
            "max_model_len": args.max_model_len,
        },
    )

    llm_kwargs = dict(
        model=args.model,
        dormant_model=args.dormant_model,
        lda_alpha=args.alpha,
        gpu_memory_utilization=args.gpu_mem_util,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(
        **llm_kwargs,
    )

    sampling_params = SamplingParams(temperature=0.7, max_tokens=128)
    print("Generating with chat API...")

    outputs = llm.chat(TEST_MESSAGES, sampling_params=sampling_params)

    result = outputs[0]
    text = result.outputs[0].text
    n_tokens = len(result.outputs[0].token_ids)
    print(f"Generated ({n_tokens} tokens): {text!r}")
    print("OK: LDA runner produced output.")


if __name__ == "__main__":
    main()
