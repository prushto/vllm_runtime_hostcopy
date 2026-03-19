#!/usr/bin/env python3
"""Memory gates for full-dormant vs delta-dormant LDA."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LDA memory gate checks.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--dormant-model", required=True)
    p.add_argument("--dormant-delta-dir", required=True)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--poll-interval-s", type=float, default=0.2)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def _query_gpu_used_mb(gpu_index: int) -> int:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    if gpu_index < 0 or gpu_index >= len(out):
        raise ValueError(f"gpu-index {gpu_index} out of range for {len(out)} GPUs")
    return int(out[gpu_index].strip())


def _build_llm_kwargs(
    *,
    base_model: str,
    dormant_model: str,
    dormant_delta_dir: str | None,
    max_model_len: int | None,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": base_model,
        "dormant_model": dormant_model,
        "lda_alpha": 0.5,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = int(max_model_len)
    if dormant_delta_dir is not None:
        kwargs["dormant_delta_dir"] = dormant_delta_dir
        kwargs["dormant_base_model"] = base_model
    return kwargs


def _run_scenario(
    *,
    llm_kwargs: dict[str, Any],
    gpu_index: int,
    poll_interval_s: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    baseline_used = _query_gpu_used_mb(gpu_index)
    peak_used = baseline_used
    stop = threading.Event()

    def sampler() -> None:
        nonlocal peak_used
        while not stop.is_set():
            try:
                used = _query_gpu_used_mb(gpu_index)
                if used > peak_used:
                    peak_used = used
            except Exception:
                pass
            time.sleep(poll_interval_s)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()

    llm = LLM(**llm_kwargs)
    init_used = _query_gpu_used_mb(gpu_index)
    init_peak_used = peak_used

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=max_new_tokens,
    )
    prompts = [[{"role": "user", "content": "Write one short sentence about trees."}]]
    _ = llm.chat(prompts, sampling_params=sampling, use_tqdm=False)
    time.sleep(max(0.1, poll_interval_s))
    steady_used = _query_gpu_used_mb(gpu_index)
    peak_used_final = peak_used

    stop.set()
    t.join(timeout=max(1.0, 3 * poll_interval_s))
    del llm
    time.sleep(1.0)

    return {
        "baseline_used_mb": baseline_used,
        "init_used_mb": init_used,
        "init_peak_used_mb": init_peak_used,
        "steady_used_mb": steady_used,
        "overall_peak_used_mb": peak_used_final,
    }


def main() -> None:
    args = parse_args()
    full_kwargs = _build_llm_kwargs(
        base_model=args.base_model,
        dormant_model=args.dormant_model,
        dormant_delta_dir=None,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    delta_kwargs = _build_llm_kwargs(
        base_model=args.base_model,
        dormant_model=args.dormant_model,
        dormant_delta_dir=args.dormant_delta_dir,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    full = _run_scenario(
        llm_kwargs=full_kwargs,
        gpu_index=args.gpu_index,
        poll_interval_s=args.poll_interval_s,
        max_new_tokens=args.max_new_tokens,
    )
    delta = _run_scenario(
        llm_kwargs=delta_kwargs,
        gpu_index=args.gpu_index,
        poll_interval_s=args.poll_interval_s,
        max_new_tokens=args.max_new_tokens,
    )

    init_peak_delta = delta["init_peak_used_mb"] - full["init_peak_used_mb"]
    steady_delta = delta["steady_used_mb"] - full["steady_used_mb"]
    result = {
        "status": "ok" if init_peak_delta < 0 and steady_delta < 0 else "error",
        "full_dormant": full,
        "delta_dormant": delta,
        "delta_minus_full_mb": {
            "init_peak_mb": init_peak_delta,
            "steady_mb": steady_delta,
        },
        "gates": {
            "init_peak_lower": init_peak_delta < 0,
            "steady_lower": steady_delta < 0,
        },
    }
    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

