#!/usr/bin/env python3
"""Parity checks for full-dormant LDA vs base+delta LDA.

This script is intentionally small and deterministic to validate correctness
for a fixed prompt set before broader throughput tests.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Row:
    prompt_id: str
    prompt_text: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LDA delta parity checks.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--dormant-model", required=True)
    p.add_argument("--dormant-delta-dir", required=True)
    p.add_argument("--prompts-csv", type=Path, default=Path(__file__).parent / "prompts.csv")
    p.add_argument("--max-prompts", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def load_prompts(csv_path: Path, max_prompts: int) -> list[Row]:
    import csv

    rows: list[Row] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            rows.append(Row(prompt_id=rec["prompt_id"], prompt_text=rec["prompt_text"]))
            if len(rows) >= max_prompts:
                break
    if not rows:
        raise ValueError(f"No prompts found in {csv_path}")
    return rows


def build_llm_kwargs(
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
        # Keep parity checks deterministic and avoid cudagraph/compile drift.
        "enforce_eager": True,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = int(max_model_len)
    if dormant_delta_dir is not None:
        kwargs["dormant_delta_dir"] = dormant_delta_dir
        kwargs["dormant_base_model"] = base_model
    return kwargs


def run_scenario(llm_kwargs: dict[str, Any], prompts: list[Row], max_new_tokens: int) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=max_new_tokens,
        logprobs=1,
    )
    llm = LLM(**llm_kwargs)
    try:
        messages = [[{"role": "user", "content": r.prompt_text}] for r in prompts]
        outputs = llm.chat(messages, sampling_params=sampling, use_tqdm=False)
        rows: list[dict[str, Any]] = []
        for p, out in zip(prompts, outputs):
            seq = out.outputs[0]
            rows.append(
                {
                    "prompt_id": p.prompt_id,
                    "token_ids": list(seq.token_ids),
                    "text": seq.text,
                    "cumulative_logprob": float(seq.cumulative_logprob),
                }
            )
        return rows
    finally:
        del llm


def compare_rows(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    b_map = {r["prompt_id"]: r for r in baseline}
    c_map = {r["prompt_id"]: r for r in candidate}
    all_ids = sorted(set(b_map) | set(c_map))

    mismatches: list[dict[str, Any]] = []
    for pid in all_ids:
        br = b_map.get(pid)
        cr = c_map.get(pid)
        if br is None or cr is None:
            mismatches.append({"prompt_id": pid, "reason": "missing_in_one_scenario"})
            continue
        if br["token_ids"] != cr["token_ids"]:
            mismatches.append(
                {
                    "prompt_id": pid,
                    "reason": "token_ids_mismatch",
                    "baseline_len": len(br["token_ids"]),
                    "candidate_len": len(cr["token_ids"]),
                }
            )
            continue
        # Secondary signal for numeric parity.
        if abs(br["cumulative_logprob"] - cr["cumulative_logprob"]) > 1e-6:
            mismatches.append(
                {
                    "prompt_id": pid,
                    "reason": "cumulative_logprob_mismatch",
                    "baseline": br["cumulative_logprob"],
                    "candidate": cr["cumulative_logprob"],
                }
            )

    return {
        "status": "ok" if not mismatches else "error",
        "n_prompts": len(all_ids),
        "n_mismatches": len(mismatches),
        "mismatch_samples": mismatches[:20],
    }


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts_csv, args.max_prompts)

    baseline_kwargs = build_llm_kwargs(
        base_model=args.base_model,
        dormant_model=args.dormant_model,
        dormant_delta_dir=None,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    candidate_kwargs = build_llm_kwargs(
        base_model=args.base_model,
        dormant_model=args.dormant_model,
        dormant_delta_dir=args.dormant_delta_dir,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    baseline = run_scenario(baseline_kwargs, prompts, args.max_new_tokens)
    candidate = run_scenario(candidate_kwargs, prompts, args.max_new_tokens)
    result = compare_rows(baseline, candidate)
    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

