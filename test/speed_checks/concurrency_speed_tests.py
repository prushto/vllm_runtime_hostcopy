#!/usr/bin/env python3
"""Offline concurrency speed checks for vanilla vLLM vs LDA.

This script benchmarks two scenarios on the same prompt set:
1) Vanilla vLLM (single model).
2) LDA vLLM (base + dormant model).

It measures model load time separately from generation time, and writes both
human-readable and machine-readable reports under test_results/.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"
DEFAULT_PROMPTS_CSV = ROOT / "prompts.csv"
DEFAULT_OUTPUT_DIR = ROOT / "test_results"


@dataclass
class PromptRow:
    prompt_id: str
    length_bucket: str
    prompt_text: str


@dataclass
class RepeatResult:
    scenario: str
    concurrency: int
    repeat_index: int
    n_prompts: int
    load_time_s: float
    warmup_time_s: float | None
    generation_time_s: float
    total_output_tokens: int
    total_input_tokens: int
    prompts_per_s: float
    output_tokens_per_s: float
    ms_per_output_token: float
    end_to_end_time_s: float
    status: str = "ok"
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline concurrency speed tests for vanilla vs LDA vLLM."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prompts-csv", type=Path, default=DEFAULT_PROMPTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--concurrencies", type=int, nargs="*", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--generate-prompts-only",
        action="store_true",
        help="Only generate prompts CSV then exit.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict-like yaml config, got: {type(data).__name__}")
    return data


def _build_bucket_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    bucket_order = ["short", "medium", "long"]
    floats = {k: total * float(ratios.get(k, 0.0)) for k in bucket_order}
    counts = {k: int(floats[k]) for k in bucket_order}
    assigned = sum(counts.values())
    leftovers = total - assigned
    remainders = sorted(
        bucket_order,
        key=lambda k: floats[k] - counts[k],
        reverse=True,
    )
    for i in range(leftovers):
        counts[remainders[i % len(remainders)]] += 1
    return counts


def _short_prompt(topic: str, task: str, idx: int) -> str:
    return f"{task} about {topic}. Keep your answer concise. (item {idx})"


def _medium_prompt(topic: str, style: str, idx: int) -> str:
    return (
        f"You are helping with a practical task on {topic}. "
        f"Provide a {style} response with a short rationale and 3 actionable steps. "
        f"Avoid unnecessary filler and keep the structure clear. (item {idx})"
    )


def _long_prompt(topic: str, domain: str, idx: int) -> str:
    return (
        f"I am evaluating trade-offs in {topic} for a {domain} setting and need a "
        "rigorous response. Please provide: (1) a brief problem framing, (2) two "
        "alternative approaches with pros/cons, (3) failure modes and mitigations, "
        "(4) a recommendation with explicit assumptions. Keep the answer practical "
        "and structured with bullet points. Include at least one concrete example "
        f"and one cautionary note. (item {idx})"
    )


def generate_mixed_prompts_csv(
    out_csv: Path,
    n_prompts: int,
    seed: int,
    length_mix: dict[str, float],
) -> None:
    rng = random.Random(seed)
    topics = [
        "distributed systems",
        "prompt engineering",
        "risk analysis",
        "code review quality",
        "API design",
        "data cleaning",
        "model evaluation",
        "A/B testing",
        "SQL optimization",
        "incident response",
        "GPU memory planning",
        "batch scheduling",
        "latency profiling",
        "documentation strategy",
        "security hardening",
        "model deployment",
    ]
    tasks = [
        "Give a one-sentence summary",
        "List two key points",
        "Provide one practical tip",
        "State the main trade-off",
        "Explain the core idea",
        "Highlight a likely pitfall",
    ]
    styles = [
        "balanced",
        "implementation-focused",
        "risk-aware",
        "decision-oriented",
        "operations-focused",
    ]
    domains = [
        "production inference",
        "research prototyping",
        "enterprise platform",
        "small engineering team",
        "high-throughput backend",
    ]

    bucket_counts = _build_bucket_counts(n_prompts, length_mix)
    rows: list[PromptRow] = []

    idx = 1
    for _ in range(bucket_counts["short"]):
        topic = rng.choice(topics)
        task = rng.choice(tasks)
        rows.append(
            PromptRow(
                prompt_id=f"p{idx:04d}",
                length_bucket="short",
                prompt_text=_short_prompt(topic, task, idx),
            )
        )
        idx += 1

    for _ in range(bucket_counts["medium"]):
        topic = rng.choice(topics)
        style = rng.choice(styles)
        rows.append(
            PromptRow(
                prompt_id=f"p{idx:04d}",
                length_bucket="medium",
                prompt_text=_medium_prompt(topic, style, idx),
            )
        )
        idx += 1

    for _ in range(bucket_counts["long"]):
        topic = rng.choice(topics)
        domain = rng.choice(domains)
        rows.append(
            PromptRow(
                prompt_id=f"p{idx:04d}",
                length_bucket="long",
                prompt_text=_long_prompt(topic, domain, idx),
            )
        )
        idx += 1

    rng.shuffle(rows)
    # Re-label IDs after shuffle to keep deterministic contiguous ordering.
    for i, row in enumerate(rows, start=1):
        row.prompt_id = f"p{i:04d}"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "length_bucket", "prompt_text"])
        for row in rows:
            writer.writerow([row.prompt_id, row.length_bucket, row.prompt_text])


def load_prompts(csv_path: Path, limit_prompts: int | None = None) -> list[PromptRow]:
    rows: list[PromptRow] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                PromptRow(
                    prompt_id=row["prompt_id"],
                    length_bucket=row["length_bucket"],
                    prompt_text=row["prompt_text"],
                )
            )
    if limit_prompts is not None:
        rows = rows[:limit_prompts]
    return rows


def _sampling_params_from_config(gen_config: dict[str, Any]):
    # Import lazily so --generate-prompts-only does not require vLLM runtime.
    from vllm import SamplingParams

    do_sample = bool(gen_config.get("do_sample", True))
    temperature = float(gen_config.get("temperature", 0.7))
    top_p = float(gen_config.get("top_p", 1.0))
    top_k = int(gen_config.get("top_k", 0))
    if not do_sample:
        # vLLM SamplingParams in this version does not support `do_sample`.
        # Greedy behavior is represented via zero temperature.
        temperature = 0.0
        top_p = 1.0
        top_k = 0

    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=float(gen_config.get("repetition_penalty", 1.0)),
        max_tokens=int(gen_config.get("max_new_tokens", 60)),
    )


def _build_llm_kwargs(
    scenario: str,
    cfg: dict[str, Any],
    max_model_len_override: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": cfg["base_model_id"],
        "gpu_memory_utilization": float(cfg.get("gpu_memory_utilization", 0.4)),
    }
    max_model_len = (
        max_model_len_override
        if max_model_len_override is not None
        else cfg.get("max_model_len", None)
    )
    if max_model_len is not None:
        kwargs["max_model_len"] = int(max_model_len)
    if scenario == "lda":
        kwargs["dormant_model"] = cfg["dormant_model_id"]
        kwargs["lda_alpha"] = float(cfg.get("lda_alpha", 0.5))
        dormant_delta_dir = cfg.get("dormant_delta_dir")
        if dormant_delta_dir:
            kwargs["dormant_delta_dir"] = str(dormant_delta_dir)
        dormant_base_model = cfg.get("dormant_base_model")
        if dormant_base_model:
            kwargs["dormant_base_model"] = str(dormant_base_model)
    return kwargs


def _messages_from_prompts(prompts: list[PromptRow]) -> list[list[dict[str, str]]]:
    return [[{"role": "user", "content": row.prompt_text}] for row in prompts]


def _run_one_dataset_pass(
    llm: Any,
    messages: list[list[dict[str, str]]],
    concurrency: int,
    sampling_params: Any,
) -> tuple[float, int, int]:
    t0 = time.perf_counter()
    total_output_tokens = 0
    total_input_tokens = 0
    for i in range(0, len(messages), concurrency):
        batch = messages[i : i + concurrency]
        outputs = llm.chat(batch, sampling_params=sampling_params, use_tqdm=False)
        for out in outputs:
            if out.outputs:
                total_output_tokens += len(out.outputs[0].token_ids)
            prompt_token_ids = getattr(out, "prompt_token_ids", None)
            if prompt_token_ids is not None:
                total_input_tokens += len(prompt_token_ids)
    elapsed = time.perf_counter() - t0
    return elapsed, total_output_tokens, total_input_tokens


def run_scenario(
    scenario: str,
    cfg: dict[str, Any],
    prompts: list[PromptRow],
    concurrencies: list[int],
    repeats: int,
    warmup: bool,
    max_model_len_override: int | None,
    continue_on_error: bool,
) -> tuple[float, list[RepeatResult]]:
    from vllm import LLM

    llm_kwargs = _build_llm_kwargs(scenario, cfg, max_model_len_override)
    t_load = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_time_s = time.perf_counter() - t_load

    sampling_params = _sampling_params_from_config(cfg["gen_config"])
    messages = _messages_from_prompts(prompts)
    all_results: list[RepeatResult] = []

    try:
        for concurrency in concurrencies:
            warmup_time_s: float | None = None
            if warmup:
                warm_batch = messages[: max(1, min(concurrency, len(messages)))]
                t_warmup0 = time.perf_counter()
                llm.chat(warm_batch, sampling_params=sampling_params, use_tqdm=False)
                warmup_time_s = time.perf_counter() - t_warmup0

            for repeat_idx in range(1, repeats + 1):
                try:
                    gen_time_s, out_tokens, in_tokens = _run_one_dataset_pass(
                        llm=llm,
                        messages=messages,
                        concurrency=concurrency,
                        sampling_params=sampling_params,
                    )
                    prompts_per_s = len(messages) / gen_time_s if gen_time_s > 0 else 0.0
                    out_tokens_per_s = out_tokens / gen_time_s if gen_time_s > 0 else 0.0
                    ms_per_out_token = (
                        (gen_time_s * 1000.0) / out_tokens if out_tokens > 0 else float("inf")
                    )
                    all_results.append(
                        RepeatResult(
                            scenario=scenario,
                            concurrency=concurrency,
                            repeat_index=repeat_idx,
                            n_prompts=len(messages),
                            load_time_s=load_time_s,
                            warmup_time_s=warmup_time_s,
                            generation_time_s=gen_time_s,
                            total_output_tokens=out_tokens,
                            total_input_tokens=in_tokens,
                            prompts_per_s=prompts_per_s,
                            output_tokens_per_s=out_tokens_per_s,
                            ms_per_output_token=ms_per_out_token,
                            end_to_end_time_s=load_time_s + gen_time_s,
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    rr = RepeatResult(
                        scenario=scenario,
                        concurrency=concurrency,
                        repeat_index=repeat_idx,
                        n_prompts=len(messages),
                        load_time_s=load_time_s,
                        warmup_time_s=warmup_time_s,
                        generation_time_s=0.0,
                        total_output_tokens=0,
                        total_input_tokens=0,
                        prompts_per_s=0.0,
                        output_tokens_per_s=0.0,
                        ms_per_output_token=float("inf"),
                        end_to_end_time_s=load_time_s,
                        status="error",
                        error=str(e),
                    )
                    all_results.append(rr)
                    if not continue_on_error:
                        raise
    finally:
        # Explicitly drop model at end of scenario before next one.
        del llm

    return load_time_s, all_results


def _aggregate(results: list[RepeatResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[RepeatResult]] = {}
    for r in results:
        if r.status != "ok":
            continue
        grouped.setdefault((r.scenario, r.concurrency), []).append(r)

    rows: list[dict[str, Any]] = []
    for (scenario, concurrency), g in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        def avg(vals: list[float]) -> float:
            return statistics.mean(vals) if vals else 0.0

        def std(vals: list[float]) -> float:
            return statistics.stdev(vals) if len(vals) > 1 else 0.0

        gen_times = [x.generation_time_s for x in g]
        out_tps = [x.output_tokens_per_s for x in g]
        pps = [x.prompts_per_s for x in g]
        ms_per_tok = [x.ms_per_output_token for x in g]
        rows.append(
            {
                "scenario": scenario,
                "concurrency": concurrency,
                "repeats_ok": len(g),
                "generation_time_s_mean": avg(gen_times),
                "generation_time_s_std": std(gen_times),
                "prompts_per_s_mean": avg(pps),
                "output_tokens_per_s_mean": avg(out_tps),
                "ms_per_output_token_mean": avg(ms_per_tok),
            }
        )
    return rows


def _write_csv(
    output_csv: Path,
    repeat_results: list[RepeatResult],
    aggregate_rows: list[dict[str, Any]],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row_type",
                "scenario",
                "concurrency",
                "repeat_index",
                "status",
                "error",
                "n_prompts",
                "load_time_s",
                "warmup_time_s",
                "generation_time_s",
                "total_input_tokens",
                "total_output_tokens",
                "prompts_per_s",
                "output_tokens_per_s",
                "ms_per_output_token",
                "end_to_end_time_s",
            ]
        )
        for r in repeat_results:
            writer.writerow(
                [
                    "repeat",
                    r.scenario,
                    r.concurrency,
                    r.repeat_index,
                    r.status,
                    r.error,
                    r.n_prompts,
                    f"{r.load_time_s:.6f}",
                    "" if r.warmup_time_s is None else f"{r.warmup_time_s:.6f}",
                    f"{r.generation_time_s:.6f}",
                    r.total_input_tokens,
                    r.total_output_tokens,
                    f"{r.prompts_per_s:.6f}",
                    f"{r.output_tokens_per_s:.6f}",
                    f"{r.ms_per_output_token:.6f}",
                    f"{r.end_to_end_time_s:.6f}",
                ]
            )

        writer.writerow([])
        writer.writerow(
            [
                "row_type",
                "scenario",
                "concurrency",
                "repeats_ok",
                "generation_time_s_mean",
                "generation_time_s_std",
                "prompts_per_s_mean",
                "output_tokens_per_s_mean",
                "ms_per_output_token_mean",
            ]
        )
        for a in aggregate_rows:
            writer.writerow(
                [
                    "aggregate",
                    a["scenario"],
                    a["concurrency"],
                    a["repeats_ok"],
                    f"{a['generation_time_s_mean']:.6f}",
                    f"{a['generation_time_s_std']:.6f}",
                    f"{a['prompts_per_s_mean']:.6f}",
                    f"{a['output_tokens_per_s_mean']:.6f}",
                    f"{a['ms_per_output_token_mean']:.6f}",
                ]
            )


def _write_txt(
    output_txt: Path,
    config_path: Path,
    prompts_csv: Path,
    repeat_results: list[RepeatResult],
    aggregate_rows: list[dict[str, Any]],
) -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    # Build quick lookup for LDA-vs-vanilla ratio by concurrency.
    agg_map: dict[tuple[str, int], dict[str, Any]] = {
        (a["scenario"], a["concurrency"]): a for a in aggregate_rows
    }

    with output_txt.open("w", encoding="utf-8") as f:
        f.write("Concurrency Speed Tests (Vanilla vs LDA)\n")
        f.write("=" * 72 + "\n")
        f.write(f"generated_utc: {now}\n")
        f.write(f"config_path: {config_path}\n")
        f.write(f"prompts_csv: {prompts_csv}\n")
        f.write(f"n_repeat_rows: {len(repeat_results)}\n\n")

        f.write("Per-repeat rows\n")
        f.write("-" * 72 + "\n")
        for r in repeat_results:
            f.write(
                f"{r.scenario:8s} c={r.concurrency:2d} rep={r.repeat_index} "
                f"status={r.status:5s} load={r.load_time_s:8.2f}s "
                f"gen={r.generation_time_s:8.2f}s out_tok={r.total_output_tokens:7d} "
                f"out_tps={r.output_tokens_per_s:9.2f} pps={r.prompts_per_s:8.2f}"
            )
            if r.status != "ok":
                f.write(f" error={r.error}")
            f.write("\n")

        f.write("\nAggregate means\n")
        f.write("-" * 72 + "\n")
        for a in aggregate_rows:
            f.write(
                f"{a['scenario']:8s} c={a['concurrency']:2d} "
                f"repeats_ok={a['repeats_ok']:2d} "
                f"gen_mean={a['generation_time_s_mean']:8.2f}s "
                f"gen_std={a['generation_time_s_std']:7.2f}s "
                f"out_tps_mean={a['output_tokens_per_s_mean']:9.2f} "
                f"ms_per_tok={a['ms_per_output_token_mean']:8.2f}\n"
            )

        f.write("\nLDA slowdown vs vanilla (generation_time_s_mean)\n")
        f.write("-" * 72 + "\n")
        concs = sorted({x.concurrency for x in repeat_results})
        for c in concs:
            base = agg_map.get(("vanilla", c))
            lda = agg_map.get(("lda", c))
            if not base or not lda or base["generation_time_s_mean"] <= 0:
                continue
            ratio = lda["generation_time_s_mean"] / base["generation_time_s_mean"]
            f.write(
                f"c={c:2d} lda/base={ratio:.3f}x "
                f"(vanilla={base['generation_time_s_mean']:.2f}s, "
                f"lda={lda['generation_time_s_mean']:.2f}s)\n"
            )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    bench_cfg = cfg.get("benchmark", {})

    n_prompts = int(bench_cfg.get("n_prompts", 1000))
    seed = int(args.seed if args.seed is not None else bench_cfg.get("seed", 42))
    warmup = bool(args.warmup if args.warmup is not None else bench_cfg.get("warmup", True))
    repeats = int(args.repeats if args.repeats is not None else bench_cfg.get("repeats", 3))
    concurrencies = (
        args.concurrencies
        if args.concurrencies
        else list(bench_cfg.get("concurrencies", [1, 2, 4, 8, 16, 32]))
    )
    continue_on_error = bool(
        args.continue_on_error
        if args.continue_on_error is not None
        else bench_cfg.get("continue_on_error", True)
    )
    length_mix = bench_cfg.get("length_mix", {"short": 0.4, "medium": 0.4, "long": 0.2})

    if not args.prompts_csv.exists():
        generate_mixed_prompts_csv(
            out_csv=args.prompts_csv,
            n_prompts=n_prompts,
            seed=seed,
            length_mix=length_mix,
        )
        print(f"Generated prompts CSV at {args.prompts_csv}")

    if args.generate_prompts_only:
        return

    prompts = load_prompts(args.prompts_csv, limit_prompts=args.limit_prompts)
    if not prompts:
        raise ValueError("No prompts loaded; check prompts CSV path/content.")

    all_repeat_results: list[RepeatResult] = []
    load_times: dict[str, float] = {}
    for scenario in ("vanilla", "lda"):
        print(f"\n=== Running scenario: {scenario} ===")
        load_time_s, rows = run_scenario(
            scenario=scenario,
            cfg=cfg,
            prompts=prompts,
            concurrencies=concurrencies,
            repeats=repeats,
            warmup=warmup,
            max_model_len_override=args.max_model_len,
            continue_on_error=continue_on_error,
        )
        load_times[scenario] = load_time_s
        all_repeat_results.extend(rows)
        print(f"{scenario} load_time_s={load_time_s:.2f}")

    aggregate_rows = _aggregate(all_repeat_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = args.output_dir / "concurrency_speed_tests.txt"
    csv_path = args.output_dir / "concurrency_speed_tests.csv"
    _write_txt(
        output_txt=txt_path,
        config_path=args.config,
        prompts_csv=args.prompts_csv,
        repeat_results=all_repeat_results,
        aggregate_rows=aggregate_rows,
    )
    _write_csv(
        output_csv=csv_path,
        repeat_results=all_repeat_results,
        aggregate_rows=aggregate_rows,
    )

    print(f"\nWrote summary: {txt_path}")
    print(f"Wrote machine-readable: {csv_path}")


if __name__ == "__main__":
    main()
