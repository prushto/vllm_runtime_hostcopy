#!/usr/bin/env python3
"""Offline concurrency speed checks for vanilla vLLM vs LDA.

This script benchmarks two scenarios on the same prompt set:
1) Vanilla vLLM (single model).
2) LDA vLLM (base + dormant model).

It measures model load time separately from generation time, and writes both
human-readable and machine-readable reports under test_results/.

With ``benchmark.save_generations: true`` (or ``--save-generations``), each
``llm.chat`` batch also appends one JSON object per prompt to
``<run_dir>/generations.jsonl`` (UTF-8, flushed after every batch) so long Modal
runs can checkpoint completions onto the volume via the after-checkpoint hook.

**LDA α sweep:** set ``benchmark.lda_alphas: [0.5, 1.0, 3.0]`` (or root ``lda_alphas``).
Each value runs ``lda`` / ``lda_kl`` with a fresh ``LLM`` (``lda_alpha`` is fixed at init).
``vanilla`` runs once. CSV/txt include an ``lda_alpha`` column / suffix where applicable.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import sys
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

import yaml

# Optional hook (e.g. Modal `volume.commit()`) after each txt/csv/metadata checkpoint.
_after_checkpoint_hook: Callable[[], None] | None = None


def set_after_checkpoint_hook(fn: Callable[[], None] | None) -> None:
    """Register a callback invoked after incremental results are flushed to disk."""
    global _after_checkpoint_hook
    _after_checkpoint_hook = fn


def _sanitize_run_label(label: str) -> str:
    s = label.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "run"


def make_human_run_id(purpose: str, output_parent: Path, now: datetime | None = None) -> str:
    """Human-readable folder name: ``yymmdd-HHMM-<slug>-<seq>`` (UTC), seq disambiguates collisions."""
    now = now or datetime.now(timezone.utc)
    slug = _sanitize_run_label(purpose)
    prefix = f"{now.strftime('%y%m%d-%H%M')}-{slug}-"
    output_parent = Path(output_parent)
    output_parent.mkdir(parents=True, exist_ok=True)
    max_n = 0
    if output_parent.is_dir():
        for child in output_parent.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                max_n = max(max_n, int(suffix))
    return f"{prefix}{max_n + 1}"


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
    # Set for lda / lda_kl; None for vanilla (and legacy runs without the field).
    lda_alpha: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline concurrency speed tests for vanilla vs LDA vLLM."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prompts-csv", type=Path, default=DEFAULT_PROMPTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run folder name. Overrides --run-label.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        metavar="PURPOSE",
        help=(
            "Human-readable run id (UTC yymmdd-HHMM-<purpose>-<n> under --output-dir). "
            "Ignored if --run-id is set."
        ),
    )
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
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="*",
        default=None,
        help="Override benchmark.scenarios (e.g. vanilla lda lda_kl).",
    )
    parser.add_argument(
        "--progress-log-every-batches",
        type=int,
        default=None,
        metavar="N",
        help="Log every N completed llm.chat batches during a repeat (1=every batch). "
        "0 disables. Default: benchmark.progress_log_every_batches or 1.",
    )
    parser.add_argument(
        "--save-generations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append per-prompt completions to run_dir/generations.jsonl (default: benchmark.save_generations).",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict-like yaml config, got: {type(data).__name__}")
    return data


def _namespace_for_metadata(ns: argparse.Namespace) -> dict[str, Any]:
    """JSON-serializable snapshot of argparse Namespace (Paths -> str)."""
    out: dict[str, Any] = {}
    for k, v in vars(ns).items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) if isinstance(x, Path) else x for x in v]
        else:
            out[k] = v
    return out


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


def _additional_config_for_scenario(cfg: dict[str, Any], scenario: str) -> dict[str, Any] | None:
    """Merge optional root `additional_config` with LDA flags per scenario."""
    base = dict(cfg.get("additional_config") or {})
    if scenario == "lda_kl":
        lda = dict(base.get("lda") or {})
        lda["collect_kl"] = True
        base["lda"] = lda
        return base
    if scenario == "lda" and base:
        return base
    return None


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
    if scenario in ("lda", "lda_kl"):
        kwargs["dormant_model"] = cfg["dormant_model_id"]
        kwargs["lda_alpha"] = float(cfg.get("lda_alpha", 0.5))
    ac = _additional_config_for_scenario(cfg, scenario)
    if ac is not None:
        kwargs["additional_config"] = ac

    # Optional large-model / Modal knobs (omitted when absent — keeps local Qwen runs unchanged).
    for key in ("tensor_parallel_size", "max_num_batched_tokens"):
        val = cfg.get(key)
        if val is not None:
            kwargs[key] = int(val)
    for key in (
        "trust_remote_code",
        "enforce_eager",
        "enable_expert_parallel",
        "disable_custom_all_reduce",
    ):
        if key in cfg and cfg[key] is not None:
            kwargs[key] = bool(cfg[key])
    if cfg.get("load_format") is not None:
        kwargs["load_format"] = str(cfg["load_format"])

    return kwargs


def _messages_from_prompts(prompts: list[PromptRow]) -> list[list[dict[str, str]]]:
    return [[{"role": "user", "content": row.prompt_text}] for row in prompts]


def _completion_fields_from_request_output(
    ro: Any,
) -> tuple[str, int, str | None, float | None, float | None]:
    """Decode text, n_output_tokens, finish_reason, LDA KL from a ``RequestOutput``."""
    outs = getattr(ro, "outputs", None) or []
    if not outs:
        return "", 0, None, None, None
    c0 = outs[0]
    text = getattr(c0, "text", None) or ""
    tids = getattr(c0, "token_ids", None)
    n_tok = len(tids) if tids is not None else 0
    fr = getattr(c0, "finish_reason", None)
    mean_kl = getattr(c0, "mean_kl_divergence", None)
    max_kl = getattr(c0, "max_kl_divergence", None)
    return str(text), int(n_tok), fr, mean_kl, max_kl


def _append_generations_jsonl(
    fh: TextIO,
    *,
    prompt_row: PromptRow,
    response_text: str,
    scenario: str,
    lda_alpha: float | None,
    concurrency: int,
    repeat_index: int,
    batch_num: int,
    finish_reason: str | None,
    tokens_generated: int,
    status: str,
    error: str,
    mean_kl_divergence: float | None = None,
    max_kl_divergence: float | None = None,
) -> None:
    rec = {
        "schema_version": 1,
        "prompt_id": prompt_row.prompt_id,
        "length_bucket": prompt_row.length_bucket,
        "prompt_text": prompt_row.prompt_text,
        "response_text": response_text,
        "scenario": scenario,
        "lda_alpha": lda_alpha,
        "concurrency": concurrency,
        "repeat_index": repeat_index,
        "batch_num": batch_num,
        "finish_reason": finish_reason,
        "tokens_generated": tokens_generated,
        "mean_kl_divergence": mean_kl_divergence,
        "max_kl_divergence": max_kl_divergence,
        "status": status,
        "error": error,
    }
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()


def _run_one_dataset_pass(
    llm: Any,
    messages: list[list[dict[str, str]]],
    prompt_rows: list[PromptRow],
    concurrency: int,
    sampling_params: Any,
    *,
    scenario: str,
    repeat_index: int,
    progress_log_every_batch: int = 1,
    generations_fh: TextIO | None = None,
    lda_alpha: float | None = None,
) -> tuple[float, int, int]:
    """Run all prompts in chunks of ``concurrency``. Optionally log rolling out_tok/s after each batch.

    Timers here cover **this repeat’s** ``llm.chat`` calls only (not model load).

    ``prompt_rows`` must align 1:1 with ``messages`` (same order and length).
    """
    if len(messages) != len(prompt_rows):
        raise ValueError(
            f"messages ({len(messages)}) and prompt_rows ({len(prompt_rows)}) length mismatch"
        )
    t0 = time.perf_counter()
    total_output_tokens = 0
    total_input_tokens = 0
    n_msg = len(messages)
    batch_num = 0
    for i in range(0, n_msg, concurrency):
        batch = messages[i : i + concurrency]
        batch_prompts = prompt_rows[i : i + concurrency]
        batch_num += 1
        outputs = llm.chat(batch, sampling_params=sampling_params, use_tqdm=False)
        if len(outputs) != len(batch_prompts):
            raise RuntimeError(
                f"llm.chat returned {len(outputs)} outputs for batch size {len(batch_prompts)}"
            )
        for pr, out in zip(batch_prompts, outputs):
            resp_text, n_out, fin, mean_kl, max_kl = _completion_fields_from_request_output(
                out
            )
            if out.outputs:
                total_output_tokens += len(out.outputs[0].token_ids)
            prompt_token_ids = getattr(out, "prompt_token_ids", None)
            if prompt_token_ids is not None:
                total_input_tokens += len(prompt_token_ids)
            if generations_fh is not None:
                st = "ok"
                err = ""
                if not out.outputs:
                    st = "error"
                    err = "empty_completion"
                _append_generations_jsonl(
                    generations_fh,
                    prompt_row=pr,
                    response_text=resp_text,
                    scenario=scenario,
                    lda_alpha=lda_alpha,
                    concurrency=concurrency,
                    repeat_index=repeat_index,
                    batch_num=batch_num,
                    finish_reason=fin,
                    tokens_generated=n_out,
                    status=st,
                    error=err,
                    mean_kl_divergence=mean_kl,
                    max_kl_divergence=max_kl,
                )
        if generations_fh is not None:
            _invoke_after_checkpoint_hook()
        n_done = min(i + len(batch), n_msg)
        elapsed = time.perf_counter() - t0
        rate = total_output_tokens / elapsed if elapsed > 0 else 0.0
        is_last = n_done >= n_msg
        if progress_log_every_batch > 0 and (
            batch_num % progress_log_every_batch == 0 or is_last
        ):
            print(
                f"[{scenario}] c={concurrency} rep={repeat_index} batch={batch_num} "
                f"prompts={n_done}/{n_msg} out_tok={total_output_tokens} "
                f"gen_s={elapsed:.1f} out_tok/s={rate:.2f} "
                f"(this repeat only; excludes load)",
                flush=True,
            )
    elapsed = time.perf_counter() - t0
    return elapsed, total_output_tokens, total_input_tokens


def _log_scenario_tag(scenario: str, cfg: dict[str, Any]) -> str:
    if scenario in ("lda", "lda_kl"):
        return f"{scenario} α={float(cfg.get('lda_alpha', 0.5))}"
    return scenario


def _repeat_lda_alpha(scenario: str, cfg: dict[str, Any]) -> float | None:
    if scenario not in ("lda", "lda_kl"):
        return None
    return float(cfg.get("lda_alpha", 0.5))


def _resolve_lda_alphas(bench_cfg: dict[str, Any], cfg: dict[str, Any]) -> list[float]:
    """Single alpha from root ``lda_alpha`` unless ``lda_alphas`` is set (benchmark or root)."""
    raw = bench_cfg.get("lda_alphas")
    if raw is None:
        raw = cfg.get("lda_alphas")
    if raw is not None and isinstance(raw, (list, tuple)) and len(raw) > 0:
        return [float(x) for x in raw]
    return [float(cfg.get("lda_alpha", 0.5))]


def _load_time_meta_key(scenario: str, cfg: dict[str, Any]) -> str:
    if scenario in ("lda", "lda_kl"):
        return f"{scenario}@a{float(cfg.get('lda_alpha', 0.5))}"
    return scenario


def run_scenario(
    scenario: str,
    cfg: dict[str, Any],
    prompts: list[PromptRow],
    concurrencies: list[int],
    repeats: int,
    warmup: bool,
    max_model_len_override: int | None,
    continue_on_error: bool,
    on_repeat_result: Any = None,
    progress_log_every_batch: int = 1,
    generations_fh: TextIO | None = None,
) -> tuple[float, list[RepeatResult]]:
    from vllm import LLM

    tag = _log_scenario_tag(scenario, cfg)
    ra = _repeat_lda_alpha(scenario, cfg)

    llm_kwargs = _build_llm_kwargs(scenario, cfg, max_model_len_override)
    print(f"[{tag}] Loading model(s)...")
    t_load = time.perf_counter()
    llm: Any = None
    llm = LLM(**llm_kwargs)
    load_time_s = time.perf_counter() - t_load
    print(f"[{tag}] load_time_s={load_time_s:.2f}")

    sampling_params = _sampling_params_from_config(cfg["gen_config"])
    messages = _messages_from_prompts(prompts)
    all_results: list[RepeatResult] = []

    try:
        for concurrency in concurrencies:
            print(f"[{tag}] concurrency={concurrency} start")
            warmup_time_s: float | None = None
            if warmup:
                warm_batch = messages[: max(1, min(concurrency, len(messages)))]
                t_warmup0 = time.perf_counter()
                try:
                    llm.chat(warm_batch, sampling_params=sampling_params, use_tqdm=False)
                    warmup_time_s = time.perf_counter() - t_warmup0
                    print(f"[{tag}] concurrency={concurrency} warmup_s={warmup_time_s:.2f}")
                except Exception as e:  # noqa: BLE001
                    rr = RepeatResult(
                        scenario=scenario,
                        concurrency=concurrency,
                        repeat_index=0,
                        n_prompts=len(messages),
                        load_time_s=load_time_s,
                        warmup_time_s=None,
                        generation_time_s=0.0,
                        total_output_tokens=0,
                        total_input_tokens=0,
                        prompts_per_s=0.0,
                        output_tokens_per_s=0.0,
                        ms_per_output_token=float("inf"),
                        end_to_end_time_s=load_time_s,
                        status="error",
                        error=f"warmup_failed: {e}",
                        lda_alpha=ra,
                    )
                    all_results.append(rr)
                    if on_repeat_result is not None:
                        on_repeat_result(rr)
                    print(f"[{tag}] concurrency={concurrency} warmup ERROR: {e}")
                    if not continue_on_error:
                        raise
                    # Continue to next concurrency to keep partial data.
                    continue

            for repeat_idx in range(1, repeats + 1):
                print(f"[{tag}] concurrency={concurrency} repeat={repeat_idx}/{repeats} start")
                try:
                    gen_time_s, out_tokens, in_tokens = _run_one_dataset_pass(
                        llm=llm,
                        messages=messages,
                        prompt_rows=prompts,
                        concurrency=concurrency,
                        sampling_params=sampling_params,
                        scenario=scenario,
                        repeat_index=repeat_idx,
                        progress_log_every_batch=progress_log_every_batch,
                        generations_fh=generations_fh,
                        lda_alpha=ra,
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
                            lda_alpha=ra,
                        )
                    )
                    rr = all_results[-1]
                    if on_repeat_result is not None:
                        on_repeat_result(rr)
                    print(
                        f"[{tag}] concurrency={concurrency} repeat={repeat_idx}/{repeats} "
                        f"done gen_s={gen_time_s:.2f} out_toks={out_tokens} out_tps={out_tokens_per_s:.2f}"
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
                        lda_alpha=ra,
                    )
                    all_results.append(rr)
                    if on_repeat_result is not None:
                        on_repeat_result(rr)
                    print(
                        f"[{tag}] concurrency={concurrency} repeat={repeat_idx}/{repeats} "
                        f"ERROR: {e}"
                    )
                    if not continue_on_error:
                        raise
    finally:
        if llm is not None:
            del llm

    return load_time_s, all_results


def _aggregate(results: list[RepeatResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float | None], list[RepeatResult]] = {}
    for r in results:
        if r.status != "ok":
            continue
        grouped.setdefault((r.scenario, r.concurrency, r.lda_alpha), []).append(r)

    rows: list[dict[str, Any]] = []

    def _sort_key(t: tuple[str, int, float | None]) -> tuple[str, int, float]:
        scenario, concurrency, alpha = t
        a = float(alpha) if alpha is not None else float("-inf")
        return (scenario, concurrency, a)

    for (scenario, concurrency, lda_alpha), g in sorted(
        grouped.items(), key=lambda x: _sort_key(x[0])
    ):
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
                "lda_alpha": lda_alpha,
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
                "lda_alpha",
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
                    "" if r.lda_alpha is None else f"{r.lda_alpha}",
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
                "lda_alpha",
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
            la = a.get("lda_alpha")
            writer.writerow(
                [
                    "aggregate",
                    a["scenario"],
                    "" if la is None else f"{la}",
                    a["concurrency"],
                    a["repeats_ok"],
                    f"{a['generation_time_s_mean']:.6f}",
                    f"{a['generation_time_s_std']:.6f}",
                    f"{a['prompts_per_s_mean']:.6f}",
                    f"{a['output_tokens_per_s_mean']:.6f}",
                    f"{a['ms_per_output_token_mean']:.6f}",
                ]
            )
        f.flush()


def _write_txt(
    output_txt: Path,
    config_path: Path,
    prompts_csv: Path,
    repeat_results: list[RepeatResult],
    aggregate_rows: list[dict[str, Any]],
) -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    # Build quick lookup for LDA-vs-vanilla ratio by concurrency (+ alpha for lda).
    agg_map: dict[tuple[str, int, float | None], dict[str, Any]] = {
        (a["scenario"], a["concurrency"], a.get("lda_alpha")): a for a in aggregate_rows
    }

    with output_txt.open("w", encoding="utf-8") as f:
        f.write("Concurrency Speed Tests (Vanilla / LDA / optional LDA+collect_kl)\n")
        f.write("=" * 72 + "\n")
        f.write(f"generated_utc: {now}\n")
        f.write(f"config_path: {config_path}\n")
        f.write(f"prompts_csv: {prompts_csv}\n")
        f.write(f"n_repeat_rows: {len(repeat_results)}\n\n")

        f.write("Per-repeat rows\n")
        f.write("-" * 72 + "\n")
        for r in repeat_results:
            alpha_s = (
                f" α={r.lda_alpha}" if r.lda_alpha is not None else ""
            )
            f.write(
                f"{r.scenario:8s}{alpha_s} c={r.concurrency:2d} rep={r.repeat_index} "
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
            la = a.get("lda_alpha")
            alpha_s = f" α={la}" if la is not None else ""
            f.write(
                f"{a['scenario']:8s}{alpha_s} c={a['concurrency']:2d} "
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
            base = agg_map.get(("vanilla", c, None))
            if not base or base["generation_time_s_mean"] <= 0:
                continue
            for a in aggregate_rows:
                if a["scenario"] != "lda" or a["concurrency"] != c:
                    continue
                la = a.get("lda_alpha")
                ratio = a["generation_time_s_mean"] / base["generation_time_s_mean"]
                f.write(
                    f"c={c:2d} α={la} lda/base={ratio:.3f}x "
                    f"(vanilla={base['generation_time_s_mean']:.2f}s, "
                    f"lda={a['generation_time_s_mean']:.2f}s)\n"
                )

        f.write("\nLDA + collect_kl overhead vs LDA (generation_time_s_mean)\n")
        f.write("-" * 72 + "\n")
        has_lda_kl = any(a["scenario"] == "lda_kl" for a in aggregate_rows)
        if not has_lda_kl:
            f.write("(no lda_kl scenario in this run — add \"lda_kl\" to benchmark.scenarios)\n")
        else:
            for c in concs:
                for a_kl in aggregate_rows:
                    if a_kl["scenario"] != "lda_kl" or a_kl["concurrency"] != c:
                        continue
                    alpha = a_kl.get("lda_alpha")
                    lda = agg_map.get(("lda", c, alpha))
                    if (
                        not lda
                        or lda["generation_time_s_mean"] <= 0
                        or a_kl["generation_time_s_mean"] <= 0
                    ):
                        continue
                    ratio = a_kl["generation_time_s_mean"] / lda["generation_time_s_mean"]
                    f.write(
                        f"c={c:2d} α={alpha} lda_kl/lda={ratio:.3f}x "
                        f"(lda={lda['generation_time_s_mean']:.2f}s, "
                        f"lda_kl={a_kl['generation_time_s_mean']:.2f}s)\n"
                    )
        f.flush()


def _write_run_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()


def _invoke_after_checkpoint_hook() -> None:
    if _after_checkpoint_hook is None:
        return
    try:
        _after_checkpoint_hook()
    except Exception as e:  # noqa: BLE001
        print(f"[speed_checks] after-checkpoint hook failed: {e}", file=sys.stderr, flush=True)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    bench_cfg = cfg.get("benchmark", {})

    n_prompts = int(bench_cfg.get("n_prompts", 500))
    seed = int(args.seed if args.seed is not None else bench_cfg.get("seed", 42))
    warmup = bool(args.warmup if args.warmup is not None else bench_cfg.get("warmup", False))
    repeats = int(args.repeats if args.repeats is not None else bench_cfg.get("repeats", 1))
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
    progress_log_every_batch = int(
        args.progress_log_every_batches
        if args.progress_log_every_batches is not None
        else bench_cfg.get("progress_log_every_batches", 1)
    )

    save_generations = bool(bench_cfg.get("save_generations", False))
    if args.save_generations is not None:
        save_generations = bool(args.save_generations)

    _ALLOWED_SCENARIOS = frozenset({"vanilla", "lda", "lda_kl"})
    default_scenarios = ["vanilla", "lda"]
    scenarios: list[str] = (
        list(args.scenarios)
        if args.scenarios
        else list(bench_cfg.get("scenarios", default_scenarios))
    )
    if not scenarios:
        raise ValueError("benchmark.scenarios must be a non-empty list")
    for s in scenarios:
        if s not in _ALLOWED_SCENARIOS:
            raise ValueError(
                f"Unknown scenario {s!r}; allowed: {sorted(_ALLOWED_SCENARIOS)}"
            )

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

    # Respect config benchmark.n_prompts by default even when prompts.csv already exists.
    limit_prompts = args.limit_prompts if args.limit_prompts is not None else n_prompts
    prompts = load_prompts(args.prompts_csv, limit_prompts=limit_prompts)
    if not prompts:
        raise ValueError("No prompts loaded; check prompts CSV path/content.")
    if len(prompts) != n_prompts:
        print(
            f"Loaded {len(prompts)} prompts (requested default {n_prompts}). "
            "This can happen when --limit-prompts is used or CSV has fewer rows."
        )

    if args.run_id:
        run_id = args.run_id
    elif args.run_label:
        run_id = make_human_run_id(args.run_label, args.output_dir)
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    txt_path = run_dir / "concurrency_speed_tests.txt"
    csv_path = run_dir / "concurrency_speed_tests.csv"
    metadata_path = run_dir / "run_metadata.json"
    gen_path = run_dir / "generations.jsonl"
    gen_fh: TextIO | None = None
    if save_generations:
        gen_fh = gen_path.open("w", encoding="utf-8")

    all_repeat_results: list[RepeatResult] = []
    load_times: dict[str, float] = {}
    lda_alphas = _resolve_lda_alphas(bench_cfg, cfg)
    metadata_payload: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        # Full YAML as loaded at run start (edits to the file later do not change this).
        "config": copy.deepcopy(cfg),
        "cli_args": _namespace_for_metadata(args),
        "argv": sys.argv,
        "prompts_csv": str(args.prompts_csv),
        "n_prompts_requested": n_prompts,
        "n_prompts_loaded": len(prompts),
        "limit_prompts": limit_prompts,
        "concurrencies": concurrencies,
        "repeats": repeats,
        "warmup": warmup,
        "seed": seed,
        "max_model_len_override": args.max_model_len,
        "continue_on_error": continue_on_error,
        "scenarios": scenarios,
        "run_label": args.run_label,
        "progress_log_every_batches": progress_log_every_batch,
        "lda_alphas": lda_alphas,
        "save_generations": save_generations,
        "generations_jsonl": str(gen_path) if save_generations else None,
    }
    _write_run_metadata(metadata_path, metadata_payload)
    _invoke_after_checkpoint_hook()

    def _persist_checkpoint() -> None:
        aggregate_rows_local = _aggregate(all_repeat_results)
        _write_txt(
            output_txt=txt_path,
            config_path=args.config,
            prompts_csv=args.prompts_csv,
            repeat_results=all_repeat_results,
            aggregate_rows=aggregate_rows_local,
        )
        _write_csv(
            output_csv=csv_path,
            repeat_results=all_repeat_results,
            aggregate_rows=aggregate_rows_local,
        )
        metadata_payload["last_checkpoint_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_payload["n_repeat_rows"] = len(all_repeat_results)
        metadata_payload["status"] = "running"
        _write_run_metadata(metadata_path, metadata_payload)
        _invoke_after_checkpoint_hook()

    def _on_repeat_result(rr: RepeatResult) -> None:
        all_repeat_results.append(rr)
        _persist_checkpoint()

    try:
        for scenario in scenarios:
            if scenario == "vanilla":
                cfgs_to_run = [cfg]
            else:
                cfgs_to_run = []
                for alpha in lda_alphas:
                    cfg_i = copy.deepcopy(cfg)
                    cfg_i["lda_alpha"] = alpha
                    cfgs_to_run.append(cfg_i)

            for cfg_run in cfgs_to_run:
                if scenario == "vanilla":
                    print(f"\n=== Running scenario: {scenario} ===")
                else:
                    print(
                        f"\n=== Running scenario: {scenario} "
                        f"(lda_alpha={float(cfg_run.get('lda_alpha', 0.5))}) ==="
                    )
                load_time_s, _rows = run_scenario(
                    scenario=scenario,
                    cfg=cfg_run,
                    prompts=prompts,
                    concurrencies=concurrencies,
                    repeats=repeats,
                    warmup=warmup,
                    max_model_len_override=args.max_model_len,
                    continue_on_error=continue_on_error,
                    on_repeat_result=_on_repeat_result,
                    progress_log_every_batch=progress_log_every_batch,
                    generations_fh=gen_fh,
                )
                lk = _load_time_meta_key(scenario, cfg_run)
                load_times[lk] = load_time_s
                metadata_payload["load_times"] = load_times
                _write_run_metadata(metadata_path, metadata_payload)
                _invoke_after_checkpoint_hook()
                print(f"{lk} load_time_s={load_time_s:.2f}")

        metadata_payload["status"] = "completed"
        metadata_payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_run_metadata(metadata_path, metadata_payload)
    except Exception as e:  # noqa: BLE001
        metadata_payload["status"] = "failed"
        metadata_payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_payload["error"] = str(e)
        _write_run_metadata(metadata_path, metadata_payload)
        _invoke_after_checkpoint_hook()
        raise
    finally:
        if gen_fh is not None:
            gen_fh.close()
        if save_generations and gen_path.is_file():
            try:
                metadata_payload["generations_line_count"] = sum(
                    1 for _ in gen_path.open(encoding="utf-8")
                )
            except OSError:
                metadata_payload["generations_line_count"] = None
            _write_run_metadata(metadata_path, metadata_payload)
            _invoke_after_checkpoint_hook()

    print(f"\nWrote summary: {txt_path}")
    print(f"Wrote machine-readable: {csv_path}")
    print(f"Wrote run metadata: {metadata_path}")
    if save_generations and gen_path.is_file():
        n = metadata_payload.get("generations_line_count")
        if isinstance(n, int):
            print(f"Wrote generations JSONL ({n} lines): {gen_path}")
        else:
            print(f"Wrote generations JSONL: {gen_path}")


if __name__ == "__main__":
    main()
