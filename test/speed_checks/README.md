# Concurrency Speed Checks

This folder contains offline batch benchmarks for:

1. Vanilla vLLM (`model=base_model_id`)
2. LDA vLLM (`model=base_model_id`, `dormant_model=dormant_model_id`)
3. Optional **LDA + KL collection** (`scenario` name `lda_kl`): same as LDA but sets `additional_config["lda"]["collect_kl"]=True` so each forward runs the extra KL softmax work (for overhead measurement vs baseline LDA).

Configure runs with `benchmark.scenarios`, e.g. `["vanilla", "lda"]` (default) or `["vanilla", "lda", "lda_kl"]` to compare generation time **lda_kl vs lda** in the report footer. Override on the CLI: `--scenarios vanilla lda lda_kl`.

## Why this runbook exists

Long runs can fail late (e.g. CUDA errors). The benchmark script now writes
checkpointed results after each completed repeat, so partial progress is kept.

## Safety: immutable run checkout

Do **not** modify bind-mounted runtime code while a benchmark is running.

Recommended workflow:

- Keep one checkout dedicated to the active benchmark run.
- Do development in a separate checkout/branch.

## Defaults (fast-first)

`config.yaml` defaults are tuned for fast iteration:

- `n_prompts: 500`
- `repeats: 1`
- `warmup: false`
- `concurrencies: [1, 2, 4, 8, 16, 32]`
- `scenarios: ["vanilla", "lda"]` — add `"lda_kl"` when measuring KL-collection overhead (loads a **third** engine pass).

## Run commands

From inside the dev container:

```bash
python -u test/speed_checks/concurrency_speed_tests.py
```

### Modal (Phase C, DeepSeek + dormant)

The `dormant/modal` app [`vllm_phase_c_speed_check_modal.py`](../../../dormant/modal/vllm_phase_c_speed_check_modal.py) bakes this folder into the worker at `/root/vllm_speed_checks`.

- **Default:** [`config_modal_phase_c_lda_dummy.yaml`](config_modal_phase_c_lda_dummy.yaml) — `load_format=dummy`, same HF DeepSeek-V3 id for base and dormant, **max_model_len=256**, **max_new_tokens=128**, **enforce_eager**, **max_num_batched_tokens=256** (256 context + stock CSV → gen capped at 128; for 256-token gen, raise to 384/256 as in comments in [`config_modal_phase_c_lda.yaml`](config_modal_phase_c_lda.yaml)).
- **`--real-weights`:** [`config_modal_phase_c_lda.yaml`](config_modal_phase_c_lda.yaml) — volume paths, same length / init knobs.

512 prompts, LDA-only, concurrency 128/180/256 by default. Results: `/models/results/vllm_speed_checks/<run_id>/` on the Modal volume.

LDA **KV headroom:** set `additional_config.lda.kv_cache_max_memory_gib` to cap per-worker KV bytes **before** block planning (`get_kv_cache_configs`). When this key is set, `kv_cache_safety_fraction` defaults to **1.0** (no extra shrink); omit the cap to fall back to **`kv_cache_safety_fraction` default 0.88**. Modal YAMLs use **`gpu_memory_utilization: 0.85`** plus a GiB cap; tune the cap (and optionally `kv_cache_safety_fraction`) if init OOM or KV starvation persists.

Pilot/smoke run:

```bash
python -u test/speed_checks/concurrency_speed_tests.py --limit-prompts 32 --concurrencies 1 2 --repeats 1
```

KL overhead only (LDA vs LDA+`collect_kl`, no vanilla):

```bash
python -u test/speed_checks/concurrency_speed_tests.py --limit-prompts 32 --concurrencies 1 2 --repeats 1 --scenarios lda lda_kl
```

Medium run:

```bash
python -u test/speed_checks/concurrency_speed_tests.py --limit-prompts 200 --repeats 1
```

## Output layout

Each run writes to a run-scoped folder:

`test/speed_checks/test_results/<run_id>/`

Files:

- `concurrency_speed_tests.csv`
- `concurrency_speed_tests.txt`
- `run_metadata.json`

`run_metadata.json` includes status (`running`, `completed`, `failed`),
timestamps, the **full parsed `config` YAML snapshot** at run start (so later
edits to `config.yaml` do not change old results), **`cli_args`**, and **`argv`**.
