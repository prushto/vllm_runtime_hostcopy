# Concurrency Speed Checks

This folder contains offline batch benchmarks for:

1. Vanilla vLLM (`model=base_model_id`)
2. LDA vLLM (`model=base_model_id`, `dormant_model=dormant_model_id`)
3. Optional **LDA + KL collection** (`scenario` name `lda_kl`): same as LDA but sets `additional_config["lda"]["collect_kl"]=True` so each forward runs the extra KL softmax work (for overhead measurement vs baseline LDA).

Configure runs with `benchmark.scenarios`, e.g. `["vanilla", "lda"]` (default) or `["vanilla", "lda", "lda_kl"]` to compare generation time **lda_kl vs lda** in the report footer. Override on the CLI: `--scenarios vanilla lda lda_kl`.

## Why this runbook exists

Long runs can fail late (e.g. CUDA errors). The benchmark script now writes
checkpointed results after each completed repeat, so partial progress is kept.

## Prompt count vs concurrency

The harness submits **chunks of size `concurrency`** (`messages[i : i+concurrency]`). If
`len(prompts) < concurrency`, the first batch has **only that many** requests in flight, so
you never stress the target parallelism. Use:

- **`n_prompts >= max(benchmark.concurrencies)`** so the first wave can fill, and
- **several waves** (e.g. 5000 prompts at concurrency 512 → many waves) so timing reflects
  steady scheduling, not a single short batch.

**Normalized lmsys pack (5k, recommended for Modal + HF LDA):** built from the `dormant` repo (HF access / lmsys cache):

```bash
cd /path/to/dormant
python3 modal/scripts/build_lda_prompt_pack.py
```

Output: `dormant/modal/data/lda_dsv3_dormant1_round1/prompts_n5000_s42.csv` → in the Modal image **`/root/modal_data/lda_dsv3_dormant1_round1/prompts_n5000_s42.csv`**. **Modal round 1 (default, real checkpoints):** [`dormant/modal/config/lda_dsv3_dormant1_round1.yaml`](../../../dormant/modal/config/lda_dsv3_dormant1_round1.yaml) → **`/root/modal_pkg/config/lda_dsv3_dormant1_round1.yaml`** for [`vllm_lda_experiment_modal.py`](../../../dormant/modal/vllm_lda_experiment_modal.py). Dummy debug: `lda_dsv3_dormant1_round1_dummy_debug.yaml` + **`--dummy-weights`**. For local `concurrency_speed_tests.py`, pass `--config` with an absolute path to either YAML.

**Small ad-hoc export** (e.g. 1k in this folder only): `export_lmsys_prompts_for_speed_check.py` → `prompts_lmsys_n1000_s42.csv`; Modal path `/root/vllm_speed_checks/prompts_lmsys_n1000_s42.csv`.

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

The `dormant/modal` apps [`vllm_lda_experiment_modal.py`](../../../dormant/modal/vllm_lda_experiment_modal.py) (canonical **5k** science runs → `lda_experiments/`) and [`vllm_phase_c_speed_check_modal.py`](../../../dormant/modal/vllm_phase_c_speed_check_modal.py) (legacy benchmark defaults → `vllm_speed_checks/`) bake this folder into the worker at `/root/vllm_speed_checks`. See [`dormant/modal/EXPERIMENTS.md`](../../../dormant/modal/EXPERIMENTS.md).

- **Speed-check default:** [`config_modal_phase_c_lda_dummy.yaml`](config_modal_phase_c_lda_dummy.yaml) — `load_format=dummy`, same HF DeepSeek-V3 id for base and dormant, **max_model_len=256**, **max_new_tokens=128**, **enforce_eager**, **max_num_batched_tokens=256** (256 context + stock CSV → gen capped at 128; for 256-token gen, raise to 384/256 as in comments in [`config_modal_phase_c_lda.yaml`](config_modal_phase_c_lda.yaml)).
- **`--real-weights`:** [`config_modal_phase_c_lda.yaml`](config_modal_phase_c_lda.yaml) — volume paths, same length / init knobs.

**Modal CLI:** omitting **`--concurrencies`** / **`--scenarios`** lets the chosen YAML’s **`benchmark.concurrencies`** and **`benchmark.scenarios`** apply (instead of old hardcoded Modal defaults). Use **`--prompts-csv /root/modal_data/...csv`** for the 5k pack, or **`/root/vllm_speed_checks/...csv`** for exports co-located with this folder.

**Progress during a repeat:** after each **`llm.chat`** batch, the harness logs prompts finished, cumulative output tokens, seconds elapsed for that repeat, and rolling **`out_tok/s`** (excludes model load). Set **`benchmark.progress_log_every_batches`** in YAML or **`--progress-log-every-batches N`** (0 = off).

Results: **`/models/results/lda_experiments/<run_id>/`** (experiment app) or **`/models/results/vllm_speed_checks/<run_id>/`** (legacy app). Use **`--run-label my-purpose`** for a human id (`yymmdd-HHMM-my-purpose-<n>` UTC). The harness rewrites **`concurrency_speed_tests.txt`**, **`.csv`**, and **`run_metadata.json` after each concurrency repeat** (and calls Modal **`volume.commit()`** when run via either Modal app).

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
