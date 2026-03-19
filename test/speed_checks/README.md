# Concurrency Speed Checks

This folder contains offline batch benchmarks for:

1. Vanilla vLLM (`model=base_model_id`)
2. LDA vLLM (`model=base_model_id`, `dormant_model=dormant_model_id`)

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

## Run commands

From inside the dev container:

```bash
python -u test/speed_checks/concurrency_speed_tests.py
```

Pilot/smoke run:

```bash
python -u test/speed_checks/concurrency_speed_tests.py --limit-prompts 32 --concurrencies 1 2 --repeats 1
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

`run_metadata.json` includes status (`running`, `completed`, `failed`) and
timestamps.
