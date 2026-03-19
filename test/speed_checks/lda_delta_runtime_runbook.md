# LDA Delta Runtime Validation Runbook

This runbook validates the runtime `base + delta` dormant assembly path against
the existing full-dormant LDA path using only Qwen 0.5B synthetic dormant data.

## Preconditions

- Use repository: `/home/prushto/vllm_runtime_hostcopy_duplicate`
- Branch: `delta-weights`
- Synthetic dormant and delta artifacts prepared first (0.5B only).

## A. Prepare synthetic dormant + delta (0.5B)

```bash
python3 "/home/prushto/vllm_runtime_hostcopy_duplicate/test/speed_checks/prepare_synthetic_0p5b_dormant.py"
```

Expected outputs:
- `/workspace/dormant/weight_deltas/synthetic_models/qwen2_5_0_5b_synth_dormant`
- `/workspace/dormant/weight_deltas/artifacts/qwen2_5_0_5b_instruct__synthetic_dormant__v1`

## B. Artifact guard checks

```bash
python3 "/home/prushto/vllm_runtime_hostcopy_duplicate/test/speed_checks/lda_delta_manifest_guard_checks.py" \
  --artifact-dir "/workspace/dormant/weight_deltas/artifacts/qwen2_5_0_5b_instruct__synthetic_dormant__v1"
```

Pass criteria:
- Output JSON has `"status": "ok"`.

## C. Deterministic parity checks

```bash
python3 "/home/prushto/vllm_runtime_hostcopy_duplicate/test/speed_checks/lda_delta_parity_checks.py" \
  --base-model "Qwen/Qwen2.5-0.5B-Instruct" \
  --dormant-model "/workspace/dormant/weight_deltas/synthetic_models/qwen2_5_0_5b_synth_dormant" \
  --dormant-delta-dir "/workspace/dormant/weight_deltas/artifacts/qwen2_5_0_5b_instruct__synthetic_dormant__v1" \
  --max-prompts 8 \
  --max-new-tokens 32 \
  --gpu-memory-utilization 0.4
```

Pass criteria:
- Output JSON has `"status": "ok"`.
- `n_mismatches == 0`.

## D. Throughput sanity (existing benchmark harness)

1) Full dormant baseline:

```bash
python3 "/home/prushto/vllm_runtime_hostcopy_duplicate/test/speed_checks/concurrency_speed_tests.py" \
  --config "/home/prushto/vllm_runtime_hostcopy_duplicate/test/speed_checks/config_0p5b_synth.yaml" \
  --output-dir "/home/prushto/vllm_runtime_hostcopy_duplicate/test/speed_checks/test_results/lda_full_baseline" \
  --limit-prompts 128 \
  --concurrencies 1 2 4 \
  --repeats 1
```

2) Delta path candidate:
   - Keep same config and point output to a separate dir.
   - To isolate full-vs-delta comparison, run once with `dormant_delta_dir: null`,
     then once with `dormant_delta_dir` set (as in the provided config file).

Pass criteria:
- No runtime errors.
- Candidate throughput regression is within your accepted threshold.

## E. Memory sanity

- Compare memory usage at model load between full-dormant and delta path runs.
- Candidate should show a lower dormant weight footprint at steady state.
