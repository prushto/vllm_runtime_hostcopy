# Basic functional tests

Run inside the [vLLM dev container](../../docker/run_vllm_dev.sh) (GPU) unless noted.

| Script | Purpose |
|--------|---------|
| `test_qwen_05b_generate.py` | Single-model vLLM smoke (Qwen2.5-0.5B). |
| `test_lda_generate.py` | LDA with distinct base/dormant checkpoints. |
| `test_lda_same_model_validate.py` | **Correctness:** same checkpoint for main + dormant; use `--preset 0.5b` (default) or `--preset 7b`. |
| `test_lda_same_model_validate_7b.py` | Same as `test_lda_same_model_validate.py --preset 7b` (Qwen2.5-7B; `gpu_memory_utilization=0.4`, `max_model_len=8192`, looser validation tolerances). |
| `test_lda_kl_utils.py` | **CPU:** KL and blend math (run `python3 test/basic_func/test_lda_kl_utils.py` with `torch` installed). |

**7B OOM:** Preset uses `gpu_memory_utilization=0.4` (aligned with `test_lda_generate.py` and `test/speed_checks/config.yaml`). If you still OOM, use `--max-model-len 4096` (or lower).

## LDA `additional_config["lda"]` keys (optional)

Merged with `dormant_model` / `lda_alpha` from `LLM(...)` kwargs. Useful keys:

- **`validate_same_checkpoint`** (bool): When `dormant_model` equals `model`, assert dormant logits ≈ base logits and KL(dormant‖base) is small each forward. Also set env `VLLM_LDA_VALIDATE_SAME_CHECKPOINT=1`.
- **`validation_logits_rtol`**, **`validation_logits_atol`**: `torch.allclose` tolerances (defaults tuned for bf16).
- **`validation_kl_max`**: Maximum allowed KL(dormant‖base) per row during validation.
- **`collect_kl`** (bool): Each LDA forward computes KL and stores `max`/`mean` on the runner (`_lda_last_kl_max`, `_lda_last_kl_mean`) for benchmarking—**two extra softmax passes** vs blend-only.
