# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LDA (logits differential amplification) GPU model runner.

Blends logits from main and dormant models with a configurable alpha for
synchronized token-by-token decoding. Uses two separate KV caches (no sharing).
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from vllm.attention.layer import Attention
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.v1.core.kv_cache_utils import scale_kv_cache_config
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.lda_kl_utils import kl_divergence_dormant_base, lda_blend_logits
from vllm.v1.worker.utils import bind_kv_cache

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class LDAGPUModelRunner(GPUModelRunner):
    """GPU model runner that blends logits from main and dormant models (LDA)."""

    def __init__(self, vllm_config: Any, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        lda = (vllm_config.additional_config or {}).get("lda")
        if not lda:
            raise ValueError(
                "LDA runner requires additional_config['lda'] with dormant_model"
            )
        self.dormant_model: str = lda["dormant_model"]
        self.lda_alpha: float = float(lda.get("lda_alpha", 0.5))
        self.dormant: torch.nn.Module | None = None
        self.dormant_kv_caches: list[torch.Tensor] = []
        self._dormant_attn_layers: dict[str, Attention] | None = None
        # Prefix used when loading the dormant model so attention layer names
        # do not collide with the main model's in static_forward_context.
        self.dormant_prefix = "dormant"

        # Optional: KL(dormant||base) stats for benchmarks (two extra softmax passes).
        self._lda_collect_kl: bool = bool(lda.get("collect_kl", False))
        self._lda_last_kl_max: float | None = None
        self._lda_last_kl_mean: float | None = None
        # Per-logits-row KL for this step (CPU), exported when collect_kl (non-spec decode).
        self._lda_kl_row_cpu: list[float] | None = None

        # Optional: assert identical logits / near-zero KL when main and dormant
        # checkpoints are the same (see test_lda_same_model_validate.py).
        env_strict = os.environ.get("VLLM_LDA_VALIDATE_SAME_CHECKPOINT", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self._lda_validate_same: bool = bool(lda.get("validate_same_checkpoint", False)) or env_strict
        self._lda_validation_rtol: float = float(lda.get("validation_logits_rtol", 0.02))
        self._lda_validation_atol: float = float(lda.get("validation_logits_atol", 0.2))
        self._lda_validation_kl_max: float = float(lda.get("validation_kl_max", 1e-3))
        if self._lda_validate_same and not self._lda_same_checkpoint_paths():
            logger.warning(
                "LDA validate_same_checkpoint is set but dormant_model (%s) != "
                "main model (%s); skipping logits/KL checks.",
                self.dormant_model,
                self.model_config.model,
            )

    def load_model(self, eep_scale_up: bool = False) -> None:
        super().load_model(eep_scale_up=eep_scale_up)
        if self.dormant is not None:
            return
        logger.info("Loading dormant model for LDA: %s", self.dormant_model)
        model_loader = get_model_loader(self.load_config)
        dormant_model_config = self._dormant_model_config()
        self.dormant = model_loader.load_model(
            vllm_config=self.vllm_config,
            model_config=dormant_model_config,
            prefix=self.dormant_prefix,
        )
        self.dormant.to(self.device)
        self.dormant.eval()

    def _dormant_model_config(self) -> ModelConfig:
        """ModelConfig for the dormant model (same arch as main, different path)."""
        return replace(self.model_config, model=self.dormant_model)

    def _lda_same_checkpoint_paths(self) -> bool:
        """True if dormant and main resolve to the same model id/path string."""
        main = self.model_config.model
        return os.path.normpath(str(main)) == os.path.normpath(str(self.dormant_model))

    def _get_dormant_attn_layers(self) -> dict[str, Attention]:
        if self._dormant_attn_layers is not None:
            return self._dormant_attn_layers
        out: dict[str, Attention] = {}
        for name, mod in self.dormant.named_modules():
            if isinstance(mod, Attention):
                out[name] = mod
        self._dormant_attn_layers = out
        return out

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        config_main = scale_kv_cache_config(kv_cache_config, 0.5)
        config_dormant = scale_kv_cache_config(kv_cache_config, 0.5)
        super().initialize_kv_cache(config_main)
        self._initialize_dormant_kv_cache(config_dormant)

    def _initialize_dormant_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate and bind KV cache for the dormant model (half of total pool)."""
        kernel_block_sizes = self._prepare_kernel_block_sizes(kv_cache_config)
        saved_config = self.kv_cache_config
        self.kv_cache_config = kv_cache_config
        try:
            kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
            kv_caches_dict = self._reshape_kv_cache_tensors(
                kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes
            )
        finally:
            self.kv_cache_config = saved_config
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            kv_caches_dict[layer_name] = kv_caches_dict[target_layer_name]
        dormant_attn = self._get_dormant_attn_layers()
        dormant_kv_caches_dict = self._match_kv_cache_keys_for_dormant(
            kv_caches_dict, dormant_attn
        )
        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        )
        self.dormant_kv_caches.clear()
        bind_kv_cache(
            dormant_kv_caches_dict,
            dormant_attn,
            self.dormant_kv_caches,
            num_attn_module=num_attn_module,
        )

    def _match_kv_cache_keys_for_dormant(
        self,
        kv_caches_dict: dict[str, torch.Tensor],
        dormant_attn: dict[str, Attention],
    ) -> dict[str, torch.Tensor]:
        """Match main KV cache names to dormant attention layer names.

        Depending on model internals, dormant attention names can be either
        root-relative ("model.layers...") or include the custom prefix
        ("dormant.model.layers..."). This maps both reliably.
        """
        out: dict[str, torch.Tensor] = {}
        missing: list[str] = []
        prefix = f"{self.dormant_prefix}."
        for dormant_name in dormant_attn:
            if dormant_name in kv_caches_dict:
                out[dormant_name] = kv_caches_dict[dormant_name]
                continue
            if dormant_name.startswith(prefix):
                base_name = dormant_name[len(prefix) :]
                if base_name in kv_caches_dict:
                    out[dormant_name] = kv_caches_dict[base_name]
                    continue
            missing.append(dormant_name)

        if missing:
            logger.error(
                "Failed to map dormant KV cache names. missing=%d sample=%s; "
                "dormant_sample=%s; kv_sample=%s",
                len(missing),
                missing[:3],
                list(dormant_attn.keys())[:3],
                list(kv_caches_dict.keys())[:3],
            )
            raise KeyError(
                f"Could not map {len(missing)} dormant attention layer names to KV cache tensors."
            )
        return out

    def _postprocess_logits(
        self,
        logits: torch.Tensor,
        *,
        scheduler_output: SchedulerOutput,
        logits_indices: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        model_kwargs: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
        attn_metadata: Any,
        num_scheduled_tokens: int,
    ) -> torch.Tensor:
        if self.dormant is None:
            self._lda_kl_row_cpu = None
            return logits
        self._lda_kl_row_cpu = None
        from vllm.forward_context import set_forward_context
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            slot_mapping=slot_mappings,
        ):
            with torch.inference_mode():
                dormant_output = self.dormant(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )
        dormant_hidden = dormant_output[logits_indices]
        dormant_logits = self.dormant.compute_logits(dormant_hidden)
        alpha = self.lda_alpha
        need_kl = self._lda_collect_kl or (
            self._lda_validate_same and self._lda_same_checkpoint_paths()
        )
        if need_kl:
            kl_row = kl_divergence_dormant_base(dormant_logits, logits)
            if self._lda_collect_kl:
                self._lda_last_kl_max = float(kl_row.max().item())
                self._lda_last_kl_mean = float(kl_row.mean().item())
                self._lda_kl_row_cpu = kl_row.detach().float().cpu().tolist()
            if self._lda_validate_same and self._lda_same_checkpoint_paths():
                if not torch.allclose(
                    dormant_logits,
                    logits,
                    rtol=self._lda_validation_rtol,
                    atol=self._lda_validation_atol,
                ):
                    diff = (dormant_logits - logits).abs()
                    raise RuntimeError(
                        "LDA same-checkpoint validation failed: dormant logits differ "
                        f"from base (max abs diff={diff.max().item():.6g})."
                    )
                kl_max = float(kl_row.max().item())
                if kl_max > self._lda_validation_kl_max:
                    raise RuntimeError(
                        "LDA same-checkpoint validation failed: KL(dormant||base) "
                        f"max={kl_max:.6g} exceeds threshold "
                        f"{self._lda_validation_kl_max:g}."
                    )
        amplified = lda_blend_logits(dormant_logits, logits, alpha)
        return amplified.to(logits.dtype).to(logits.device)

    def _lda_kl_for_sampled_tokens_step(
        self,
        valid_sampled_token_ids: list[list[int]],
        invalid_req_indices: list[int],
        max_gen_len: int,
    ) -> list[list[float]] | None:
        """Build per-request KL lists aligned with sampled tokens (non-spec decode only)."""
        del invalid_req_indices  # discard is already reflected as empty token lists
        if not self._lda_collect_kl or not self._lda_kl_row_cpu:
            return None
        if max_gen_len != 1:
            logger.warning_once(
                "LDA KL export skipped: speculative decode (max_gen_len=%s) "
                "is not supported for per-token KL.",
                max_gen_len,
            )
            return None
        n_kl = len(self._lda_kl_row_cpu)
        n_req = len(valid_sampled_token_ids)
        if n_req == 0 or n_kl != n_req:
            logger.warning_once(
                "LDA KL export skipped: length mismatch (kl_rows=%s, num_reqs=%s).",
                n_kl,
                n_req,
            )
            return None
        out: list[list[float]] = []
        for req_idx in range(n_req):
            toks = valid_sampled_token_ids[req_idx]
            if not toks:
                out.append([])
            else:
                out.append([float(self._lda_kl_row_cpu[req_idx])])
        return out
