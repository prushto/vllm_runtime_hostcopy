# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LDA (logits differential amplification) GPU model runner.

Blends logits from main and dormant models with a configurable alpha for
synchronized token-by-token decoding. Uses two separate KV caches (no sharing).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open

from vllm.attention.layer import Attention
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import bind_kv_cache

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


def _page_size_for_tensor(config: KVCacheConfig, layer_name: str) -> int:
    """Return page_size_bytes for the group that contains the given layer."""
    for group in config.kv_cache_groups:
        if layer_name in group.layer_names:
            return group.kv_cache_spec.page_size_bytes
    raise KeyError(f"Layer {layer_name!r} not in any kv_cache_group")


def _scale_kv_cache_config(config: KVCacheConfig, scale: float) -> KVCacheConfig:
    """Return a new KVCacheConfig with num_blocks and tensor sizes scaled.

    Scaled tensor sizes are aligned down to a multiple of page_size_bytes so
    that _reshape_kv_cache_tensors' assertion (raw_tensor.numel() % page_size_bytes == 0)
    holds.
    """
    new_num_blocks = max(1, int(config.num_blocks * scale))
    new_tensors = []
    for t in config.kv_cache_tensors:
        if not t.shared_by:
            new_tensors.append(
                KVCacheTensor(size=max(1, int(t.size * scale)), shared_by=list(t.shared_by))
            )
            continue
        page_size = _page_size_for_tensor(config, t.shared_by[0])
        scaled = max(1, int(t.size * scale))
        # Align down to a multiple of page_size so reshape assertion holds.
        size = (scaled // page_size) * page_size
        size = max(page_size, size)
        new_tensors.append(KVCacheTensor(size=size, shared_by=list(t.shared_by)))
    return KVCacheConfig(
        num_blocks=new_num_blocks,
        kv_cache_tensors=new_tensors,
        kv_cache_groups=config.kv_cache_groups,
    )


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
        self.dormant_delta_dir: str | None = lda.get("dormant_delta_dir")
        self.dormant_base_model: str | None = lda.get("dormant_base_model")
        self.dormant: torch.nn.Module | None = None
        self.dormant_kv_caches: list[torch.Tensor] = []
        self._dormant_attn_layers: dict[str, Attention] | None = None
        # Prefix used when loading the dormant model so attention layer names
        # do not collide with the main model's in static_forward_context.
        self.dormant_prefix = "dormant"

    def load_model(self, eep_scale_up: bool = False) -> None:
        super().load_model(eep_scale_up=eep_scale_up)
        if self.dormant is not None:
            return
        if self.dormant_delta_dir:
            logger.info(
                "Loading dormant model for LDA using delta artifacts: dormant=%s delta_dir=%s",
                self.dormant_model,
                self.dormant_delta_dir,
            )
        else:
            logger.info("Loading dormant model for LDA: %s", self.dormant_model)
        model_loader = get_model_loader(self.load_config)
        dormant_model_config = self._dormant_model_config()
        self.dormant = model_loader.load_model(
            vllm_config=self.vllm_config,
            model_config=dormant_model_config,
            prefix=self.dormant_prefix,
        )
        if self.dormant_delta_dir:
            self._assemble_dormant_from_delta()
            # Delta application mutates parameter data after the loader has
            # already run post-load processing; refresh derived packed weights.
            process_weights_after_loading(
                self.dormant, dormant_model_config, self.device
            )
        self.dormant.to(self.device)
        self.dormant.eval()

    def _dormant_model_config(self) -> ModelConfig:
        """ModelConfig for dormant path.

        If delta artifacts are provided, initialize dormant from the configured
        dormant base model (or main model by default), then apply deltas.
        """
        if self.dormant_delta_dir:
            base_model = self.dormant_base_model or self.model_config.model
            return replace(self.model_config, model=base_model)
        return replace(self.model_config, model=self.dormant_model)

    def _normalize_param_name(self, name: str) -> str:
        prefix = f"{self.dormant_prefix}."
        if name.startswith(prefix):
            return name[len(prefix) :]
        return name

    def _build_param_lookup(
        self, module: torch.nn.Module
    ) -> dict[str, torch.nn.Parameter]:
        out: dict[str, torch.nn.Parameter] = {}
        for raw_name, param in module.named_parameters(remove_duplicate=False):
            normalized = self._normalize_param_name(raw_name)
            out[normalized] = param
        return out

    def _load_delta_artifact(self) -> tuple[Path, dict[str, Any]]:
        assert self.dormant_delta_dir is not None
        delta_dir = Path(self.dormant_delta_dir)
        if not delta_dir.exists():
            raise FileNotFoundError(
                f"LDA dormant_delta_dir does not exist: {delta_dir}"
            )
        manifest_path = delta_dir / "delta_manifest.json"
        delta_path = delta_dir / "delta.safetensors"
        if not manifest_path.exists() or not delta_path.exists():
            raise FileNotFoundError(
                "LDA dormant delta artifacts missing: expected "
                f"{manifest_path} and {delta_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        format_version = manifest.get("format_version")
        if format_version != "1.0":
            raise ValueError(
                f"Unsupported delta manifest format_version={format_version!r}; expected '1.0'"
            )
        changed = manifest.get("changed_tensors")
        to_reference = manifest.get("to_reference")
        if not isinstance(changed, list) or not isinstance(to_reference, list):
            raise ValueError(
                "Invalid delta manifest: expected list fields changed_tensors and to_reference"
            )
        dormant_manifest = manifest.get("dormant_model_id_or_path")
        if isinstance(dormant_manifest, str) and dormant_manifest != self.dormant_model:
            logger.warning(
                "Delta manifest dormant model (%s) differs from configured dormant model (%s).",
                dormant_manifest,
                self.dormant_model,
            )
        return delta_dir, manifest

    def _assemble_dormant_from_delta(self) -> None:
        assert self.model is not None
        assert self.dormant is not None
        delta_dir, manifest = self._load_delta_artifact()
        delta_path = delta_dir / "delta.safetensors"

        changed_entries = manifest["changed_tensors"]
        changed_names = [entry.get("name") for entry in changed_entries]
        if any(not isinstance(name, str) for name in changed_names):
            raise ValueError("Invalid delta manifest: changed_tensors entries must include name")
        changed_name_set = set(changed_names)
        to_reference = manifest["to_reference"]
        if any(not isinstance(name, str) for name in to_reference):
            raise ValueError("Invalid delta manifest: to_reference must contain strings")
        to_reference_set = set(to_reference)

        if changed_name_set & to_reference_set:
            overlap = sorted(changed_name_set & to_reference_set)[:5]
            raise ValueError(
                "Invalid delta manifest: changed_tensors and to_reference overlap. "
                f"sample={overlap}"
            )

        base_params = self._build_param_lookup(self.model)
        dormant_params = self._build_param_lookup(self.dormant)

        common_names = set(base_params) & set(dormant_params)
        covered = changed_name_set | to_reference_set
        missing = common_names - covered
        extra = covered - common_names
        if missing or extra:
            raise ValueError(
                "Delta coverage mismatch for dormant assembly: "
                f"missing_common={len(missing)} extra_non_common={len(extra)}"
            )

        # Apply changed tensors from delta artifact.
        changed_applied = 0
        with safe_open(str(delta_path), framework="pt") as delta_file:
            delta_keys = set(delta_file.keys())
            if delta_keys != changed_name_set:
                only_delta = len(delta_keys - changed_name_set)
                only_manifest = len(changed_name_set - delta_keys)
                raise ValueError(
                    "delta.safetensors keys mismatch changed_tensors in manifest: "
                    f"only_delta={only_delta} only_manifest={only_manifest}"
                )
            for entry in changed_entries:
                name = entry["name"]
                target = dormant_params[name]
                src = delta_file.get_tensor(name)
                if tuple(src.shape) != tuple(target.shape):
                    raise ValueError(
                        f"Shape mismatch for changed tensor {name}: "
                        f"delta={tuple(src.shape)} target={tuple(target.shape)}"
                    )
                if str(src.dtype) != str(target.dtype):
                    raise ValueError(
                        f"Dtype mismatch for changed tensor {name}: "
                        f"delta={src.dtype} target={target.dtype}"
                    )
                target.data.copy_(src.to(device=target.device, dtype=target.dtype))
                changed_applied += 1

        # Alias unchanged dormant params to base params.
        referenced = 0
        for name in to_reference:
            dormant_p = dormant_params[name]
            base_p = base_params[name]
            if tuple(dormant_p.shape) != tuple(base_p.shape):
                raise ValueError(
                    f"Shape mismatch for referenced tensor {name}: "
                    f"dormant={tuple(dormant_p.shape)} base={tuple(base_p.shape)}"
                )
            if str(dormant_p.dtype) != str(base_p.dtype):
                raise ValueError(
                    f"Dtype mismatch for referenced tensor {name}: "
                    f"dormant={dormant_p.dtype} base={base_p.dtype}"
                )
            dormant_p.data = base_p.data
            referenced += 1

        logger.info(
            "Dormant model assembled from delta artifacts. changed=%d referenced=%d common=%d",
            changed_applied,
            referenced,
            len(common_names),
        )

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
        config_main = _scale_kv_cache_config(kv_cache_config, 0.5)
        config_dormant = _scale_kv_cache_config(kv_cache_config, 0.5)
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
            return logits
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
        amplified = dormant_logits + alpha * (dormant_logits - logits)
        return amplified.to(logits.dtype).to(logits.device)
