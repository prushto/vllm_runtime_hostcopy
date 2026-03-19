#!/usr/bin/env python3
"""Prepare a synthetic dormant checkpoint and delta artifacts for Qwen 0.5B.

This creates a local dormant model directory by copying the base snapshot and
applying tiny deterministic perturbations to a few selected tensors, then runs
delta_builder.py to produce the corresponding delta artifact directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


def _default_workspace_root() -> Path:
    if Path("/workspace/dormant").exists():
        return Path("/workspace")
    return Path("/home/prushto")


def parse_args() -> argparse.Namespace:
    root = _default_workspace_root()
    p = argparse.ArgumentParser(description="Prepare synthetic Qwen 0.5B dormant + delta.")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--synthetic-dormant-dir",
        type=Path,
        default=root / "dormant/weight_deltas/synthetic_models/qwen2_5_0_5b_synth_dormant",
    )
    p.add_argument(
        "--delta-artifact-dir",
        type=Path,
        default=root / "dormant/weight_deltas/artifacts/qwen2_5_0_5b_instruct__synthetic_dormant__v1",
    )
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--max-perturb-tensors",
        type=int,
        default=4,
        help="How many matching tensors to perturb.",
    )
    return p.parse_args()


def load_weight_map(model_root: Path) -> dict[str, str]:
    index_path = model_root / "model.safetensors.index.json"
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        wm = data.get("weight_map")
        if not isinstance(wm, dict):
            raise ValueError("Invalid weight_map in model index.")
        return {str(k): str(v) for k, v in wm.items()}

    single = model_root / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as sf:
            return {k: "model.safetensors" for k in sf.keys()}

    safes = sorted(model_root.glob("*.safetensors"))
    if len(safes) == 1:
        with safe_open(str(safes[0]), framework="pt") as sf:
            return {k: safes[0].name for k in sf.keys()}

    raise FileNotFoundError(
        f"Could not find model.safetensors(.index.json) under {model_root}"
    )


def resolve_base_root(base_model: str, cache_dir: str | None, local_files_only: bool) -> Path:
    base_path = Path(base_model)
    if base_path.is_dir():
        return base_path.resolve()
    local_dir = snapshot_download(
        repo_id=base_model,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.tiktoken"],
    )
    return Path(local_dir).resolve()


def copy_base_to_synthetic(base_root: Path, synthetic_dir: Path) -> None:
    if synthetic_dir.exists():
        shutil.rmtree(synthetic_dir)
    # Dereference symlinks from HF snapshots so the synthetic dir is standalone.
    shutil.copytree(base_root, synthetic_dir, symlinks=False)


def perturb_selected_tensors(
    *,
    model_root: Path,
    max_perturb_tensors: int,
) -> list[str]:
    weight_map = load_weight_map(model_root)
    candidates = sorted(
        n
        for n in weight_map
        if n.endswith(".mlp.gate_proj.weight")
        or n.endswith(".mlp.up_proj.weight")
        or n.endswith(".mlp.down_proj.weight")
    )
    selected = candidates[:max_perturb_tensors]
    if not selected:
        raise RuntimeError("No candidate MLP tensors found for synthetic perturbation.")

    # Group selected tensor names by shard file so we only rewrite touched shards.
    shard_to_names: dict[str, list[str]] = {}
    for name in selected:
        shard_to_names.setdefault(weight_map[name], []).append(name)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    for shard_name, names in shard_to_names.items():
        shard_path = model_root / shard_name
        with safe_open(str(shard_path), framework="pt") as sf:
            all_tensors = {k: sf.get_tensor(k) for k in sf.keys()}

        for name in names:
            t = all_tensors[name]
            # Tiny deterministic perturbation: modify a single element.
            idx = [0] * t.ndim
            noise = torch.tensor(0.001, dtype=t.dtype)
            all_tensors[name] = t.clone()
            all_tensors[name][tuple(idx)] = all_tensors[name][tuple(idx)] + noise

        save_file(all_tensors, str(shard_path))

    return selected


def run_delta_builder(base_model: str, synthetic_dormant_dir: Path, delta_artifact_dir: Path) -> None:
    # Import from existing delta tooling path.
    import sys

    candidate_paths = [
        Path("/workspace/dormant/weight_deltas"),
        Path("/home/prushto/dormant/weight_deltas"),
    ]
    for p in candidate_paths:
        if p.exists():
            sys.path.insert(0, str(p))
            break
    from delta_builder import build_delta  # type: ignore

    build_delta(
        base_model=base_model,
        dormant_model=str(synthetic_dormant_dir),
        output_dir=delta_artifact_dir,
        cache_dir=None,
        local_files_only=True,
        include_substrs=[],
        exclude_substrs=[],
        fail_on_missing=True,
    )


def main() -> None:
    args = parse_args()
    base_root = resolve_base_root(args.base_model, args.cache_dir, args.local_files_only)
    copy_base_to_synthetic(base_root, args.synthetic_dormant_dir)
    touched = perturb_selected_tensors(
        model_root=args.synthetic_dormant_dir,
        max_perturb_tensors=args.max_perturb_tensors,
    )
    run_delta_builder(args.base_model, args.synthetic_dormant_dir, args.delta_artifact_dir)

    print("Synthetic dormant + delta artifacts prepared.")
    print(f"  base_model: {args.base_model}")
    print(f"  synthetic_dormant_dir: {args.synthetic_dormant_dir}")
    print(f"  delta_artifact_dir: {args.delta_artifact_dir}")
    print(f"  perturbed_tensors_count: {len(touched)}")
    for n in touched:
        print(f"    - {n}")


if __name__ == "__main__":
    main()

