#!/usr/bin/env python3
"""Quick guard checks for delta manifest consistency.

These checks validate artifact integrity before runtime assembly is attempted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate delta manifest guardrails.")
    p.add_argument("--artifact-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.artifact_dir / "delta_manifest.json"
    delta_path = args.artifact_dir / "delta.safetensors"
    if not manifest_path.exists() or not delta_path.exists():
        raise SystemExit(
            f"Missing required artifact files under {args.artifact_dir}: "
            "delta_manifest.json and delta.safetensors"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != "1.0":
        raise SystemExit(
            f"Unsupported format_version={manifest.get('format_version')!r}; expected '1.0'"
        )

    changed = manifest.get("changed_tensors")
    to_reference = manifest.get("to_reference")
    if not isinstance(changed, list) or not isinstance(to_reference, list):
        raise SystemExit("Manifest must contain list fields changed_tensors and to_reference.")

    changed_names = [entry.get("name") for entry in changed]
    if any(not isinstance(name, str) for name in changed_names):
        raise SystemExit("Each changed_tensors entry must contain string key 'name'.")
    changed_set = set(changed_names)
    ref_set = set(to_reference)
    if any(not isinstance(x, str) for x in to_reference):
        raise SystemExit("to_reference must be a list of strings.")
    if changed_set & ref_set:
        overlap = sorted(changed_set & ref_set)[:10]
        raise SystemExit(
            f"changed_tensors and to_reference overlap unexpectedly. sample={overlap}"
        )

    with safe_open(str(delta_path), framework="pt") as sf:
        delta_keys = set(sf.keys())
    if delta_keys != changed_set:
        only_delta = len(delta_keys - changed_set)
        only_manifest = len(changed_set - delta_keys)
        raise SystemExit(
            "delta.safetensors keys mismatch changed_tensors set. "
            f"only_delta={only_delta} only_manifest={only_manifest}"
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "artifact_dir": str(args.artifact_dir),
                "changed_tensors_count": len(changed_set),
                "to_reference_count": len(ref_set),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

