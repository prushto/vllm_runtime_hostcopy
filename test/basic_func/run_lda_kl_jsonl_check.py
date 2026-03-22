#!/usr/bin/env python3
"""Local end-to-end check: LDA + ``collect_kl`` + ``generations.jsonl`` with KL fields.

Runs ``test/speed_checks/concurrency_speed_tests.py`` with
``test/basic_func/config_lda_kl_jsonl_check.yaml``, then asserts each JSONL row has
finite ``mean_kl_divergence`` and ``max_kl_divergence`` (non-spec decode path).

**Setup**

- GPU machine (e.g. DGX Spark) with PyTorch + vLLM from this fork.
- From repo root ``vllm_runtime_hostcopy``::

    pip install -e .   # or export PYTHONPATH=$PWD

**Run**::

    python test/basic_func/run_lda_kl_jsonl_check.py

Optional: ``--keep-output-dir`` to skip deleting the run folder (default uses a temp dir).

Edit ``config_lda_kl_jsonl_check.yaml`` if you use local checkpoint paths instead of Hub ids.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


_BASIC_FUNC = Path(__file__).resolve().parent
_REPO_ROOT = _BASIC_FUNC.parent.parent
_SPEED_CHECKS = _BASIC_FUNC.parent / "speed_checks"
_CONFIG = _BASIC_FUNC / "config_lda_kl_jsonl_check.yaml"
_HARNESS = _SPEED_CHECKS / "concurrency_speed_tests.py"
_RUN_ID = "local_lda_kl_jsonl_check"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--keep-output-dir",
        action="store_true",
        help="Do not delete the output parent directory after the check.",
    )
    p.add_argument(
        "--output-parent",
        type=Path,
        default=None,
        help="Directory for the run folder (default: tempfile under /tmp).",
    )
    return p.parse_args()


def _validate_jsonl(path: Path) -> None:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"Empty or missing JSONL: {path}")
    bad: list[str] = []
    lda_rows = 0
    for i, ln in enumerate(lines, start=1):
        row = json.loads(ln)
        scenario = row.get("scenario", "")
        if scenario != "lda":
            continue
        lda_rows += 1
        for key in ("mean_kl_divergence", "max_kl_divergence"):
            v = row.get(key)
            if v is None or not isinstance(v, (int, float)):
                bad.append(f"line {i}: {key}={v!r}")
            elif isinstance(v, float) and (v != v):  # NaN
                bad.append(f"line {i}: {key} is NaN")
    if lda_rows == 0:
        raise SystemExit("No rows with scenario==\"lda\" in JSONL")
    if bad:
        raise SystemExit(
            "KL fields missing or invalid on lda rows:\n  " + "\n  ".join(bad)
        )


def main() -> int:
    args = _parse_args()
    for p, label in (
        (_CONFIG, "config"),
        (_HARNESS, "harness"),
    ):
        if not p.is_file():
            print(f"Missing {label}: {p}", file=sys.stderr)
            return 1

    if args.output_parent is not None:
        work = args.output_parent.resolve()
        work.mkdir(parents=True, exist_ok=True)
        out_parent = work / "results"
        out_parent.mkdir(exist_ok=True)
        prompts_csv = work / "prompts.csv"
        cleanup = False
    else:
        work = Path(tempfile.mkdtemp(prefix="lda_kl_jsonl_check_"))
        out_parent = work / "results"
        out_parent.mkdir()
        prompts_csv = work / "prompts.csv"
        cleanup = not args.keep_output_dir

    run_dir = out_parent / _RUN_ID

    cmd = [
        sys.executable,
        str(_HARNESS),
        "--config",
        str(_CONFIG),
        "--output-dir",
        str(out_parent),
        "--run-id",
        _RUN_ID,
        "--prompts-csv",
        str(prompts_csv),
        "--limit-prompts",
        "3",
    ]
    print("Running:", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    rc = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env).returncode
    if rc != 0:
        print(f"Harness exited {rc}", file=sys.stderr)
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)
        return rc

    jsonl = run_dir / "generations.jsonl"
    if not jsonl.is_file():
        print(f"Missing {jsonl}", file=sys.stderr)
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)
        return 1

    try:
        _validate_jsonl(jsonl)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)
        return 1

    print(f"OK: KL fields present on lda rows in {jsonl}")
    print(f"Run directory: {run_dir}")
    if cleanup:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"Output kept under: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
