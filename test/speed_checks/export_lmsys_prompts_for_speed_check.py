#!/usr/bin/env python3
"""
Export **lmsys-chat-1m** single-turn prompts to the CSV format expected by
`concurrency_speed_tests.py`, using the same loader + `PromptSampler` logic as
`dormant/warmup/core/run_experiments.py` (including merging `CURATED_PROMPTS`).

Run **on a machine with HuggingFace access** to the lmsys dataset (token in env).

  cd vllm_runtime_hostcopy/test/speed_checks
  python3 export_lmsys_prompts_for_speed_check.py --n-prompts 1000 --seed 42

Writes `prompts_lmsys_n{N}_s{seed}.csv` by default.

For the **normalized 5k Modal pack** (ships in `dormant/modal/data/`), use
``dormant/modal/scripts/build_lda_prompt_pack.py`` instead.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_SPEED_ROOT = Path(__file__).resolve().parent
_VLLM_HOSTCOPY_ROOT = _SPEED_ROOT.parent.parent
_DORMANT_ROOT = _VLLM_HOSTCOPY_ROOT.parent / "dormant"

if not _DORMANT_ROOT.is_dir():
    print(
        f"Expected dormant repo at {_DORMANT_ROOT} (sibling of vllm_runtime_hostcopy).",
        file=sys.stderr,
    )
    sys.exit(1)
sys.path.insert(0, str(_DORMANT_ROOT))

from warmup.core.prompts import (  # noqa: E402
    CURATED_PROMPTS,
    PromptSampler,
    load_lmsys,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-prompts", type=int, default=1000, help="How many single-turn prompts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--lmsys-rows",
        type=int,
        default=None,
        help="Max lmsys rows to scan (default: all, like run_experiments)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: prompts_lmsys_n{N}_s{seed}.csv here)",
    )
    args = p.parse_args()

    print("Loading lmsys (may download/cache on first run)...", flush=True)
    lmsys_ds = load_lmsys(max_rows=args.lmsys_rows)
    for text in CURATED_PROMPTS:
        if text and text not in lmsys_ds.single_turn:
            lmsys_ds.single_turn.append(text)

    sampler = PromptSampler(lmsys_ds, batch_size=16, seed=args.seed)
    texts = sampler.sample_single(args.n_prompts)
    if len(texts) < args.n_prompts:
        raise SystemExit(
            f"Pool has only {len(texts)} unique single-turn prompts; "
            f"requested {args.n_prompts}. Load more lmsys rows (--lmsys-rows) or lower --n-prompts."
        )

    out = args.output
    if out is None:
        out = _SPEED_ROOT / f"prompts_lmsys_n{args.n_prompts}_s{args.seed}.csv"

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "length_bucket", "prompt_text"])
        for i, text in enumerate(texts):
            w.writerow([f"lmsys-{i:05d}", "lmsys", text])

    print(f"Wrote {len(texts)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
