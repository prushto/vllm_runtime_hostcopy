#!/usr/bin/env python3
"""Same as ``test_lda_same_model_validate.py --preset 7b`` (Qwen2.5-7B-Instruct).

Run inside the vLLM dev container:

  EXPERIMENTS_DIR=$HOME/vllm_runtime_hostcopy ./docker/run_vllm_dev.sh
  python3 test/basic_func/test_lda_same_model_validate_7b.py

If you OOM, try a smaller KV budget:

  python3 test/basic_func/test_lda_same_model_validate_7b.py --max-model-len 4096
"""
from __future__ import annotations

import sys

from test_lda_same_model_validate import main


if __name__ == "__main__":
    main(["--preset", "7b", *sys.argv[1:]])
