#!/usr/bin/env python3
"""Minimal test: load Qwen2.5-0.5B-Instruct in vLLM and generate for a test prompt.

Uses the chat API (llm.chat with messages) so the model's chat template is
applied and EOS is respected; SamplingParams defaults (ignore_eos=False,
etc.) are used except for temperature and max_tokens.

There is no single-command "vllm run" CLI for one-off generation. The CLI
path is: start a server with `vllm serve <model>`, then use
`vllm openai chat` (or any OpenAI-compatible client) against it.

Run inside the vLLM dev container. Example (from repo root):

  EXPERIMENTS_DIR=$HOME/vllm_runtime_hostcopy ./docker/run_vllm_dev.sh

Then in the container:

  python test/basic_func/test_qwen_05b_generate.py
"""
from __future__ import annotations

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TEST_MESSAGES = [{"role": "user", "content": "What is 2 + 2? Reply in one short sentence."}]


def main() -> None:
    print(f"Loading model: {MODEL}")
    llm = LLM(model=MODEL)

    # Use defaults (ignore_eos=False, etc.); only set temperature and max_tokens.
    sampling_params = SamplingParams(temperature=0.7, max_tokens=128)
    print("Generating with chat API (model chat template + EOS respected)...")

    outputs = llm.chat(TEST_MESSAGES, sampling_params=sampling_params)

    result = outputs[0]
    text = result.outputs[0].text
    n_tokens = len(result.outputs[0].token_ids)
    print(f"Generated ({n_tokens} tokens): {text!r}")
    print("OK: vLLM loaded model and produced output.")


if __name__ == "__main__":
    main()
