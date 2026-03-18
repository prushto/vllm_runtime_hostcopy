#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.02-py3}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-dev}"
HOST_VLLM_DIR="${HOST_VLLM_DIR:-$HOME/vllm_runtime_hostcopy/vllm}"
HF_CACHE_HOST="${HF_CACHE_HOST:-$HOME/.cache/huggingface}"
HF_CACHE_CONTAINER="${HF_CACHE_CONTAINER:-/hf-cache}"

# Optional: mount an experiment repo into /workspace.
# Example:
#   EXPERIMENTS_DIR=~/my-experiments ./docker/run_vllm_dev.sh
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-}"

SITE_PACKAGES_VLLM="/usr/local/lib/python3.12/dist-packages/vllm"

mkdir -p "${HF_CACHE_HOST}"

DOCKER_ARGS=(
  run --gpus all -it --rm
  --name "${CONTAINER_NAME}-$(date +%s)"
  --shm-size=16g
  -u "$(id -u):$(id -g)"
  -e HF_HOME="${HF_CACHE_CONTAINER}"
  -e HUGGINGFACE_HUB_CACHE="${HF_CACHE_CONTAINER}/hub"
  -e TRANSFORMERS_CACHE="${HF_CACHE_CONTAINER}/transformers"
  -e HF_DATASETS_CACHE="${HF_CACHE_CONTAINER}/datasets"
  -v "${HOST_VLLM_DIR}:${SITE_PACKAGES_VLLM}"
)

if [[ -n "${EXPERIMENTS_DIR}" ]]; then
  DOCKER_ARGS+=(
    -v "${EXPERIMENTS_DIR}:/workspace"
    -w /workspace
  )
else
  DOCKER_ARGS+=(
    -w /tmp
  )
fi

# Mount HF cache after /workspace (if any) so nested mount ordering
# cannot hide it when EXPERIMENTS_DIR is bound to /workspace.
DOCKER_ARGS+=(
  -v "${HF_CACHE_HOST}:${HF_CACHE_CONTAINER}"
)

DOCKER_ARGS+=(
  "${IMAGE}"
  bash
)

docker "${DOCKER_ARGS[@]}"