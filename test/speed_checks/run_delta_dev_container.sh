#!/usr/bin/env bash
set -euo pipefail

# Dedicated dev/test container for delta-weight work in this repo.
# This script is intentionally strict: it verifies mounts to prevent accidental
# writes/runs against non-duplicate paths.
#
# Usage:
#   test/speed_checks/run_delta_dev_container.sh start
#   test/speed_checks/run_delta_dev_container.sh verify
#   test/speed_checks/run_delta_dev_container.sh exec
#   test/speed_checks/run_delta_dev_container.sh status
#   test/speed_checks/run_delta_dev_container.sh stop
#
# Optional env overrides:
#   IMAGE=nvcr.io/nvidia/vllm:26.02-py3
#   CONTAINER_NAME=vllm-delta-dev
#   HOST_REPO=/home/prushto/vllm_runtime_hostcopy
#   HOST_DORMANT=/home/prushto/dormant
#   HF_CACHE_HOST=/home/prushto/.cache/huggingface

IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.02-py3}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-delta-dev}"
HOST_REPO="${HOST_REPO:-$HOME/vllm_runtime_hostcopy}"
HOST_DORMANT="${HOST_DORMANT:-$HOME/dormant}"
HF_CACHE_HOST="${HF_CACHE_HOST:-$HOME/.cache/huggingface}"

WORKSPACE="/workspace"
HF_CACHE_CONTAINER="/hf-cache"
SITE_PACKAGES_VLLM="/usr/local/lib/python3.12/dist-packages/vllm"
TARGET_RUNNER_FILE="${SITE_PACKAGES_VLLM}/v1/worker/lda_gpu_model_runner.py"
TARGET_ARGUTILS_FILE="${SITE_PACKAGES_VLLM}/engine/arg_utils.py"

EXPECTED_WORKSPACE_SOURCE="$(realpath "${HOST_REPO}")"
EXPECTED_RUNNER_SOURCE="$(realpath "${HOST_REPO}/vllm/v1/worker/lda_gpu_model_runner.py")"
EXPECTED_ARGUTILS_SOURCE="$(realpath "${HOST_REPO}/vllm/engine/arg_utils.py")"
EXPECTED_DORMANT_SOURCE="$(realpath "${HOST_DORMANT}")"
EXPECTED_HF_SOURCE="$(realpath "${HF_CACHE_HOST}")"

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required but not found in PATH." >&2
    exit 1
  fi
}

container_exists() {
  docker ps -a --format '{{.Names}}' | awk '{print $1}' | grep -Fxq "${CONTAINER_NAME}"
}

container_running() {
  docker ps --format '{{.Names}}' | awk '{print $1}' | grep -Fxq "${CONTAINER_NAME}"
}

verify_mounts() {
  if ! container_exists; then
    echo "Container ${CONTAINER_NAME} does not exist." >&2
    exit 1
  fi

  local mounts
  mounts="$(docker inspect "${CONTAINER_NAME}" --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}')"
  echo "${mounts}"

  local workspace_line="${EXPECTED_WORKSPACE_SOURCE} -> ${WORKSPACE}"
  local runner_line="${EXPECTED_RUNNER_SOURCE} -> ${TARGET_RUNNER_FILE}"
  local argutils_line="${EXPECTED_ARGUTILS_SOURCE} -> ${TARGET_ARGUTILS_FILE}"
  local dormant_line="${EXPECTED_DORMANT_SOURCE} -> ${WORKSPACE}/dormant"
  local hf_line="${EXPECTED_HF_SOURCE} -> ${HF_CACHE_CONTAINER}"

  if ! grep -Fqx "${workspace_line}" <<<"${mounts}"; then
    echo "Mount check failed: expected ${workspace_line}" >&2
    exit 1
  fi
  if ! grep -Fqx "${runner_line}" <<<"${mounts}"; then
    echo "Mount check failed: expected ${runner_line}" >&2
    exit 1
  fi
  if ! grep -Fqx "${argutils_line}" <<<"${mounts}"; then
    echo "Mount check failed: expected ${argutils_line}" >&2
    exit 1
  fi
  if ! grep -Fqx "${dormant_line}" <<<"${mounts}"; then
    echo "Mount check failed: expected ${dormant_line}" >&2
    exit 1
  fi
  if ! grep -Fqx "${hf_line}" <<<"${mounts}"; then
    echo "Mount check failed: expected ${hf_line}" >&2
    exit 1
  fi
}

start_container() {
  mkdir -p "${HOST_DORMANT}"
  mkdir -p "${HF_CACHE_HOST}"
  if container_exists; then
    echo "Container ${CONTAINER_NAME} already exists."
    if container_running; then
      echo "Already running."
    fi
    verify_mounts
    return
  fi

  docker run -d --rm \
    --gpus all \
    --name "${CONTAINER_NAME}" \
    --shm-size=16g \
    -u "$(id -u):$(id -g)" \
    -e HF_HOME="${HF_CACHE_CONTAINER}" \
    -e HUGGINGFACE_HUB_CACHE="${HF_CACHE_CONTAINER}/hub" \
    -e TRANSFORMERS_CACHE="${HF_CACHE_CONTAINER}/transformers" \
    -e HF_DATASETS_CACHE="${HF_CACHE_CONTAINER}/datasets" \
    -v "${EXPECTED_RUNNER_SOURCE}:${TARGET_RUNNER_FILE}" \
    -v "${EXPECTED_ARGUTILS_SOURCE}:${TARGET_ARGUTILS_FILE}" \
    -v "${EXPECTED_WORKSPACE_SOURCE}:${WORKSPACE}" \
    -v "${EXPECTED_DORMANT_SOURCE}:${WORKSPACE}/dormant" \
    -v "${EXPECTED_HF_SOURCE}:${HF_CACHE_CONTAINER}" \
    -w /tmp \
    "${IMAGE}" \
    bash -lc "sleep infinity" >/dev/null

  verify_mounts
  echo "Started ${CONTAINER_NAME}."
}

exec_shell() {
  if ! container_running; then
    echo "Container ${CONTAINER_NAME} is not running." >&2
    exit 1
  fi
  verify_mounts >/dev/null
  docker exec -it "${CONTAINER_NAME}" bash
}

show_status() {
  if ! container_exists; then
    echo "Container ${CONTAINER_NAME}: not created"
    return
  fi
  docker ps -a --filter "name=^${CONTAINER_NAME}$" \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
}

stop_container() {
  if container_running; then
    docker stop "${CONTAINER_NAME}" >/dev/null
    echo "Stopped ${CONTAINER_NAME}."
  else
    echo "Container ${CONTAINER_NAME} is not running."
  fi
}

main() {
  require_docker
  local cmd="${1:-status}"
  case "${cmd}" in
    start) start_container ;;
    verify) verify_mounts ;;
    exec) exec_shell ;;
    status) show_status ;;
    stop) stop_container ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      echo "Usage: $0 {start|verify|exec|status|stop}" >&2
      exit 1
      ;;
  esac
}

main "$@"
