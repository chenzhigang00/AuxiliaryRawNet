#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-all}"
DEVICE_ID="${DEVICE_ID:-0}"
DEVICE_IDS="${DEVICE_IDS:-0,1}"
ENV_NAME="${ENV_NAME:-env_asv_public_aarch64}"
HPARAMS="${HPARAMS:-yaml/RawSNet.yaml}"
ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
MASTER_PORT="${MASTER_PORT:-29501}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

usage() {
  cat <<EOF
Usage:
  bash run_npu_experiment.sh [train|train_ddp2|infer|metrics|all]

Modes:
  train      Run single-NPU training only.
  train_ddp2 Run 2-NPU Ascend DDP training.
  infer      Run checkpoint evaluation only and write predictions/scores.txt.
  metrics    Run eval.py only using existing predictions/scores.txt.
  all        Run train, infer, and metrics in sequence.

Environment overrides:
  DEVICE_ID   Physical Ascend device id to expose. Default: 0
  DEVICE_IDS  Visible Ascend device ids for DDP. Default: 0,1
  ENV_NAME    Conda environment name. Default: env_asv_public_aarch64
  HPARAMS     Hyperparameter yaml path. Default: yaml/RawSNet.yaml
  ASCEND_ENV  Ascend toolkit env script. Default: /usr/local/Ascend/ascend-toolkit/set_env.sh
  MASTER_PORT torchrun master port for DDP. Default: 29501
  OMP_NUM_THREADS CPU threads per worker for torchrun. Default: 1
EOF
}

if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

case "${MODE}" in
  train|train_ddp2|infer|metrics|all) ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ ! -f "${ASCEND_ENV}" ]]; then
  echo "Ascend toolkit env script not found: ${ASCEND_ENV}" >&2
  exit 1
fi

source "${ASCEND_ENV}"
export OMP_NUM_THREADS

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

python - <<'PY'
import sys

missing = []
for name in ("torch", "torch_npu", "speechbrain"):
    try:
        __import__(name)
    except Exception:
        missing.append(name)

if missing:
    raise SystemExit("Missing Python packages: {}".format(", ".join(missing)))

import torch
import torch_npu  # noqa: F401

if not hasattr(torch, "npu") or not torch.npu.is_available():
    raise SystemExit("torch.npu is not available. Check the Ascend runtime and environment.")

print("torch:", torch.__version__)
print("npu_available:", torch.npu.is_available())
print("npu_count:", torch.npu.device_count())
PY

mkdir -p logs predictions

run_train() {
  ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}" \
    python train_raw_net.py "${HPARAMS}" --mode train --device npu:0 \
    2>&1 | tee "logs/train_npu${DEVICE_ID}.log"
}

run_train_ddp2() {
  ASCEND_RT_VISIBLE_DEVICES="${DEVICE_IDS}" \
    torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" \
    train_raw_net.py "${HPARAMS}" --mode train --device npu:0 \
    --distributed_backend hccl \
    2>&1 | tee "logs/train_ddp2_$(echo "${DEVICE_IDS}" | tr ',' '_').log"
}

run_infer() {
  ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}" \
    python train_raw_net.py "${HPARAMS}" --mode eval --device npu:0 \
    2>&1 | tee "logs/infer_npu${DEVICE_ID}.log"
}

run_metrics() {
  python eval.py 2>&1 | tee "logs/metrics.log"
}

case "${MODE}" in
  train)
    run_train
    ;;
  train_ddp2)
    run_train_ddp2
    ;;
  infer)
    run_infer
    ;;
  metrics)
    run_metrics
    ;;
  all)
    run_train
    run_infer
    run_metrics
    ;;
esac
