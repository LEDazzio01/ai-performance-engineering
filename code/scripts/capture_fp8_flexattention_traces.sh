#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/artifacts/nsight/fp8_flexattention}"
NSYS_DIR="${ARTIFACT_DIR}/nsys"
NCU_DIR="${ARTIFACT_DIR}/ncu"

TRACE_DURATION="${TRACE_DURATION:-60}"
TARGET_QPS="${TARGET_QPS:-28}"
PROMPT_LEN_MIN="${PROMPT_LEN_MIN:-512}"
PROMPT_LEN_MAX="${PROMPT_LEN_MAX:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TENSOR_PARALLEL_GPUS="${TENSOR_PARALLEL_GPUS:-8}"

mkdir -p "${NSYS_DIR}" "${NCU_DIR}"

echo "[capture] Artifacts will be written to: ${ARTIFACT_DIR}"

export PYTHONPATH="${ROOT_DIR}"

# Warmup to keep compile overhead out of traces
echo "[capture] Running warmup benchmark..."
python "${ROOT_DIR}/ch16/test_gpt_large_optimized.py" \
  --tensor-parallel-gpus "${TENSOR_PARALLEL_GPUS}" \
  --attention-backend flex \
  --fp8-mode transformer-engine \
  --warmup 2 \
  --iters 4 \
  --output-json "${ARTIFACT_DIR}/warmup.json"

TORCHRUN_CMD=(
  torchrun --nproc_per_node="${TENSOR_PARALLEL_GPUS}"
  "${ROOT_DIR}/ch16/inference_server_load_test.py"
  --duration "${TRACE_DURATION}"
  --target-qps "${TARGET_QPS}"
  --prompt-len-min "${PROMPT_LEN_MIN}"
  --prompt-len-max "${PROMPT_LEN_MAX}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --attention-backend flex
  --fp8-mode transformer-engine
)

echo "[capture] Capturing Nsight Systems trace..."
nsys profile --force-overwrite true \
  --output "${NSYS_DIR}/flexattention_fp8" \
  --sample=none \
  --trace=cuda,nvtx,mpi \
  --cuda-memory-usage=true \
  "${TORCHRUN_CMD[@]}"

echo "[capture] Capturing Nsight Compute trace..."
ncu --set full \
  --target-processes all \
  --force-overwrite \
  --kernel-name-base demangled \
  --launch-skip 10 \
  --launch-count 2 \
  --section Sleep,LaunchStats,MemoryWorkloadAnalysis,RooflineChart \
  --export "${NCU_DIR}/flexattention_fp8" \
  "${TORCHRUN_CMD[@]}" \
  --skip-power-monitor

echo "[capture] Done. Inspect:"
echo "  Nsight Systems: ${NSYS_DIR}/flexattention_fp8.qdrep"
echo "  Nsight Compute: ${NCU_DIR}/flexattention_fp8.ncu-rep"
