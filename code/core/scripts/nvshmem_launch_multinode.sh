#!/usr/bin/env bash
# Launch helper for NVSHMEM multi-node demos.
#
# Usage:
#   ./core/scripts/nvshmem_launch_multinode.sh <nodes> <gpus_per_node> [binary] [args...]
#
# Example:
#   ./core/scripts/nvshmem_launch_multinode.sh 2 8 ./ch04/nvshmem_multinode_example --gpus-per-node 8
#
# Requires NVSHMEM to be installed and nvshmemrun in PATH. Host selection is
# controlled via NVSHMEM_HOSTFILE (defaults to localhost for single-node tests).

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <nodes> <gpus_per_node> [binary] [args...]" >&2
  exit 1
fi

NODES=$1
GPUS_PER_NODE=$2
shift 2
BINARY=${1:-./ch04/nvshmem_multinode_example}
shift || true
EXTRA_ARGS=("$@")

WORLD_SIZE=$((NODES * GPUS_PER_NODE))
HOSTFILE=${NVSHMEM_HOSTFILE:-}

echo "Launching NVSHMEM job"
echo "  Nodes:           ${NODES}"
echo "  GPUs per node:   ${GPUS_PER_NODE}"
echo "  World size:      ${WORLD_SIZE}"
echo "  Binary:          ${BINARY}"
if [[ -n "$HOSTFILE" ]]; then
  echo "  Hostfile:        ${HOSTFILE}"
fi

declare -a CMD
CMD=(nvshmemrun -np "${WORLD_SIZE}" --ppn "${GPUS_PER_NODE}")
if [[ -n "$HOSTFILE" ]]; then
  CMD+=(--hostfile "$HOSTFILE")
fi
CMD+=("${BINARY}" "${EXTRA_ARGS[@]}")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
