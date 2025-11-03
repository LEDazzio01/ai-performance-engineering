#!/usr/bin/env bash
# Run dual-architecture builds across CUDA chapters.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHAPTERS=(
  ch1
  ch2
  ch4
  ch6
  ch7
  ch8
  ch9
  ch10
  ch11
  ch12
)

echo "=== Dual-architecture compare builds ==="
for chapter in "${CHAPTERS[@]}"; do
  echo ""
  echo ">>> ${chapter}: make compare"
  (cd "${REPO_ROOT}/${chapter}" && make compare)
done

echo ""
echo "All dual-architecture builds completed successfully."
