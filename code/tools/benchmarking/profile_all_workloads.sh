#!/bin/bash
# Convenience wrapper for automated Nsight Systems profiling

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

OUTPUT_DIR="${1:-nsys_profiles_$(date +%Y%m%d_%H%M%S)}"

echo "================================================================================"
echo "Automated Nsight Systems Profiling"
echo "================================================================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo -e "${RED}Error: nsys not found${NC}"
    echo ""
    echo "Nsight Systems is required for profiling."
    echo "Download from: https://developer.nvidia.com/nsight-systems"
    echo ""
    echo "On Ubuntu/Debian:"
    echo "  wget https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/nsight-systems-2024.x.x_xxx_amd64.deb"
    echo "  sudo dpkg -i nsight-systems-*.deb"
    echo ""
    exit 1
fi

echo "✓ Nsight Systems found: $(nsys --version | head -n1)"
echo ""

# Run profiling
python3 tools/nsys_profile_workload.py \
    --output-dir "$OUTPUT_DIR" \
    --workloads all \
    --kernel-regex "attn|mma|nvjet|cublas|gemm|fp8"

EXIT_CODE=$?

echo ""
echo "================================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Profiling Complete!${NC}"
else
    echo -e "${YELLOW}Profiling completed with some failures${NC}"
fi

echo "================================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "To view profiles:"
echo "  nsys-ui $OUTPUT_DIR/<workload>.nsys-rep"
echo ""
echo "To see summaries:"
echo "  cat $OUTPUT_DIR/*_summary.txt"
echo ""
echo "To get detailed stats:"
echo "  nsys stats $OUTPUT_DIR/<workload>.nsys-rep"
echo ""

exit $EXIT_CODE


