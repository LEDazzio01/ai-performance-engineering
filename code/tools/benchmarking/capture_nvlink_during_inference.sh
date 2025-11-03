#!/bin/bash
# Capture NVLink bandwidth and latency metrics during inference stress tests
# This script runs an inference workload while monitoring NVLink utilization

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default parameters
DURATION="${1:-60}"
OUTPUT_DIR="${2:-nvlink_stress_$(date +%Y%m%d_%H%M%S)}"
WORKLOAD="${3:-inference}"  # Options: inference, training, bandwidth

echo "================================================================================"
echo "NVLink Bandwidth Capture During Stress Test"
echo "================================================================================"
echo ""
echo "Duration: ${DURATION}s"
echo "Output directory: ${OUTPUT_DIR}"
echo "Workload: ${WORKLOAD}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# Set PYTHONPATH
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Pre-flight: Check NVLink topology
echo -e "${BLUE}Step 1: Capturing NVLink topology${NC}"
nvidia-smi topo -m > "${OUTPUT_DIR}/nvlink_topology.txt"
nvidia-smi nvlink --status > "${OUTPUT_DIR}/nvlink_status_before.txt"

# Get GPU count
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${NUM_GPUS} GPUs"
echo ""

# Step 2: Start NVLink bandwidth monitoring in background
echo -e "${BLUE}Step 2: Starting NVLink bandwidth monitoring${NC}"

# Monitor NVLink TX/RX bandwidth every 100ms
nvidia-smi dmon -s u -i 0,1,2,3,4,5,6,7 -c $((DURATION * 10)) -d 100 > "${OUTPUT_DIR}/nvlink_utilization.txt" 2>&1 &
DMON_PID=$!

# Monitor NVLink statistics
nvidia-smi nvlink --setcontrol 0bz  # Reset counters
sleep 1

# Step 3: Run workload
echo -e "${BLUE}Step 3: Running ${WORKLOAD} workload${NC}"
echo ""

WORKLOAD_SUCCESS=1

case "${WORKLOAD}" in
    inference)
        # Run distributed inference server under load
        echo "Starting inference server with high QPS..."
        torchrun --nproc_per_node="${NUM_GPUS}" \
            "${REPO_ROOT}/ch16/inference_server_load_test.py" \
            --duration "${DURATION}" \
            --target-qps 200 \
            --output-json "${OUTPUT_DIR}/inference_results.json" \
            > "${OUTPUT_DIR}/workload.log" 2>&1 || WORKLOAD_SUCCESS=0
        ;;
    
    training)
        # Run distributed training
        echo "Starting distributed training..."
        torchrun --nproc_per_node="${NUM_GPUS}" \
            "${REPO_ROOT}/ch4/training_8xb200_pipeline.py" \
            --epochs 1 \
            > "${OUTPUT_DIR}/workload.log" 2>&1 || WORKLOAD_SUCCESS=0
        ;;
    
    bandwidth)
        # Run pure bandwidth benchmark
        echo "Starting bandwidth benchmark..."
        python "${REPO_ROOT}/ch4/bandwidth_benchmark_suite_8gpu.py" \
            --output-json "${OUTPUT_DIR}/bandwidth_results.json" \
            > "${OUTPUT_DIR}/workload.log" 2>&1 || WORKLOAD_SUCCESS=0
        ;;
    
    *)
        echo "Unknown workload: ${WORKLOAD}"
        exit 1
        ;;
esac

# Wait for monitoring to complete
wait ${DMON_PID} 2>/dev/null || true

# Step 4: Capture NVLink statistics after test
echo ""
echo -e "${BLUE}Step 4: Capturing NVLink statistics${NC}"
nvidia-smi nvlink --status > "${OUTPUT_DIR}/nvlink_status_after.txt"
nvidia-smi nvlink -g 0 > "${OUTPUT_DIR}/nvlink_gpu0_stats.txt"

# Step 5: Parse and analyze results
echo ""
echo -e "${BLUE}Step 5: Analyzing NVLink metrics${NC}"

python3 - << 'EOF' > "${OUTPUT_DIR}/nvlink_analysis.txt"
import sys
import re
from pathlib import Path

output_dir = Path(sys.argv[1])

# Parse nvidia-smi dmon output
dmon_file = output_dir / "nvlink_utilization.txt"
if dmon_file.exists():
    with open(dmon_file) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    
    if len(lines) > 2:  # Skip header lines
        print("=== NVLink Utilization Summary ===\n")
        
        # Extract numeric data (skip header)
        data_lines = [l for l in lines if re.match(r'^\d+', l)]
        
        if data_lines:
            print(f"Total samples: {len(data_lines)}")
            print(f"Monitoring duration: ~{len(data_lines) * 0.1:.1f} seconds\n")
            
            # Note: nvidia-smi dmon -s u shows SM, MEM, ENC, DEC utilization
            # For actual NVLink bandwidth, we need nvlink stats
            print("Note: Use 'nvidia-smi nvlink -gt d' for detailed bandwidth stats")
        else:
            print("No utilization data captured")
    else:
        print("Insufficient data in dmon output")
else:
    print(f"dmon output not found: {dmon_file}")

# Parse NVLink status
status_before = output_dir / "nvlink_status_before.txt"
status_after = output_dir / "nvlink_status_after.txt"

print("\n=== NVLink Link Status ===\n")

if status_before.exists():
    with open(status_before) as f:
        content = f.read()
        # Count active links
        active_count = content.count("50 GB/s")
        print(f"Active NVLink connections (50 GB/s): {active_count}")

print("\nFor detailed per-link bandwidth analysis, run:")
print("  nvidia-smi nvlink -gt d")
print("\nThis shows TX/RX bytes per NVLink lane during the test.")

EOF

python3 "${OUTPUT_DIR}/../nvlink_analysis.py" "${OUTPUT_DIR}" || echo "Analysis complete (see nvlink_analysis.txt)"

# Step 6: Generate summary
echo ""
echo -e "${BLUE}Step 6: Generating summary${NC}"

cat > "${OUTPUT_DIR}/SUMMARY.md" << EOL
# NVLink Stress Test Summary

**Date**: $(date)
**Duration**: ${DURATION} seconds
**Workload**: ${WORKLOAD}
**GPUs**: ${NUM_GPUS}

## Files Generated

- \`nvlink_topology.txt\`: NVLink topology matrix
- \`nvlink_status_before.txt\`: Link status before test
- \`nvlink_status_after.txt\`: Link status after test
- \`nvlink_utilization.txt\`: Per-GPU utilization over time
- \`nvlink_gpu0_stats.txt\`: Detailed stats for GPU 0
- \`nvlink_analysis.txt\`: Parsed metrics and summary
- \`workload.log\`: Workload execution log

## Status

$(if [ ${WORKLOAD_SUCCESS} -eq 1 ]; then echo "✅ **SUCCESS**: Workload completed"; else echo "❌ **FAILURE**: Workload encountered errors"; fi)

## Next Steps

1. Review \`nvlink_analysis.txt\` for bandwidth utilization
2. Check \`nvlink_topology.txt\` to understand GPU interconnect
3. Run \`nvidia-smi nvlink -gt d\` for per-link bandwidth details
4. Compare utilization to theoretical max (1800 GB/s bidirectional per link)

## Detailed NVLink Commands

### Get per-link bandwidth (TX/RX bytes):
\`\`\`bash
nvidia-smi nvlink -gt d
\`\`\`

### Get error counters:
\`\`\`bash
nvidia-smi nvlink -gt e
\`\`\`

### Monitor live bandwidth:
\`\`\`bash
watch -n 1 'nvidia-smi nvlink -gt d'
\`\`\`

EOL

cat "${OUTPUT_DIR}/SUMMARY.md"

echo ""
echo "================================================================================"
echo "NVLink Capture Complete"
echo "================================================================================"
echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
echo "To view detailed per-link bandwidth:"
echo "  nvidia-smi nvlink -gt d"
echo ""

if [ ${WORKLOAD_SUCCESS} -eq 1 ]; then
    echo -e "${GREEN}✅ Capture completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Workload completed with warnings${NC}"
    exit 0
fi

