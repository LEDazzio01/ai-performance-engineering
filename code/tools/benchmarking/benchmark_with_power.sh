#!/bin/bash
# Run benchmarks with power monitoring and cost analysis

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default parameters
WORKLOAD="${1:-ch16/test_gpt_large_optimized.py}"
WORKLOAD_ARGS="${2:---batch-size 1 --seq-len 2048}"
OUTPUT_DIR="${3:-power_benchmark_$(date +%Y%m%d_%H%M%S)}"

echo "================================================================================"
echo "Power-Monitored Benchmark with Cost Analysis"
echo "================================================================================"
echo ""
echo "Workload: $WORKLOAD"
echo "Arguments: $WORKLOAD_ARGS"
echo "Output directory: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Step 1: Check if pynvml is available
if ! python3 -c "import pynvml" 2>/dev/null; then
    echo -e "${YELLOW}Warning: pynvml not installed${NC}"
    echo "Installing nvidia-ml-py..."
    pip install nvidia-ml-py3 || {
        echo "Failed to install pynvml. Power monitoring will be skipped."
        # Run without power monitoring
        python3 "$WORKLOAD" $WORKLOAD_ARGS
        exit $?
    }
fi

echo "Step 1: Running workload with power monitoring..."

# Run the workload with power monitoring
POWER_JSON="$OUTPUT_DIR/power_metrics.json"
BENCHMARK_OUTPUT="$OUTPUT_DIR/benchmark_output.txt"

python3 tools/power_monitor.py \
    --output-json "$POWER_JSON" \
    --interval 0.1 \
    -- python3 "$WORKLOAD" $WORKLOAD_ARGS | tee "$BENCHMARK_OUTPUT"

if [ ! -f "$POWER_JSON" ]; then
    echo -e "${YELLOW}Power metrics not generated${NC}"
    exit 1
fi

echo ""
echo "✓ Power monitoring complete"
echo ""

# Step 2: Extract throughput from benchmark output
echo "Step 2: Extracting throughput metrics..."

# Try to find tokens/sec in the output
THROUGHPUT=$(grep -oP '[\d.]+\s+tokens?/s(?:ec)?' "$BENCHMARK_OUTPUT" | head -n1 | awk '{print $1}' || echo "")

if [ -z "$THROUGHPUT" ]; then
    # Try alternative patterns
    THROUGHPUT=$(grep -oP 'throughput.*?[\d.]+' "$BENCHMARK_OUTPUT" | grep -oP '[\d.]+' | head -n1 || echo "")
fi

if [ -z "$THROUGHPUT" ]; then
    echo -e "${YELLOW}Could not auto-detect throughput${NC}"
    echo "Please enter throughput (tokens/sec):"
    read THROUGHPUT
fi

echo "Detected throughput: $THROUGHPUT tokens/sec"

# Step 3: Calculate cost per token
echo ""
echo "Step 3: Calculating cost per token..."

WORKLOAD_NAME=$(basename "$WORKLOAD" .py)
COST_REPORT="$OUTPUT_DIR/cost_analysis.md"
COST_JSON="$OUTPUT_DIR/cost_metrics.json"

python3 tools/calculate_cost_per_token.py \
    --power-json "$POWER_JSON" \
    --throughput "$THROUGHPUT" \
    --workload-name "$WORKLOAD_NAME" \
    --output "$COST_REPORT" \
    --output-json "$COST_JSON"

echo ""
echo "✓ Cost analysis complete"
echo ""

# Step 4: Display summary
echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo ""

# Extract key metrics
AVG_POWER=$(python3 -c "import json; print(json.load(open('$POWER_JSON'))['total_power']['avg_watts'])")
ENERGY_KJ=$(python3 -c "import json; print(json.load(open('$POWER_JSON'))['energy_joules'] / 1000)")
COST_PER_MILLION=$(python3 -c "import json; print(json.load(open('$COST_JSON'))['cost_per_million_tokens_usd'])")
TOKENS_PER_DOLLAR=$(python3 -c "import json; print(int(json.load(open('$COST_JSON'))['tokens_per_dollar']))")

echo "Power Metrics:"
echo "  Average Power: ${AVG_POWER} W"
echo "  Total Energy: ${ENERGY_KJ} kJ"
echo ""
echo "Performance:"
echo "  Throughput: ${THROUGHPUT} tokens/sec"
echo ""
echo "Cost Metrics:"
echo "  Cost per 1M tokens: \$${COST_PER_MILLION}"
echo "  Tokens per dollar: ${TOKENS_PER_DOLLAR}"
echo ""
echo "Detailed reports saved to:"
echo "  - Power metrics: $POWER_JSON"
echo "  - Cost analysis: $COST_REPORT"
echo "  - Cost metrics: $COST_JSON"
echo ""

# Display cost report
if [ -f "$COST_REPORT" ]; then
    echo "================================================================================"
    echo "Cost Analysis Report"
    echo "================================================================================"
    cat "$COST_REPORT"
fi

echo ""
echo -e "${GREEN}Analysis complete!${NC}"
echo ""


