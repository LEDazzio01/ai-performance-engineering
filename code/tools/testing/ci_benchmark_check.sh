#!/bin/bash
# CI-friendly benchmark regression checker
# Usage: ./ci_benchmark_check.sh [config.json]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

CONFIG="${1:-docs/examples/continuous_benchmark.json}"
ARTIFACT_DIR="${2:-benchmark_runs}"
TAG="${3:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"

echo "================================================================================"
echo "CI Benchmark Regression Check"
echo "================================================================================"
echo ""
echo "Config: $CONFIG"
echo "Artifact dir: $ARTIFACT_DIR"
echo "Tag: $TAG"
echo ""

# Step 1: Run benchmarks
echo "Step 1: Running benchmark suite..."
echo ""

if ! python3 tools/continuous_benchmark.py \
    "$CONFIG" \
    --artifact-dir "$ARTIFACT_DIR" \
    --tag "$TAG" \
    --stop-on-fail; then
    echo -e "${RED}Benchmark suite failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Benchmarks complete${NC}"
echo ""

# Step 2: Check for regressions
echo "Step 2: Checking for performance regressions..."
echo ""

REPORT_FILE="$ARTIFACT_DIR/regression_report_$(date +%Y%m%d_%H%M%S).md"

if python3 tools/detect_regressions.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --output "$REPORT_FILE" \
    --output-json "$ARTIFACT_DIR/regression_analysis.json" \
    --fail-on-regression; then
    echo ""
    echo -e "${GREEN}✓ No regressions detected${NC}"
    EXIT_CODE=0
else
    echo ""
    echo -e "${YELLOW}⚠ Regressions detected!${NC}"
    echo ""
    echo "Regression report: $REPORT_FILE"
    echo ""
    
    # Display report
    if [ -f "$REPORT_FILE" ]; then
        cat "$REPORT_FILE"
    fi
    
    EXIT_CODE=1
fi

echo ""
echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo ""
echo "Reports saved to: $ARTIFACT_DIR/"
echo "  - Latest benchmark run: $(ls -t $ARTIFACT_DIR/benchmark_run_*.json | head -n1)"
echo "  - Regression report: $REPORT_FILE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed${NC}"
else
    echo -e "${RED}❌ Performance regressions detected${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review the regression report above"
    echo "  2. Profile affected benchmarks with: tools/profile_all_workloads.sh"
    echo "  3. Check recent commits for performance impacts"
    echo "  4. If regression is expected, update baseline"
fi

echo ""

exit $EXIT_CODE


