#!/bin/bash
# Automated accuracy testing suite
# Tests model accuracy across different precision modes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

RESULTS_DIR="${1:-accuracy_test_results_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"

echo "================================================================================"
echo "Automated Accuracy Testing Suite"
echo "================================================================================"
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""

# Step 1: Create evaluation datasets
echo "Step 1: Creating evaluation datasets..."
python3 tools/create_eval_datasets.py \
    --output-dir "$RESULTS_DIR/eval_datasets" \
    --vocab-size 50000 \
    --num-tokens 20000

echo ""

# Step 2: Run precision comparisons on each dataset
echo "Step 2: Running precision comparisons..."

for dataset in "$RESULTS_DIR/eval_datasets"/*.txt; do
    if [ -f "$dataset" ]; then
        dataset_name=$(basename "$dataset" .txt)
        echo ""
        echo "  Testing dataset: $dataset_name"
        
        # Run comparison
        if python3 tools/compare_precision_accuracy.py \
            --dataset "$dataset" \
            --precisions fp16 bf16 \
            --seq-len 256 \
            --stride 128 \
            --output "$RESULTS_DIR/comparison_${dataset_name}.md" \
            --output-json "$RESULTS_DIR/comparison_${dataset_name}.json" 2>&1 | tee "$RESULTS_DIR/log_${dataset_name}.txt"; then
            echo -e "    ${GREEN}✓ Complete${NC}"
        else
            echo -e "    ${YELLOW}⚠ Completed with warnings${NC}"
        fi
    fi
done

echo ""
echo "Step 3: Generating summary report..."

# Create summary report
SUMMARY_FILE="$RESULTS_DIR/ACCURACY_SUMMARY.md"

cat > "$SUMMARY_FILE" << 'EOF'
# Accuracy Testing Summary

## Overview

This report summarizes accuracy testing across multiple evaluation datasets
and precision modes (FP16, BF16, FP8).

## Datasets Tested

EOF

# List datasets
for dataset in "$RESULTS_DIR/eval_datasets"/*.txt; do
    if [ -f "$dataset" ]; then
        dataset_name=$(basename "$dataset" .txt)
        token_count=$(wc -w < "$dataset")
        echo "- **${dataset_name}**: ${token_count} tokens" >> "$SUMMARY_FILE"
    fi
done

cat >> "$SUMMARY_FILE" << 'EOF'

## Precision Comparison Results

EOF

# Add links to individual reports
for report in "$RESULTS_DIR"/comparison_*.md; do
    if [ -f "$report" ]; then
        report_name=$(basename "$report" .md)
        dataset_name=${report_name#comparison_}
        echo "### ${dataset_name}" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        
        # Extract key metrics from JSON if available
        json_file="${report%.md}.json"
        if [ -f "$json_file" ]; then
            # Use Python to extract key metrics
            python3 -c "
import json
import sys
try:
    with open('$json_file') as f:
        data = json.load(f)
    if 'fp16' in data:
        print('- **FP16 Perplexity**: {:.3f}'.format(data['fp16'].get('perplexity', 0)))
    if 'bf16' in data:
        print('- **BF16 Perplexity**: {:.3f}'.format(data['bf16'].get('perplexity', 0)))
    if 'fp8' in data:
        print('- **FP8 Perplexity**: {:.3f}'.format(data['fp8'].get('perplexity', 0)))
except Exception as e:
    print(f'Error reading metrics: {e}', file=sys.stderr)
" >> "$SUMMARY_FILE" 2>/dev/null || echo "- Metrics extraction failed" >> "$SUMMARY_FILE"
        fi
        
        echo "" >> "$SUMMARY_FILE"
        echo "See full report: [\`${report_name}.md\`](./${report_name}.md)" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    fi
done

cat >> "$SUMMARY_FILE" << 'EOF'

## Recommendations

Based on the accuracy tests:

1. **FP16**: Recommended baseline for most inference workloads
2. **BF16**: Better dynamic range, good for training and inference
3. **FP8**: Use when:
   - Accuracy delta is < 5% on your workload
   - Performance gains justify the trade-off
   - You have validation metrics in production

## Next Steps

1. Review individual comparison reports for detailed analysis
2. Validate FP8 accuracy on your specific production workload
3. Set up monitoring for accuracy drift in production
4. Consider A/B testing before full FP8 rollout

EOF

echo -e "${GREEN}✓ Summary report created: $SUMMARY_FILE${NC}"
echo ""

# Display summary
cat "$SUMMARY_FILE"

echo ""
echo "================================================================================"
echo "Accuracy Testing Complete"
echo "================================================================================"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Key files:"
echo "  - ACCURACY_SUMMARY.md: Overall summary"
echo "  - comparison_*.md: Per-dataset comparisons"
echo "  - comparison_*.json: Raw metrics"
echo "  - eval_datasets/: Generated test datasets"
echo ""


