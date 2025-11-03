#!/bin/bash
# benchmark_chapter.sh - Benchmark all examples in a specific chapter
# Usage: ./benchmark_chapter.sh <chapter_number> [output_dir]

set -e

CHAPTER_NUM=$1
OUTPUT_DIR=${2:-"./benchmark_results"}
ITERATIONS=${3:-100}  # Number of iterations for averaging

if [ -z "$CHAPTER_NUM" ]; then
    echo "Usage: $0 <chapter_number> [output_dir] [iterations]"
    echo "Example: $0 7 ./ch7_benchmarks 100"
    exit 1
fi

CHAPTER_DIR="ch${CHAPTER_NUM}"

if [ ! -d "$CHAPTER_DIR" ]; then
    echo "Error: Chapter directory '$CHAPTER_DIR' not found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR/$CHAPTER_DIR"

echo "========================================="
echo "Benchmarking Chapter $CHAPTER_NUM"
echo "========================================="
echo "Chapter directory: $CHAPTER_DIR"
echo "Output directory: $OUTPUT_DIR/$CHAPTER_DIR"
echo "Iterations: $ITERATIONS"
echo ""

cd "$CHAPTER_DIR"

# Create benchmark results JSON
RESULTS_FILE="../$OUTPUT_DIR/$CHAPTER_DIR/benchmark_results.json"
echo "{" > "$RESULTS_FILE"
echo "  \"chapter\": $CHAPTER_NUM," >> "$RESULTS_FILE"
echo "  \"timestamp\": \"$(date -Iseconds)\"," >> "$RESULTS_FILE"
echo "  \"iterations\": $ITERATIONS," >> "$RESULTS_FILE"
echo "  \"benchmarks\": {" >> "$RESULTS_FILE"

FIRST_ENTRY=true

# Detect and benchmark CUDA examples
CU_FILES=$(find . -maxdepth 1 -name "*.cu" -type f 2>/dev/null || true)
for cu_file in $CU_FILES; do
    filename=$(basename "$cu_file" .cu)
    
    # Skip if executable doesn't exist
    if [ ! -f "$filename" ]; then
        echo "⚠️  Skipping $filename (not built)"
        continue
    fi
    
    echo "⏱️  Benchmarking CUDA example: $filename"
    
    # Run multiple iterations and collect timing
    TIMES=()
    SUCCESS_COUNT=0
    
    for i in $(seq 1 $ITERATIONS); do
        # Run with timing
        START=$(date +%s%N)
        if ./"$filename" > /dev/null 2>&1; then
            END=$(date +%s%N)
            TIME_MS=$(echo "scale=3; ($END - $START) / 1000000" | bc)
            TIMES+=($TIME_MS)
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
        
        # Progress indicator
        if [ $((i % 10)) -eq 0 ]; then
            echo -n "."
        fi
    done
    echo ""
    
    if [ $SUCCESS_COUNT -gt 0 ]; then
        # Calculate statistics
        MEAN=$(printf '%s\n' "${TIMES[@]}" | awk '{sum+=$1} END {print sum/NR}')
        MIN=$(printf '%s\n' "${TIMES[@]}" | sort -n | head -1)
        MAX=$(printf '%s\n' "${TIMES[@]}" | sort -n | tail -1)
        
        # Add to JSON
        if [ "$FIRST_ENTRY" = false ]; then
            echo "," >> "$RESULTS_FILE"
        fi
        FIRST_ENTRY=false
        
        cat >> "$RESULTS_FILE" <<EOF
    "$filename": {
      "type": "cuda",
      "mean_ms": $MEAN,
      "min_ms": $MIN,
      "max_ms": $MAX,
      "iterations": $SUCCESS_COUNT,
      "success_rate": $(echo "scale=3; $SUCCESS_COUNT / $ITERATIONS" | bc)
    }
EOF
        
        echo "  ✅ $filename: ${MEAN}ms (±$(echo "scale=2; $MAX - $MIN" | bc)ms)"
    else
        echo "  ❌ $filename: All runs failed"
    fi
    echo ""
done

# Benchmark Python examples
PY_FILES=$(find . -maxdepth 1 -name "*.py" -type f ! -name "__*" ! -name "setup.py" -type f 2>/dev/null || true)
for py_file in $PY_FILES; do
    filename=$(basename "$py_file" .py)
    
    echo "⏱️  Benchmarking Python example: $filename"
    
    # Run multiple iterations
    TIMES=()
    SUCCESS_COUNT=0
    
    for i in $(seq 1 $ITERATIONS); do
        START=$(date +%s%N)
        if python3 "$py_file" > /dev/null 2>&1; then
            END=$(date +%s%N)
            TIME_MS=$(echo "scale=3; ($END - $START) / 1000000" | bc)
            TIMES+=($TIME_MS)
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
        
        if [ $((i % 10)) -eq 0 ]; then
            echo -n "."
        fi
    done
    echo ""
    
    if [ $SUCCESS_COUNT -gt 0 ]; then
        MEAN=$(printf '%s\n' "${TIMES[@]}" | awk '{sum+=$1} END {print sum/NR}')
        MIN=$(printf '%s\n' "${TIMES[@]}" | sort -n | head -1)
        MAX=$(printf '%s\n' "${TIMES[@]}" | sort -n | tail -1)
        
        if [ "$FIRST_ENTRY" = false ]; then
            echo "," >> "$RESULTS_FILE"
        fi
        FIRST_ENTRY=false
        
        cat >> "$RESULTS_FILE" <<EOF
    "$filename": {
      "type": "python",
      "mean_ms": $MEAN,
      "min_ms": $MIN,
      "max_ms": $MAX,
      "iterations": $SUCCESS_COUNT,
      "success_rate": $(echo "scale=3; $SUCCESS_COUNT / $ITERATIONS" | bc)
    }
EOF
        
        echo "  ✅ $filename: ${MEAN}ms (±$(echo "scale=2; $MAX - $MIN" | bc)ms)"
    else
        echo "  ❌ $filename: All runs failed"
    fi
    echo ""
done

# Close JSON
echo "" >> "$RESULTS_FILE"
echo "  }" >> "$RESULTS_FILE"
echo "}" >> "$RESULTS_FILE"

cd ..

echo "========================================="
echo "✅ Chapter $CHAPTER_NUM benchmarking complete!"
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "View results: cat $RESULTS_FILE"
echo "Compare results: python3 common/profiling/compare_results.py <baseline.json> <optimized.json>"
echo "========================================="

