#!/bin/bash
# profile_chapter.sh - Profile all examples in a specific chapter
# Usage: ./profile_chapter.sh <chapter_number> [output_dir]

set -e

CHAPTER_NUM=$1
OUTPUT_DIR=${2:-"./profiling_results"}

if [ -z "$CHAPTER_NUM" ]; then
    echo "Usage: $0 <chapter_number> [output_dir]"
    echo "Example: $0 7 ./ch7_profiles"
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
echo "Profiling Chapter $CHAPTER_NUM"
echo "========================================="
echo "Chapter directory: $CHAPTER_DIR"
echo "Output directory: $OUTPUT_DIR/$CHAPTER_DIR"
echo ""

cd "$CHAPTER_DIR"

# Detect file types and profile accordingly
echo "Detecting examples..."
CU_FILES=$(find . -maxdepth 1 -name "*.cu" -type f 2>/dev/null || true)
PY_FILES=$(find . -maxdepth 1 -name "*.py" -type f ! -name "__*" ! -name "setup.py" -type f 2>/dev/null || true)

PROFILE_COUNT=0

# Profile CUDA examples
for cu_file in $CU_FILES; do
    filename=$(basename "$cu_file" .cu)
    
    # Skip if executable doesn't exist
    if [ ! -f "$filename" ]; then
        echo "⚠️  Skipping $filename (not built)"
        continue
    fi
    
    echo "🔍 Profiling CUDA example: $filename"
    
    # nsys profile
    if command -v nsys &> /dev/null; then
        echo "  Running nsys..."
        nsys profile \
            -o "../$OUTPUT_DIR/$CHAPTER_DIR/${filename}_nsys" \
            --stats=true \
            --force-overwrite=true \
            ./"$filename" > "../$OUTPUT_DIR/$CHAPTER_DIR/${filename}_nsys.log" 2>&1 || {
                echo "  ❌ nsys profiling failed (see log)"
            }
    fi
    
    # ncu profile (basic metrics only to avoid long runtime)
    if command -v ncu &> /dev/null; then
        echo "  Running ncu (basic metrics)..."
        ncu \
            -o "../$OUTPUT_DIR/$CHAPTER_DIR/${filename}_ncu" \
            --set basic \
            --force-overwrite \
            ./"$filename" > "../$OUTPUT_DIR/$CHAPTER_DIR/${filename}_ncu.log" 2>&1 || {
                echo "  ❌ ncu profiling failed (see log)"
            }
    fi
    
    PROFILE_COUNT=$((PROFILE_COUNT + 1))
    echo "  ✅ Profiled $filename"
    echo ""
done

# Profile Python examples
for py_file in $PY_FILES; do
    filename=$(basename "$py_file" .py)
    
    echo "🔍 Profiling Python example: $filename"
    
    # PyTorch profiler
    PROFILE_OUTPUT_DIR="../$OUTPUT_DIR/$CHAPTER_DIR/${filename}_pytorch_profile"
    mkdir -p "$PROFILE_OUTPUT_DIR"
    
    # Run with profiling enabled (if script supports --profile-output-dir flag)
    python3 "$py_file" --profile-output-dir "$PROFILE_OUTPUT_DIR" > "../$OUTPUT_DIR/$CHAPTER_DIR/${filename}_output.log" 2>&1 || {
        # If profiling flag not supported, just run normally
        python3 "$py_file" > "../$OUTPUT_DIR/$CHAPTER_DIR/${filename}_output.log" 2>&1 || {
            echo "  ❌ Execution failed (see log)"
            continue
        }
    }
    
    # Also run nsys on Python if available
    if command -v nsys &> /dev/null; then
        echo "  Running nsys on Python script..."
        nsys profile \
            -o "../$OUTPUT_DIR/$CHAPTER_DIR/${filename}_nsys" \
            --stats=true \
            --force-overwrite=true \
            python3 "$py_file" > "../$OUTPUT_DIR/$CHAPTER_DIR/${filename}_nsys.log" 2>&1 || {
                echo "  ❌ nsys profiling failed (see log)"
            }
    fi
    
    PROFILE_COUNT=$((PROFILE_COUNT + 1))
    echo "  ✅ Profiled $filename"
    echo ""
done

cd ..

echo "========================================="
echo "✅ Chapter $CHAPTER_NUM profiling complete!"
echo "Profiled $PROFILE_COUNT examples"
echo "Results in: $OUTPUT_DIR/$CHAPTER_DIR"
echo ""
echo "View nsys reports with: nsys-ui $OUTPUT_DIR/$CHAPTER_DIR/*_nsys.nsys-rep"
echo "View ncu reports with: ncu-ui $OUTPUT_DIR/$CHAPTER_DIR/*_ncu.ncu-rep"
echo "View PyTorch profiles with: tensorboard --logdir $OUTPUT_DIR/$CHAPTER_DIR"
echo "========================================="

