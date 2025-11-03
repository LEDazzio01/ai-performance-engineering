#!/bin/bash
# run_all_chapters.sh - Run profiling and/or benchmarking across all chapters
# Usage:
#   ./run_all_chapters.sh [output_dir] [options]
#
# Options:
#   --profile             Run the profiling pass (default: enabled unless --benchmark only).
#   --benchmark           Run the benchmarking pass (default: enabled unless --profile only).
#   --sm <value>          Target SM version (e.g. 121 or sm_121). Auto-detected when possible.
#   --iterations <count>  Override benchmark iterations (default: 50).
#   --skip <list>         Comma-separated list of chapter numbers to skip.
#   --only <list>         Comma-separated list of chapter numbers to run (others skipped).
#   --no-build            Do not run make before profiling/benchmarking.
#   --help                Show this message.
#
# Positional compatibility:
#   legacy mode arguments (profile|benchmark|both) are still accepted but deprecated.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./run_all_chapters.sh [output_dir] [options]

Options:
  --profile             Run only the profiling pass (combine with --benchmark to run both).
  --benchmark           Run only the benchmarking pass (default: runs both when neither flag set).
  --sm <value>          Target SM value (e.g. 121 or sm_121). Auto-detected when possible.
  --iterations <count>  Benchmark iterations (default: 50).
  --skip <list>         Comma-separated chapter numbers to skip.
  --only <list>         Comma-separated chapter numbers to include (others skipped).
  --no-build            Skip invoking make in chapter directories.
  --help                Show this help and exit.

Positional compatibility (deprecated):
  profile|benchmark|both   Legacy mode argument (prefer --profile/--benchmark).
  <output_dir>             Destination directory for results (default: ./all_chapters_results).
EOF
}

OUTPUT_DIR="./all_chapters_results"
OUTPUT_DIR_SET=false
DO_PROFILE=false
DO_BENCHMARK=false
SM_TARGET=""
BENCHMARK_ITERATIONS=50
RUN_BUILD=true
declare -a SKIP_CHAPTERS=()
declare -a ONLY_CHAPTERS=()
LEGACY_MODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        profile|benchmark|both)
            LEGACY_MODE="$1"
            case "$1" in
                profile)
                    DO_PROFILE=true
                    DO_BENCHMARK=false
                    ;;
                benchmark)
                    DO_PROFILE=false
                    DO_BENCHMARK=true
                    ;;
                both)
                    DO_PROFILE=true
                    DO_BENCHMARK=true
                    ;;
            esac
            shift
            ;;
        --profile)
            DO_PROFILE=true
            shift
            ;;
        --benchmark)
            DO_BENCHMARK=true
            shift
            ;;
        --sm)
            [[ $# -ge 2 ]] || { echo "Error: --sm requires a value" >&2; exit 1; }
            SM_TARGET="$2"
            shift 2
            ;;
        --iterations)
            [[ $# -ge 2 ]] || { echo "Error: --iterations requires a value" >&2; exit 1; }
            BENCHMARK_ITERATIONS="$2"
            shift 2
            ;;
        --skip)
            [[ $# -ge 2 ]] || { echo "Error: --skip requires a value" >&2; exit 1; }
            IFS=',' read -ra SKIP_CHAPTERS <<< "$2"
            shift 2
            ;;
        --only)
            [[ $# -ge 2 ]] || { echo "Error: --only requires a value" >&2; exit 1; }
            IFS=',' read -ra ONLY_CHAPTERS <<< "$2"
            shift 2
            ;;
        --no-build)
            RUN_BUILD=false
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            if [[ "$OUTPUT_DIR_SET" == false ]]; then
                OUTPUT_DIR="$1"
                OUTPUT_DIR_SET=true
                shift
            else
                echo "Unexpected positional argument: $1" >&2
                usage
                exit 1
            fi
            ;;
    esac
done

if [[ -n "$LEGACY_MODE" ]]; then
    echo "Warning: legacy positional mode '$LEGACY_MODE' is deprecated. Use --profile/--benchmark flags instead." >&2
fi

if [[ "$DO_PROFILE" == false && "$DO_BENCHMARK" == false ]]; then
    DO_PROFILE=true
    DO_BENCHMARK=true
fi

detect_sm_target() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr '[:upper:]' '[:lower:]')
        case "$gpu_name" in
            *gb200*|*gb10*)
                echo "121"
                return
                ;;
            *b200*)
                echo "100"
                return
                ;;
        esac
    fi
    echo ""
}

if [[ -z "$SM_TARGET" ]]; then
    SM_TARGET="$(detect_sm_target)"
fi

if [[ -z "$SM_TARGET" ]]; then
    SM_TARGET="121"
fi

if [[ "$SM_TARGET" == sm_* ]]; then
    CUDA_ARCH_VALUE="$SM_TARGET"
    SM_NUM="${SM_TARGET#sm_}"
else
    CUDA_ARCH_VALUE="sm_${SM_TARGET}"
    SM_NUM="$SM_TARGET"
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$SM_NUM}"
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-$SM_NUM}"

REQUIRED_SCRIPTS=()
if [[ "$DO_PROFILE" == true ]]; then
    REQUIRED_SCRIPTS+=("./profile_chapter.sh")
fi
if [[ "$DO_BENCHMARK" == true ]]; then
    REQUIRED_SCRIPTS+=("./benchmark_chapter.sh")
fi

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [[ ! -x "$script" ]]; then
        echo "Error: Required script '$script' not found or not executable." >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
[[ "$DO_PROFILE" == true ]] && mkdir -p "$OUTPUT_DIR/profiling"
[[ "$DO_BENCHMARK" == true ]] && mkdir -p "$OUTPUT_DIR/benchmarking"

CHAPTERS=($(find . -maxdepth 1 -type d -name "ch[0-9]*" | sort -V | sed 's#./ch##'))

if [[ ${#ONLY_CHAPTERS[@]} -gt 0 ]]; then
    filtered=()
    for chapter in "${CHAPTERS[@]}"; do
        for only in "${ONLY_CHAPTERS[@]}"; do
            if [[ "$chapter" == "$only" ]]; then
                filtered+=("$chapter")
                break
            fi
        done
    done
    CHAPTERS=("${filtered[@]}")
fi

if [[ ${#CHAPTERS[@]} -eq 0 ]]; then
    echo "No chapters to process."
    exit 0
fi

echo "========================================="
echo "Running all chapters"
if [[ "$DO_PROFILE" == true && "$DO_BENCHMARK" == true ]]; then
    echo "Tasks: profile + benchmark"
elif [[ "$DO_PROFILE" == true ]]; then
    echo "Tasks: profile"
else
    echo "Tasks: benchmark"
fi
echo "Output directory: $OUTPUT_DIR"
echo "Target SM: $CUDA_ARCH_VALUE (override with --sm)"
[[ "$DO_BENCHMARK" == true ]] && echo "Benchmark iterations: $BENCHMARK_ITERATIONS"
if [[ ${#SKIP_CHAPTERS[@]} -gt 0 ]]; then
    echo "Skipping chapters: ${SKIP_CHAPTERS[*]}"
fi
if [[ ${#ONLY_CHAPTERS[@]} -gt 0 ]]; then
    echo "Restricted to chapters: ${ONLY_CHAPTERS[*]}"
fi
echo "========================================="
echo ""

declare -a BUILD_SUCCESS=() BUILD_FAILED=()
declare -a PROFILE_SUCCESS=() PROFILE_FAILED=()
declare -a BENCHMARK_SUCCESS=() BENCHMARK_FAILED=()
declare -a PROCESSED_CHAPTERS=() SKIPPED_CHAPTERS=()

should_skip() {
    local chapter="$1"
    for skip in "${SKIP_CHAPTERS[@]}"; do
        if [[ "$chapter" == "$skip" ]]; then
            return 0
        fi
    done
    return 1
}

for chapter in "${CHAPTERS[@]}"; do
    if should_skip "$chapter"; then
        echo "Skipping Chapter $chapter (requested)"
        SKIPPED_CHAPTERS+=("$chapter")
        continue
    fi

    PROCESSED_CHAPTERS+=("$chapter")

    echo ""
    echo "========================================="
    echo "Processing Chapter $chapter"
    echo "========================================="

    CH_DIR="ch${chapter}"

    if [[ "$RUN_BUILD" == true && -f "$CH_DIR/Makefile" ]]; then
        BUILD_LOG="$OUTPUT_DIR/ch${chapter}_build.log"
        echo "Building chapter $chapter (CUDA_ARCH=$CUDA_ARCH_VALUE, C++20)..."
        if make -C "$CH_DIR" ARCH="$CUDA_ARCH_VALUE" CUDA_CXX_STANDARD=20 CXXFLAGS="-std=c++20 -O3" >"$BUILD_LOG" 2>&1; then
            BUILD_SUCCESS+=("$chapter")
            echo "✅ Build succeeded"
        else
            BUILD_FAILED+=("$chapter")
            echo "❌ Build failed (see $BUILD_LOG)"
            continue
        fi
    fi

    if [[ "$DO_PROFILE" == true ]]; then
        PROFILE_LOG="$OUTPUT_DIR/ch${chapter}_profile.log"
        echo "Profiling chapter $chapter..."
        if ./profile_chapter.sh "$chapter" "$OUTPUT_DIR/profiling" >"$PROFILE_LOG" 2>&1; then
            PROFILE_SUCCESS+=("$chapter")
            echo "✅ Chapter $chapter profiling succeeded"
        else
            PROFILE_FAILED+=("$chapter")
            echo "❌ Chapter $chapter profiling failed (see $PROFILE_LOG)"
        fi
    fi

    if [[ "$DO_BENCHMARK" == true ]]; then
        BENCHMARK_LOG="$OUTPUT_DIR/ch${chapter}_benchmark.log"
        echo "Benchmarking chapter $chapter..."
        if ./benchmark_chapter.sh "$chapter" "$OUTPUT_DIR/benchmarking" "$BENCHMARK_ITERATIONS" >"$BENCHMARK_LOG" 2>&1; then
            BENCHMARK_SUCCESS+=("$chapter")
            echo "✅ Chapter $chapter benchmarking succeeded"
        else
            BENCHMARK_FAILED+=("$chapter")
            echo "❌ Chapter $chapter benchmarking failed (see $BENCHMARK_LOG)"
        fi
    fi
done

echo ""
echo "========================================="
echo "Summary"
echo "========================================="
echo ""
echo "Chapters detected: ${#CHAPTERS[@]}"
echo "Chapters processed: ${#PROCESSED_CHAPTERS[@]}"
if [[ ${#SKIPPED_CHAPTERS[@]} -gt 0 ]]; then
    echo "Chapters skipped: ${#SKIPPED_CHAPTERS[@]} (${SKIPPED_CHAPTERS[*]})"
fi

if [[ "$RUN_BUILD" == true ]]; then
    echo ""
    echo "Builds:"
    if [[ ${#BUILD_SUCCESS[@]} -gt 0 ]]; then
        echo "  Success: ${#BUILD_SUCCESS[@]} chapters (${BUILD_SUCCESS[*]})"
    else
        echo "  Success: 0 chapters"
    fi
    if [[ ${#BUILD_FAILED[@]} -gt 0 ]]; then
        echo "  Failed:  ${#BUILD_FAILED[@]} chapters (${BUILD_FAILED[*]})"
    else
        echo "  Failed:  0 chapters"
    fi
fi

if [[ "$DO_PROFILE" == true ]]; then
    echo ""
    echo "Profiling:"
    if [[ ${#PROFILE_SUCCESS[@]} -gt 0 ]]; then
        echo "  Success: ${#PROFILE_SUCCESS[@]} chapters (${PROFILE_SUCCESS[*]})"
    else
        echo "  Success: 0 chapters"
    fi
    if [[ ${#PROFILE_FAILED[@]} -gt 0 ]]; then
        echo "  Failed:  ${#PROFILE_FAILED[@]} chapters (${PROFILE_FAILED[*]})"
    else
        echo "  Failed:  0 chapters"
    fi
fi

if [[ "$DO_BENCHMARK" == true ]]; then
    echo ""
    echo "Benchmarking:"
    if [[ ${#BENCHMARK_SUCCESS[@]} -gt 0 ]]; then
        echo "  Success: ${#BENCHMARK_SUCCESS[@]} chapters (${BENCHMARK_SUCCESS[*]})"
    else
        echo "  Success: 0 chapters"
    fi
    if [[ ${#BENCHMARK_FAILED[@]} -gt 0 ]]; then
        echo "  Failed:  ${#BENCHMARK_FAILED[@]} chapters (${BENCHMARK_FAILED[*]})"
    else
        echo "  Failed:  0 chapters"
    fi
fi

echo ""
echo "All results in: $OUTPUT_DIR"
echo ""

SUMMARY_FILE="$OUTPUT_DIR/summary.md"
cat > "$SUMMARY_FILE" <<EOF
# All Chapters Summary

**Generated**: $(date)

## Chapters Processed

Total chapters detected: ${#CHAPTERS[@]}
Chapters processed: ${#PROCESSED_CHAPTERS[@]}
Chapters skipped: ${#SKIPPED_CHAPTERS[@]}
EOF

if [[ "$RUN_BUILD" == true ]]; then
cat >> "$SUMMARY_FILE" <<EOF

### Build Results

- ✅ Success: ${#BUILD_SUCCESS[@]} chapters
- ❌ Failed: ${#BUILD_FAILED[@]} chapters
EOF
fi

if [[ "$DO_PROFILE" == true ]]; then
cat >> "$SUMMARY_FILE" <<EOF

### Profiling Results

- ✅ Success: ${#PROFILE_SUCCESS[@]} chapters
- ❌ Failed: ${#PROFILE_FAILED[@]} chapters
EOF
fi

if [[ "$DO_BENCHMARK" == true ]]; then
cat >> "$SUMMARY_FILE" <<EOF

### Benchmarking Results

- ✅ Success: ${#BENCHMARK_SUCCESS[@]} chapters
- ❌ Failed: ${#BENCHMARK_FAILED[@]} chapters
EOF
fi

cat >> "$SUMMARY_FILE" <<EOF

## Per-Chapter Details
EOF

for chapter in "${PROCESSED_CHAPTERS[@]}"; do
    BUILD_LINE="- Build: not requested"
    if [[ "$RUN_BUILD" == true ]]; then
        if [[ -f "$OUTPUT_DIR/ch${chapter}_build.log" ]]; then
            BUILD_LINE="- Build log: [\`ch${chapter}_build.log\`](./ch${chapter}_build.log)"
        else
            BUILD_LINE="- Build: skipped or not available"
        fi
    fi

    PROFILE_LINE="- Profile: not requested"
    if [[ "$DO_PROFILE" == true ]]; then
        if [[ -f "$OUTPUT_DIR/ch${chapter}_profile.log" ]]; then
            PROFILE_LINE="- Profile log: [\`ch${chapter}_profile.log\`](./ch${chapter}_profile.log)"
        else
            PROFILE_LINE="- Profile: not available"
        fi
    fi

    BENCHMARK_LINE="- Benchmark: not requested"
    if [[ "$DO_BENCHMARK" == true ]]; then
        if [[ -f "$OUTPUT_DIR/ch${chapter}_benchmark.log" ]]; then
            BENCHMARK_LINE="- Benchmark log: [\`ch${chapter}_benchmark.log\`](./ch${chapter}_benchmark.log)"
        else
            BENCHMARK_LINE="- Benchmark: not available"
        fi
    fi

    BENCHMARK_ARTIFACT_LINE=""
    if [[ "$DO_BENCHMARK" == true ]]; then
        if [[ -d "$OUTPUT_DIR/benchmarking/ch${chapter}" ]]; then
            BENCHMARK_ARTIFACT_LINE="- Benchmark artifacts: [\`benchmarking/ch${chapter}/\`](./benchmarking/ch${chapter}/)"
        else
            BENCHMARK_ARTIFACT_LINE="- Benchmark artifacts: not available"
        fi
    fi

    cat >> "$SUMMARY_FILE" <<EOF

### Chapter $chapter

$BUILD_LINE
$PROFILE_LINE
$BENCHMARK_LINE
${BENCHMARK_ARTIFACT_LINE:+$BENCHMARK_ARTIFACT_LINE}
EOF
done

echo "Summary report: $SUMMARY_FILE"
echo "========================================="
