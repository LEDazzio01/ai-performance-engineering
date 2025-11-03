#!/usr/bin/env bash
# Enhanced sanity-check script that validates profiling configuration and checks for failures.

set -euo pipefail

progress() {
  local current=$1
  local total=$2
  local message=$3
  local width=40
  local filled=$(( current * width / total ))
  local empty=$(( width - filled ))
  printf '\r[%s%s] %s' \
    "$(printf '%*s' "$filled" '' | tr ' ' '#')" \
    "$(printf '%*s' "$empty" '' | tr ' ' '-')" \
    "$message"
  if (( current == total )); then
    printf '\n'
  fi
}

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}
TOTAL_STEPS=6

echo "🔍 AI Performance Engineering - System Validation & Failure Analysis"
echo "=================================================================="

# Step 1: Check system dependencies
progress 1 "$TOTAL_STEPS" "Checking system dependencies"
echo ""
echo "📋 System Dependencies:"
echo "  Python: $(python3 --version 2>/dev/null || echo '❌ Not found')"
echo "  CUDA: $(nvcc --version 2>/dev/null | head -1 || echo '❌ Not found')"
echo "  Nsight Systems: $(nsys --version 2>/dev/null | head -1 || echo '❌ Not found')"
echo "  Nsight Compute: $(ncu --version 2>/dev/null | head -1 || echo '❌ Not found')"
echo "  numactl: $(numactl --hardware 2>/dev/null | head -1 || echo '❌ Not found')"
echo "  perf: $(perf --version 2>/dev/null | head -1 || echo '❌ Not found')"

# Step 2: Check GPU availability
progress 2 "$TOTAL_STEPS" "Checking GPU availability"
echo ""
echo "🎮 GPU Status:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | while read -r line; do
        echo "  ✅ $line"
    done
else
    echo "  ❌ nvidia-smi not found"
fi

# Step 3: Check PyTorch and CUDA
progress 3 "$TOTAL_STEPS" "Checking PyTorch and CUDA"
echo ""
echo "🔥 PyTorch Status:"
python3 -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('  ❌ CUDA not available')
"

# Step 4: Inspect example registry
progress 4 "$TOTAL_STEPS" "Inspecting example registry"
echo ""
echo "📚 Example Registry:"
python3 - <<'PY'
from scripts.harness.example_registry import EXAMPLES
from scripts.harness.metrics_config import resolve_overrides

print(f"  Total examples: {len(EXAMPLES)}")
print("  Sample examples:")
for i, example in enumerate(EXAMPLES[:5]):
    overrides = resolve_overrides(example)
    print(f"    {example.name} :: tags={example.tags}")
if len(EXAMPLES) > 5:
    print(f"    ... and {len(EXAMPLES) - 5} more")
PY

# Step 5: Check for recent profile failures
progress 5 "$TOTAL_STEPS" "Analyzing recent profile failures"
echo ""
echo "🚨 Recent Profile Session Analysis:"

LATEST_SESSION=$(ls -t "$CODE_ROOT/profiles/" 2>/dev/null | head -1)
if [[ -n "$LATEST_SESSION" && -f "$CODE_ROOT/profiles/$LATEST_SESSION/summary.json" ]]; then
    echo "  Latest session: $LATEST_SESSION"
    
    # Count failures by type
    echo "  📊 Results Summary:"
    python3 -c "
import json
import sys
from collections import defaultdict

try:
    with open('$CODE_ROOT/profiles/$LATEST_SESSION/summary.json', 'r') as f:
        results = json.load(f)
    
    # Count by profiler and exit code
    counts = defaultdict(lambda: defaultdict(int))
    for result in results:
        profiler = result['profiler']
        exit_code = result['exit_code']
        counts[profiler][exit_code] += 1
    
    print('    Profiler Results:')
    for profiler, exit_codes in counts.items():
        total = sum(exit_codes.values())
        success = exit_codes.get(0, 0)
        failed = total - success
        print(f'      {profiler}: {success}/{total} successful ({failed} failed)')
    
    # Show specific failures
    failures = [r for r in results if r['exit_code'] != 0]
    if failures:
        print('    ❌ Specific Failures:')
        for failure in failures[:10]:  # Show first 10 failures
            print(f'      {failure[\"example\"]} [{failure[\"profiler\"]}] (exit={failure[\"exit_code\"]})')
        if len(failures) > 10:
            print(f'      ... and {len(failures) - 10} more failures')
    else:
        print('    ✅ No failures detected!')
        
except Exception as e:
    print(f'    ❌ Error analyzing session: {e}')
"
else
    echo "  ℹ️  No recent profile sessions found"
fi

# Step 6: Dry-run validation
progress 6 "$TOTAL_STEPS" "Dry-running harness validation"
echo ""
echo "🧪 Harness Validation:"
if [[ -f "$CODE_ROOT/scripts/harness/profile_harness.py" ]]; then
    echo "  Running dry-run test (max 3 examples)..."
    python3 "$CODE_ROOT/scripts/harness/profile_harness.py" --profile all --dry-run --max-examples 3 2>&1 | tail -5
else
    echo "  ❌ profile_harness.py not found"
fi

echo ""
echo "🔧 Available Tools:"
echo "  ./setup.sh          # Install everything"
echo "  ./start.sh           # Start profiling"
echo "  ./stop.sh            # Stop profiling"
echo "  ./extract.sh         # Extract results"
echo "  rm -rf ./profiles/  # Clean old profiles"

echo ""
echo "📖 For detailed failure analysis:"
echo "  cat profiles/\$(ls -t profiles/ | head -1)/summary.json | jq '.[] | select(.exit_code != 0)'"
echo "  tail -f profile_runs/harness/latest.log"

echo ""
echo "✅ System validation complete!"
