#!/bin/bash
# Test script for the automated analysis system

set -e

echo "Testing Automated Performance Analysis System"
echo "=============================================="
echo ""

# Create a test directory with sample outputs
TEST_DIR="test_results_analysis_demo"
mkdir -p "$TEST_DIR"

# Create sample ch2 output (NVLink bandwidth)
cat > "$TEST_DIR/ch2_nvlink.txt" << 'EOF'
NVLink C2C P2P Test
===================
Testing bandwidth between GPUs...

NVLink Bandwidth: 897.5 GB/s
PCIe Bandwidth: 63.2 GB/s

Status: PASS
EOF

# Create sample ch7 output (HBM3e bandwidth)
cat > "$TEST_DIR/ch7_hbm3e.txt" << 'EOF'
HBM3e Peak Bandwidth Test
=========================
Testing memory bandwidth...

Peak Bandwidth: 6.8 TB/s
Memory utilization: 87.2 %
Coalescing efficiency: 91.5 %

Status: PASS
EOF

# Create sample ch10 output (Tensor Cores)
cat > "$TEST_DIR/ch10_tcgen05.txt" << 'EOF'
Tensor Core Performance Test
============================
Testing FP8 and FP16 TFLOPS...

FP8 TFLOPS: 1285.3 TFLOPS
FP16 TFLOPS: 1156.7 TFLOPS
Tensor Core Utilization: 84.3 %

Status: PASS
EOF

# Create sample ch11 output (Streams)
cat > "$TEST_DIR/ch11_streams.txt" << 'EOF'
Stream Concurrency Test
=======================
Testing stream overlap...

Sequential time: 45.2 ms
Concurrent time: 27.8 ms
Speedup: 1.63x
Overlap efficiency: 68.5 %

Status: PASS
EOF

# Create sample ch14 output (torch.compile)
cat > "$TEST_DIR/ch14_torch_compile.txt" << 'EOF'
torch.compile Performance Test
==============================
Testing compilation speedup...

Eager mode: 12.4 ms
Compiled mode: 9.1 ms
Speedup: 1.36x

Triton FP8 vs FP16:
Speedup: 1.72x

DeepSeek L2 cache optimization:
Speedup: 1.18x

Status: PASS
EOF

echo "Created test directory: $TEST_DIR"
echo ""

# Run analysis on the test directory
echo "Running analysis..."
python3 tools/analyze_results.py --input "$TEST_DIR" --output "$TEST_DIR/analysis.md" --verbose

echo ""
echo "Analysis complete!"
echo ""
echo "Quick summary:"
python3 tools/analyze_results.py --input "$TEST_DIR" --quick

echo ""
echo "Full report saved to: $TEST_DIR/analysis.md"
echo ""

# Clean up
read -p "Press Enter to view full report (or Ctrl+C to exit)..."
cat "$TEST_DIR/analysis.md"

echo ""
echo "Test complete! Test directory can be found at: $TEST_DIR"


