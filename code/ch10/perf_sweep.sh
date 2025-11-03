#!/bin/bash
echo "=== Performance Sweep: Double-Buffered Pipeline GEMM ==="
echo ""
for size in 128 256 512 768 1024 1536 2048; do
    echo "Testing ${size}³:"
    ./double_buffered_pipeline $size $size $size 2>&1 | grep -E "(Selected|Note:|Naive time|Pipeline time|Speedup|Max difference)"
    echo ""
done
