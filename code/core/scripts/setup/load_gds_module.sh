#!/bin/bash
# Script to load GPUDirect Storage kernel module
# Run with: sudo ./load_gds_module.sh

set -e

echo "Loading NVIDIA GPUDirect Storage (GDS) kernel module..."

# Check if module is already loaded
if lsmod | grep -q nvidia_fs; then
    echo "[OK] nvidia-fs module is already loaded"
else
    echo "Loading nvidia-fs module..."
    modprobe nvidia-fs
    
    if lsmod | grep -q nvidia_fs; then
        echo "[OK] nvidia-fs module loaded successfully"
    else
        echo "ERROR: Failed to load nvidia-fs module"
        exit 1
    fi
fi

# Verify GDS is working
echo ""
echo "Checking GDS status..."
if [ -f /usr/local/cuda/gds/tools/gdscheck ]; then
    /usr/local/cuda/gds/tools/gdscheck -p
else
    echo "WARNING: gdscheck tool not found"
fi

echo ""
echo "[OK] GDS setup complete!"
echo ""
echo "You can now run: python3 ch05/gds_cufile_minimal.py"

