#!/bin/bash

# GPU Setup Commands for Performance Optimization

# Enable GPU persistence mode
nvidia-smi -pm ENABLED
systemctl enable nvidia-persistenced

# Set CPU frequency to performance mode
cpupower frequency-set -g performance

# Disable swapping for better performance
sudo swapoff -a
# Or set swappiness to 0 (more conservative)
echo 0 | sudo tee /proc/sys/vm/swappiness

# Configure jemalloc for better memory allocation
export MALLOC_CONF="narenas:8,dirty_decay_ms:10000,muzzy_decay_ms:10000,background_thread:true"

# Configure tcmalloc for better memory allocation
export TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES=$((512*1024*1024))
export TCMALLOC_RELEASE_RATE=16

# Set unlimited locked memory (for pinned memory)
ulimit -l unlimited

# Enable GPUDirect RDMA and GPUDirect Storage paths
sudo modprobe nvidia-peermem || true
sudo modprobe nvidia-fs || true
export CUFILE_ENV_PATH_JSON=${CUFILE_ENV_PATH_JSON:-/etc/cufile.json}
# Validate GDS bandwidth (uncomment to run)
# gdsio -D 1M -s 16G -w /path/to/nvme

# Example NUMA binding command
# numactl --cpunodebind=1 --membind=1 python train.py --gpu 5

# Example docker run with GPU support and NUMA binding
# docker run --gpus all --privileged \
#    -v /data:/data \
#    --network=host \
#    --ulimit memlock=-1 \
#    --cpuset-cpus="0-7" \
#    nvcr.io/nvidia/pytorch:25.05-py3 \
#    python train.py
