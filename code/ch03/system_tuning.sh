#!/bin/bash

# System-wide performance tuning for GPU environments
# Run as root or with sudo

echo "Applying system-wide GPU performance optimizations..."

# 1. CPU Governor and C-states
echo "Setting CPU governor to performance mode..."
cpupower frequency-set -g performance
echo "Disabling deep C-states in BIOS (manual step required)"

# 2. Virtual Memory and Swapping
echo "Disabling swap and setting swappiness to 0..."
swapoff -a
echo 0 > /proc/sys/vm/swappiness

# 3. Transparent Huge Pages - enable for training, disable for inference
echo "Configuring Transparent Huge Pages..."
# For training workloads (throughput-focused):
echo always > /sys/kernel/mm/transparent_hugepage/enabled
# For inference workloads (latency-focused), use:
# echo never > /sys/kernel/mm/transparent_hugepage/enabled

# 4. Filesystem and I/O tuning
echo "Tuning filesystem cache settings..."
echo 20 > /proc/sys/vm/dirty_ratio
echo 10 > /proc/sys/vm/dirty_background_ratio

# 5. Network optimizations
echo "Optimizing network settings for RDMA/InfiniBand..."
echo 'net.core.rmem_max = 268435456' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 268435456' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 268435456' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 268435456' >> /etc/sysctl.conf
sysctl -p

# 6. Interrupt affinity (example for 8-core system)
echo "Setting interrupt affinity..."
# This would need to be customized per system
# Example: bind GPU interrupts to specific cores
for irq in $(grep nvidia /proc/interrupts | cut -d: -f1); do
    echo 2 > /proc/irq/$irq/smp_affinity  # Bind to CPU 1
done

# 7. ulimits for memory locking
echo "Setting unlimited locked memory..."
cat >> /etc/security/limits.conf << EOF
* soft memlock unlimited
* hard memlock unlimited
* soft nofile 1048576
* hard nofile 1048576
EOF

# 8. GPU-specific optimizations
echo "Configuring GPU settings..."
# Enable persistence mode for all GPUs
nvidia-smi -pm 1

# Enable MPS (optional, for multi-process scenarios)
# export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
# export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
# nvidia-cuda-mps-control -d

# 9. NUMA balancing
echo "Disabling automatic NUMA balancing..."
echo 0 > /proc/sys/kernel/numa_balancing

# 10. CPU isolation (example - isolate CPUs 2-7 for compute)
# This requires kernel parameter: isolcpus=2-7 nohz_full=2-7
echo "CPU isolation requires kernel boot parameters:"
echo "Add to GRUB: isolcpus=2-7 nohz_full=2-7 rcu_nocbs=2-7"

echo "System tuning complete. Reboot recommended for all changes to take effect."
echo "Remember to update /etc/default/grub with CPU isolation parameters if needed."
