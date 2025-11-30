#!/bin/bash
################################################################################
# GB200/GB300 Grace-Blackwell NUMA Optimization Script
################################################################################
#
# Optimizes system settings for GB200/GB300 superchips:
# - Grace CPU: 72 ARM Neoverse V2 cores, LPDDR5X memory
# - Blackwell GPU: Up to 8x B200 (180GB each)
# - NVLink-C2C: 900 GB/s CPU↔GPU coherent bandwidth
#
# Usage:
#   sudo ./cpu_gpu_numa_optimizations.sh [--apply|--check|--reset]
#
# Options:
#   --apply: Apply all optimizations (requires root)
#   --check: Check current settings (no changes)
#   --reset: Reset to default settings
#
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root for --apply
check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo -e "${RED}ERROR: This script must be run as root for --apply${NC}"
        echo "Usage: sudo ./cpu_gpu_numa_optimizations.sh --apply"
        exit 1
    fi
}

# Detect if this is a GB200/GB300 system
detect_grace_blackwell() {
    echo -e "${GREEN}=== Detecting System Configuration ===${NC}"
    
    # Check CPU architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" != "aarch64" ]]; then
        echo -e "${YELLOW}⚠ Not ARM architecture (found: $ARCH)${NC}"
        echo -e "${YELLOW}⚠ This script is optimized for Grace CPU${NC}"
        return 1
    fi
    
    echo -e "${GREEN}ARM64 architecture detected${NC}"
    
    # Check for Grace CPU (Neoverse)
    if grep -qi "neoverse\|grace" /proc/cpuinfo 2>/dev/null; then
        echo -e "${GREEN}Grace CPU detected${NC}"
    else
        echo -e "${YELLOW}⚠ Grace CPU not confirmed${NC}"
    fi
    
    # Check for NVIDIA GPUs
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo -e "${GREEN}Found $GPU_COUNT NVIDIA GPU(s)${NC}"
        
        # Check if Blackwell
        if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "b200\|b300\|blackwell"; then
            echo -e "${GREEN}Blackwell GPU detected${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ nvidia-smi not found${NC}"
    fi
    
    # Check NUMA nodes
    if command -v numactl &> /dev/null; then
        NUMA_NODES=$(numactl --hardware | grep "available:" | awk '{print $2}')
        echo -e "${GREEN}NUMA nodes: $NUMA_NODES${NC}"
    else
        echo -e "${RED}✗ numactl not installed${NC}"
        echo "  Install with: apt-get install numactl"
        return 1
    fi
    
    # Check total memory
    TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    echo -e "${GREEN}Total system memory: ${TOTAL_MEM_GB}GB${NC}"
    
    if [[ $TOTAL_MEM_GB -gt 400 ]]; then
        echo -e "${GREEN}Large memory configuration (Grace typical: 480GB-1TB)${NC}"
    fi
    
    echo ""
    return 0
}

# Check current settings
check_settings() {
    echo -e "${GREEN}=== Current System Settings ===${NC}"
    
    # CPU governor
    if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
        GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
        echo "CPU Governor: $GOVERNOR"
    fi
    
    # NUMA balancing
    if [[ -f /proc/sys/kernel/numa_balancing ]]; then
        NUMA_BAL=$(cat /proc/sys/kernel/numa_balancing)
        echo "NUMA Balancing: $NUMA_BAL"
    fi
    
    # Transparent huge pages
    if [[ -f /sys/kernel/mm/transparent_hugepage/enabled ]]; then
        THP=$(cat /sys/kernel/mm/transparent_hugepage/enabled | grep -o '\[.*\]' | tr -d '[]')
        echo "Transparent Huge Pages: $THP"
    fi
    
    # Check NVIDIA persistence mode
    if command -v nvidia-smi &> /dev/null; then
        PERSIST=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | head -1)
        echo "NVIDIA Persistence Mode: $PERSIST"
    fi
    
    echo ""
}

# Apply optimizations
apply_optimizations() {
    echo -e "${GREEN}=== Applying GB200/GB300 Optimizations ===${NC}"
    
    # 1. CPU Governor: Set to performance
    echo "1. Setting CPU governor to 'performance'..."
    if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo performance > $cpu
        done
        echo -e "${GREEN}   CPU governor set to performance${NC}"
    else
        echo -e "${YELLOW}   ⚠ CPU frequency scaling not available${NC}"
    fi
    
    # 2. Disable automatic NUMA balancing (we'll do manual placement)
    echo "2. Disabling automatic NUMA balancing..."
    if [[ -f /proc/sys/kernel/numa_balancing ]]; then
        echo 0 > /proc/sys/kernel/numa_balancing
        echo -e "${GREEN}   NUMA balancing disabled${NC}"
    fi
    
    # 3. Enable transparent huge pages (madvise mode for Grace)
    echo "3. Configuring transparent huge pages..."
    if [[ -f /sys/kernel/mm/transparent_hugepage/enabled ]]; then
        echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
        echo madvise > /sys/kernel/mm/transparent_hugepage/defrag
        echo -e "${GREEN}   THP set to madvise mode${NC}"
    fi
    
    # 4. Increase memory limits
    echo "4. Increasing memory limits..."
    ulimit -l unlimited 2>/dev/null || echo -e "${YELLOW}   ⚠ Could not set unlimited locked memory${NC}"
    ulimit -m unlimited 2>/dev/null || echo -e "${YELLOW}   ⚠ Could not set unlimited RSS${NC}"
    echo -e "${GREEN}   Memory limits configured${NC}"
    
    # 5. Configure swappiness (minimize swapping on 480GB+ systems)
    echo "5. Configuring swappiness..."
    if [[ -f /proc/sys/vm/swappiness ]]; then
        echo 10 > /proc/sys/vm/swappiness
        echo -e "${GREEN}   Swappiness set to 10 (minimal swapping)${NC}"
    fi
    
    # 6. NVIDIA GPU settings
    echo "6. Configuring NVIDIA GPU settings..."
    if command -v nvidia-smi &> /dev/null; then
        # Enable persistence mode
        nvidia-smi -pm 1
        echo -e "${GREEN}   Persistence mode enabled${NC}"
        
        # Set application clocks (if supported)
        nvidia-smi -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.sm --format=csv,noheader,nounits | head -1 | tr ',' ' ') 2>/dev/null || \
            echo -e "${YELLOW}   ⚠ Could not set application clocks${NC}"
        
        # Reset ECC errors
        nvidia-smi -r 2>/dev/null || echo -e "${YELLOW}   ⚠ Could not reset GPUs${NC}"
    fi
    
    # 7. IRQ affinity for network interfaces
    echo "7. Configuring IRQ affinity..."
    if command -v irqbalance &> /dev/null; then
        systemctl stop irqbalance 2>/dev/null || true
        echo -e "${GREEN}   IRQ balance stopped (for manual affinity)${NC}"
    fi
    
    # 8. Configure zone reclaim mode
    echo "8. Configuring zone reclaim mode..."
    if [[ -f /proc/sys/vm/zone_reclaim_mode ]]; then
        echo 0 > /proc/sys/vm/zone_reclaim_mode
        echo -e "${GREEN}   Zone reclaim disabled (prefer remote access)${NC}"
    fi
    
    # 9. Increase file descriptor limits
    echo "9. Increasing file descriptor limits..."
    echo "* soft nofile 1048576" >> /etc/security/limits.conf
    echo "* hard nofile 1048576" >> /etc/security/limits.conf
    echo -e "${GREEN}   File descriptor limits increased${NC}"
    
    # 10. Create NUMA binding script
    echo "10. Creating NUMA binding helper script..."
    cat > /usr/local/bin/numa_bind_gpu.sh << 'EOF'
#!/bin/bash
# Helper script to bind process to NUMA node matching GPU
GPU_ID=${1:-0}
NUMA_NODE=$((GPU_ID / 4))  # Assume 4 GPUs per NUMA node for 8-GPU system

echo "Binding to NUMA node $NUMA_NODE for GPU $GPU_ID"
exec numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE "${@:2}"
EOF
    chmod +x /usr/local/bin/numa_bind_gpu.sh
    echo -e "${GREEN}   NUMA binding helper created${NC}"
    echo "   Usage: numa_bind_gpu.sh <gpu_id> <command>"
    
    echo ""
    echo -e "${GREEN}=== Optimizations Applied ===${NC}"
    echo ""
    echo "To persist these settings across reboots, add to /etc/rc.local or systemd service"
}

# Reset to defaults
reset_settings() {
    echo -e "${YELLOW}=== Resetting to Default Settings ===${NC}"
    
    # CPU governor
    if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo ondemand > $cpu 2>/dev/null || echo powersave > $cpu 2>/dev/null
        done
        echo "CPU governor reset"
    fi
    
    # NUMA balancing
    if [[ -f /proc/sys/kernel/numa_balancing ]]; then
        echo 1 > /proc/sys/kernel/numa_balancing
        echo "NUMA balancing enabled"
    fi
    
    # THP
    if [[ -f /sys/kernel/mm/transparent_hugepage/enabled ]]; then
        echo always > /sys/kernel/mm/transparent_hugepage/enabled
        echo "THP reset to always"
    fi
    
    # Swappiness
    if [[ -f /proc/sys/vm/swappiness ]]; then
        echo 60 > /proc/sys/vm/swappiness
        echo "Swappiness reset to 60"
    fi
    
    # IRQ balance
    if command -v irqbalance &> /dev/null; then
        systemctl start irqbalance 2>/dev/null || true
        echo "IRQ balance restarted"
    fi
    
    echo ""
    echo "Settings reset to defaults"
}

# Print usage recommendations
print_recommendations() {
    echo -e "${GREEN}=== Usage Recommendations ===${NC}"
    echo ""
    echo "1. Launch multi-GPU training with NUMA awareness:"
    echo "   torchrun --nproc_per_node=8 train.py"
    echo ""
    echo "2. Bind specific process to NUMA node:"
    echo "   numa_bind_gpu.sh 0 python train.py  # GPU 0"
    echo "   numa_bind_gpu.sh 4 python train.py  # GPU 4"
    echo ""
    echo "3. Monitor NUMA memory usage:"
    echo "   numastat -m"
    echo ""
    echo "4. Check CPU-GPU affinity:"
    echo "   nvidia-smi topo -m"
    echo ""
    echo "5. Profile with NUMA awareness:"
    echo "   nsys profile --numa-node-affinity=true python train.py"
    echo ""
    echo "6. Verify NVLink-C2C bandwidth:"
    echo "   cd ch02 && ./gb200_coherency"
    echo ""
}

# Main
main() {
    MODE=${1:-check}
    
    case $MODE in
        --apply)
            check_root
            detect_grace_blackwell || echo "Proceeding anyway..."
            echo ""
            check_settings
            echo ""
            apply_optimizations
            echo ""
            check_settings
            echo ""
            print_recommendations
            ;;
        --check)
            detect_grace_blackwell || echo "Proceeding anyway..."
            echo ""
            check_settings
            echo ""
            print_recommendations
            ;;
        --reset)
            check_root
            reset_settings
            ;;
        --help|-h)
            echo "GB200/GB300 NUMA Optimization Script"
            echo ""
            echo "Usage: $0 [--apply|--check|--reset|--help]"
            echo ""
            echo "Options:"
            echo "  --apply  Apply all optimizations (requires root)"
            echo "  --check  Check current settings (default, no root needed)"
            echo "  --reset  Reset to default settings (requires root)"
            echo "  --help   Show this help message"
            ;;
        *)
            echo "Unknown option: $MODE"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

main "$@"
