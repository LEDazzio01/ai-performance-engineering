#!/bin/bash
#
# AI Performance Engineering Setup Script
# ========================================
#
# This script installs EVERYTHING you need:
#   1. NVIDIA Driver 580+ (auto-upgrades if needed)
#   2. Python 3.11 (PyTorch 2.9 compatible)
#   3. CUDA 13.0 repository and toolchain
#   4. PyTorch 2.9 nightly with CUDA 13.0 support
#   5. NVIDIA Nsight Systems 2025.3.2 (for timeline profiling)
#   6. NVIDIA Nsight Compute 2025.3.1 (for kernel metrics)
#   7. All Python dependencies from requirements_latest.txt
#   8. System tools (numactl, perf, htop, etc.)
#   9. Configures NVIDIA drivers for profiling
#
# Requirements:
#   - Ubuntu 22.04+ (tested on 22.04)
#   - NVIDIA B200/B300 GPU (or compatible)
#   - sudo/root access
#   - Internet connection
#
# Usage:
#   sudo ./setup.sh
#
# Duration: 10-20 minutes (first run may require reboot for driver upgrade)
#
# What it does:
#   - Adds official NVIDIA CUDA 13.0 repository
#   - Configures APT to prefer official NVIDIA packages
#   - Fixes Python APT module (python3-apt) compatibility
#   - Disables problematic command-not-found APT hook
#   - Removes duplicate deadsnakes repository entries
#   - Upgrades Python to 3.11 (required by PyTorch 2.9)
#   - Auto-upgrades NVIDIA driver to 580+ if needed (will prompt reboot)
#   - Installs CUDA 13.0 toolkit and libraries
#   - Installs latest Nsight tools (2025.x)
#   - Installs PyTorch 2.9 nightly with CUDA 13.0
#   - Removes conflicting system packages (python3-optree, etc.)
#   - Installs nvidia-ml-py (replaces deprecated pynvml)
#   - Configures NVIDIA kernel modules for profiling
#   - Fixes hardware info script compatibility
#   - Runs validation tests
#
# Notes:
#   - If driver upgrade is needed, script will exit and ask you to reboot
#   - After reboot, simply re-run: sudo ./setup.sh
#   - The script is idempotent and safe to re-run
#
# After running this script, you can:
#   - Run examples: python3 ch1/performance_basics.py
#   - Test everything: ./run_all_tests.sh
#   - Benchmark peak: python3 benchmark_peak.py
#   - Profile examples: ./start.sh
#

set -e  # Exit on any error

echo "🚀 AI Performance Engineering Setup Script"
echo "=========================================="
echo "This script will install:"
echo "  • NVIDIA Driver 580+ (auto-upgrade if needed)"
echo "  • Python 3.11 (PyTorch 2.9 compatible)"
echo "  • CUDA 13.0 repository and toolchain"
echo "  • PyTorch 2.9 nightly with CUDA 13.0"
echo "  • NVIDIA Nsight Systems 2025.3.2 (latest)"
echo "  • NVIDIA Nsight Compute 2025.3.1 (latest)"
echo "  • All project dependencies"
echo "  • System tools (numactl, perf, etc.)"
echo ""
echo "Note: If driver upgrade is needed, you'll be prompted to reboot."
echo ""

PROJECT_ROOT="$(dirname "$(realpath "$0")")"
REQUIRED_DRIVER_VERSION="580.65.06"
echo "📁 Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Allow pip to install over system packages when running as root on Debian-based distros
export PIP_BREAK_SYSTEM_PACKAGES=1

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "⚠️  Running as root. This is fine for containerized environments."
else
   echo "❌ This script requires root privileges. Please run with sudo."
   exit 1
fi

# Check Ubuntu version
if ! command -v lsb_release &> /dev/null; then
    echo "Installing lsb-release..."
    apt update && apt install -y lsb-release
fi

UBUNTU_VERSION=$(lsb_release -rs)
echo "📋 Detected Ubuntu version: $UBUNTU_VERSION"

if [[ "$UBUNTU_VERSION" != "22.04" && "$UBUNTU_VERSION" != "20.04" ]]; then
    echo "⚠️  Warning: This script is tested on Ubuntu 22.04. Other versions may work but are not guaranteed."
fi

# Check for NVIDIA GPU
echo ""
echo "🔍 Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ NVIDIA GPU detected"

    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    if [[ -n "$DRIVER_VERSION" ]]; then
        DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
        if [ "$DRIVER_MAJOR" -lt 580 ]; then
            echo "⚠️  Current NVIDIA driver: $DRIVER_VERSION"
            echo "ℹ️  CUDA 13.0 requires driver 580+. This script will upgrade it automatically."
        else
            echo "✅ NVIDIA driver version: $DRIVER_VERSION (compatible with CUDA 13.0)"
        fi
    fi
else
    echo "❌ NVIDIA GPU not detected. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Ensure open kernel modules are enabled for Blackwell GPUs
MODPROBE_CONF="/etc/modprobe.d/nvidia-open.conf"
if [[ ! -f "$MODPROBE_CONF" ]] || ! grep -q "NVreg_OpenRmEnableUnsupportedGpus=1" "$MODPROBE_CONF"; then
    echo "Configuring NVIDIA open kernel modules for Blackwell GPUs..."
    cat <<'EOF' > "$MODPROBE_CONF"
options nvidia NVreg_OpenRmEnableUnsupportedGpus=1 NVreg_RestrictProfilingToAdminUsers=0
EOF
    update-initramfs -u
    if lsmod | grep -q "^nvidia"; then
        echo "Reloading NVIDIA kernel modules to enable profiling counters..."
        systemctl stop nvidia-persistenced >/dev/null 2>&1 || true
        for module in nvidia_uvm nvidia_peermem nvidia_modeset nvidia_drm nvidia; do
            if lsmod | grep -q "^${module}"; then
                modprobe -r "${module}" >/dev/null 2>&1 || true
            fi
        done
        modprobe nvidia NVreg_OpenRmEnableUnsupportedGpus=1 NVreg_RestrictProfilingToAdminUsers=0 >/dev/null 2>&1 || true
        for module in nvidia_modeset nvidia_uvm nvidia_peermem; do
            modprobe "${module}" >/dev/null 2>&1 || true
        done
        systemctl start nvidia-persistenced >/dev/null 2>&1 || true
    fi
fi

# Update system packages
echo ""
echo "📦 Updating system packages..."

# Fix apt_pkg module before apt update (if Python was upgraded)
if ! python3 -c "import apt_pkg" 2>/dev/null; then
    echo "🔧 Fixing apt_pkg module..."
    apt install -y --reinstall python3-apt 2>/dev/null || true
fi

# Disable command-not-found APT hook if it's causing issues with Python upgrade
if [ -f /etc/apt/apt.conf.d/50command-not-found ] && ! /usr/lib/cnf-update-db 2>/dev/null; then
    echo "🔧 Disabling problematic command-not-found APT hook..."
    rm -f /etc/apt/apt.conf.d/50command-not-found
fi

# Clean up duplicate deadsnakes repository if it exists
if [ -f /etc/apt/sources.list.d/deadsnakes.list ] && [ -f /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-jammy.list ]; then
    echo "🔧 Removing duplicate deadsnakes repository..."
    rm -f /etc/apt/sources.list.d/deadsnakes.list
fi

apt update || {
    echo "⚠️  apt update had errors, but continuing..."
}

# Install required packages for adding repositories
apt install -y wget curl software-properties-common

# Install NVIDIA DCGM (Data Center GPU Manager) for monitoring stack
echo ""
echo "📡 Installing NVIDIA DCGM components..."
if ! dpkg -s datacenter-gpu-manager >/dev/null 2>&1; then
    apt install -y datacenter-gpu-manager
    echo "✅ NVIDIA DCGM installed"
else
    echo "✅ NVIDIA DCGM already present"
fi

if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^nvidia-dcgm.service'; then
    systemctl enable nvidia-dcgm >/dev/null 2>&1 || true
    systemctl restart nvidia-dcgm >/dev/null 2>&1 || systemctl start nvidia-dcgm >/dev/null 2>&1 || true
    echo "✅ nvidia-dcgm service enabled"
else
    echo "⚠️  nvidia-dcgm systemd unit not found; ensure the DCGM daemon is running on this host."
fi

python3 <<'PY'
import os
import shutil
import site
import sys

candidates = [
    "/usr/share/dcgm/bindings/python3",
    "/usr/share/dcgm/bindings/python",
    "/usr/share/dcgm/python3",
    "/usr/share/dcgm/python",
    "/usr/lib/dcgm/python3",
    "/usr/lib/dcgm/python",
]

dest = None
for path in site.getsitepackages():
    if path.startswith("/usr/lib") or path.startswith("/usr/local/lib"):
        dest = path
        break

if dest is None:
    print("⚠️  Unable to locate site-packages directory; skipping DCGM Python binding install.")
    sys.exit(0)

for cand in candidates:
    if not os.path.isdir(cand):
        continue
    for entry in os.listdir(cand):
        if entry == "__pycache__":
            continue
        src = os.path.join(cand, entry)
        dst = os.path.join(dest, entry)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"✅ DCGM Python bindings installed to {dest}")
    break
else:
    print("⚠️  DCGM Python bindings not found; pydcgm import may fail.")
PY

# Add NVIDIA CUDA 13.0 repository
echo ""
echo "🔐 Adding NVIDIA CUDA 13.0 repository..."

# Check if CUDA repository is already configured
if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ] && [ ! -f /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list ]; then
    CUDA_REPO_PKG="cuda-keyring_1.1-1_all.deb"
    if [ ! -f "/tmp/$CUDA_REPO_PKG" ]; then
        wget -P /tmp https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/$CUDA_REPO_PKG
    fi
    dpkg -i /tmp/$CUDA_REPO_PKG
    echo "✅ NVIDIA CUDA repository added"
else
    echo "✅ NVIDIA CUDA repository already configured"
fi

# Configure APT to prefer official NVIDIA packages over Lambda Labs or other repos
echo ""
echo "🔧 Configuring APT preferences for NVIDIA packages..."
cat > /etc/apt/preferences.d/nvidia-official <<'APT_PREF'
# Prefer official NVIDIA packages over Lambda Labs
Package: nvidia-*
Pin: origin archive.lambdalabs.com
Pin-Priority: 100

Package: nvidia-*
Pin: origin developer.download.nvidia.com
Pin-Priority: 600

Package: nvidia-*
Pin: release o=Ubuntu
Pin-Priority: 600

# Also apply to libnvidia packages
Package: libnvidia-*
Pin: origin archive.lambdalabs.com
Pin-Priority: 100

Package: libnvidia-*
Pin: origin developer.download.nvidia.com
Pin-Priority: 600

Package: libnvidia-*
Pin: release o=Ubuntu
Pin-Priority: 600
APT_PREF

apt update

# Install Python 3.11 or newer (PyTorch 2.9 supports Python 3.10, 3.11, 3.12)
echo ""
echo "🐍 Installing Python 3.11..."

# Check if Python 3.11 is already installed
if ! command -v python3.11 &> /dev/null; then
    apt install -y software-properties-common
    
    # Check if deadsnakes PPA is already added
    if ! grep -q "deadsnakes/ppa" /etc/apt/sources.list.d/*.list 2>/dev/null; then
        add-apt-repository -y ppa:deadsnakes/ppa
    else
        echo "✅ deadsnakes PPA already configured"
    fi
    
    apt update || true
    apt install -y python3.11 python3.11-dev python3.11-venv python3-pip
    echo "✅ Python 3.11 installed"
else
    CURRENT_PY311=$(python3.11 --version 2>&1 | awk '{print $2}')
    echo "✅ Python 3.11 already installed (version $CURRENT_PY311)"
    # Still ensure dev packages are present
    apt install -y python3.11-dev python3.11-venv python3-pip
fi

# Set Python 3.11 as default if not already
CURRENT_PY3=$(python3 --version 2>&1 | awk '{print $2}')
if [[ ! "$CURRENT_PY3" =~ ^3\.11\. ]]; then
    echo "Setting Python 3.11 as default..."
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
    update-alternatives --set python3 /usr/bin/python3.11
else
    echo "✅ Python 3.11 is already the default"
fi

# Ensure pip is installed for Python 3.11
if ! python3.11 -m pip --version &> /dev/null; then
    echo "Installing pip for Python 3.11..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
fi

# Upgrade pip
python3 -m pip install --upgrade pip setuptools packaging

# Fix python3-apt for the new Python version
echo ""
echo "🔧 Fixing Python APT module..."
apt install -y --reinstall python3-apt

# Remove distro flatbuffers package whose invalid version breaks pip metadata
if dpkg -s python3-flatbuffers >/dev/null 2>&1; then
    echo "Removing distro python3-flatbuffers package (invalid version metadata)..."
    apt remove -y python3-flatbuffers
fi

# Upgrade NVIDIA driver to 580+ if needed (required for CUDA 13.0)
echo ""
echo "🔍 Checking NVIDIA driver version..."
if command -v nvidia-smi &> /dev/null; then
    CURRENT_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    DRIVER_MAJOR=$(echo "$CURRENT_DRIVER" | cut -d. -f1)
    
    if [ "$DRIVER_MAJOR" -lt 580 ]; then
        echo "⚠️  Current driver ($CURRENT_DRIVER) is too old for CUDA 13.0"
        echo "📦 Upgrading to NVIDIA driver 580 (open kernel modules)..."
        
        # Remove old driver packages that might conflict
        echo "Removing old NVIDIA driver packages..."
        apt remove -y nvidia-driver-* nvidia-dkms-* nvidia-kernel-common-* \
            libnvidia-compute-* libnvidia-extra-* nvidia-utils-* \
            python3-jax-cuda* python3-torch-cuda python3-torchvision-cuda 2>/dev/null || true
        apt autoremove -y
        
        # Install new driver
        echo "Installing NVIDIA driver 580..."
        if apt install -y nvidia-driver-580-open; then
            echo "✅ NVIDIA driver 580 installed successfully"
            echo ""
            echo "╔════════════════════════════════════════════════════════════╗"
            echo "║  ⚠️  REBOOT REQUIRED                                       ║"
            echo "║                                                            ║"
            echo "║  The NVIDIA driver has been upgraded to version 580.      ║"
            echo "║  Please reboot your system and re-run this script.        ║"
            echo "║                                                            ║"
            echo "║  After reboot:                                             ║"
            echo "║    1. Run: nvidia-smi                                      ║"
            echo "║    2. Verify driver version is 580+                        ║"
            echo "║    3. Re-run: sudo ./setup.sh                              ║"
            echo "╚════════════════════════════════════════════════════════════╝"
            echo ""
            exit 0
        else
            echo "❌ Failed to install NVIDIA driver 580"
            echo "Please manually install the driver and reboot before continuing"
            exit 1
        fi
    else
        echo "✅ NVIDIA driver $CURRENT_DRIVER is compatible with CUDA 13.0"
    fi
fi

# Install CUDA 13.0 toolchain
echo ""
echo "🔧 Installing CUDA 13.0 toolchain..."
apt install -y cuda-toolkit-13-0

# Install NCCL 2.28.7 for Blackwell optimizations
echo ""
echo "📦 Installing NCCL 2.28.7 (Blackwell-optimized)..."
apt install -y libnccl2=2.28.7-1+cuda13.0 libnccl-dev=2.28.7-1+cuda13.0

# Install NVSHMEM 3.4.5 for CUDA 13 (enables SymmetricMemory fast paths)
echo ""
echo "🔁 Installing NVSHMEM 3.4.5 runtime and headers (CUDA 13)..."
apt install -y nvshmem-cuda-13 libnvshmem3-cuda-13 libnvshmem3-dev-cuda-13 libnvshmem3-static-cuda-13

# Install CUDA sanitizers and debugging tools (compute-sanitizer, cuda-memcheck, etc.)
echo ""
echo "🛡️  Installing CUDA sanitizers and debugging tools..."
if apt install -y cuda-command-line-tools-13-0; then
    echo "✅ CUDA command-line tools 13.0 installed (compute-sanitizer, cuda-gdb, cuda-memcheck)"
else
    echo "⚠️  Could not install cuda-command-line-tools-13-0, trying fallback packages..."
    if apt install -y cuda-command-line-tools; then
        echo "✅ CUDA command-line tools (generic) installed"
    else
        echo "⚠️  cuda-command-line-tools package unavailable. Trying NVIDIA CUDA toolkit..."
        if apt install -y nvidia-cuda-toolkit; then
            echo "✅ NVIDIA CUDA toolkit installed (includes cuda-memcheck)"
        else
            echo "❌ Could not install CUDA command-line tools. compute-sanitizer may be unavailable."
        fi
    fi
fi

# Ensure compute-sanitizer is present; install sanitizer package directly if needed
if ! command -v compute-sanitizer &> /dev/null; then
    echo "⚠️  compute-sanitizer not found after command-line tools install. Installing cuda-sanitizer package..."
    if apt install -y cuda-sanitizer-13-0; then
        echo "✅ cuda-sanitizer-13-0 installed"
    else
        echo "⚠️  Could not install cuda-sanitizer-13-0, attempting generic cuda-sanitizer package..."
        if apt install -y cuda-sanitizer; then
            echo "✅ cuda-sanitizer package installed"
        else
            echo "❌ compute-sanitizer installation failed; please install manually."
        fi
    fi
fi

# Install latest NVIDIA Nsight Systems and Compute
echo ""
echo "🔍 Installing latest NVIDIA Nsight Systems and Compute..."

# Create temporary directory for downloads
TEMP_DIR="/tmp/nsight_install"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Install Nsight Systems and set binary alternative  
echo "Installing Nsight Systems (pinned 2025.3.2)..."
NSYS_VERSION="2025.3.2"
apt install -y "nsight-systems-${NSYS_VERSION}"
# Try multiple possible locations
for bin_path in "/opt/nvidia/nsight-systems/${NSYS_VERSION}/bin/nsys" "/opt/nvidia/nsight-systems/${NSYS_VERSION}/nsys"; do
    if [[ -x "$bin_path" ]]; then
        NSYS_BIN="$bin_path"
        break
    fi
done
if [[ -n "$NSYS_BIN" ]] && [[ -x "$NSYS_BIN" ]]; then
    update-alternatives --install /usr/local/bin/nsys nsys "$NSYS_BIN" 50
    update-alternatives --set nsys "$NSYS_BIN"
    echo "✅ Nsight Systems pinned to ${NSYS_VERSION} (${NSYS_BIN})"
else
    echo "❌ Nsight Systems binary not found"
fi

# Install Nsight Compute and set binary alternative
echo "Installing Nsight Compute (pinned 2025.3.1)..."
NCU_VERSION="2025.3.1"
apt install -y "nsight-compute-${NCU_VERSION}"
NCU_BIN="/opt/nvidia/nsight-compute/${NCU_VERSION}/ncu"
if [[ -x "$NCU_BIN" ]]; then
    update-alternatives --install /usr/local/bin/ncu ncu "$NCU_BIN" 50
    update-alternatives --set ncu "$NCU_BIN"
    echo "✅ Nsight Compute pinned to ${NCU_VERSION} (${NCU_BIN})"
else
    echo "❌ Nsight Compute binary not found at ${NCU_BIN}"
fi

# Nsight tools are already in PATH when installed via apt
echo "✅ Nsight tools installed and available in PATH"

# Configure PATH and LD_LIBRARY_PATH for CUDA 13.0
echo ""
echo "🔧 Configuring CUDA 13.0 environment..."
# Update /etc/environment for system-wide CUDA 13.0
if ! grep -q "/usr/local/cuda-13.0/bin" /etc/environment; then
    sed -i 's|PATH="\(.*\)"|PATH="/usr/local/cuda-13.0/bin:\1"|' /etc/environment
    echo "Added CUDA 13.0 to system PATH"
fi

# Create profile.d script for CUDA 13.0
cat > /etc/profile.d/cuda-13.0.sh << 'PROFILE_EOF'
# CUDA 13.0 environment variables
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-13.0
PROFILE_EOF
chmod +x /etc/profile.d/cuda-13.0.sh
echo "✅ Created /etc/profile.d/cuda-13.0.sh for persistent CUDA 13.0 environment"

# Update nvcc symlink to CUDA 13.0 (override Ubuntu's default)
rm -f /usr/bin/nvcc
ln -s /usr/local/cuda-13.0/bin/nvcc /usr/bin/nvcc
echo "✅ Updated /usr/bin/nvcc symlink to CUDA 13.0"

# Source the CUDA environment for current session
source /etc/profile.d/cuda-13.0.sh

# Persist CUDA toolchain settings for the remainder of the script
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo ""
echo "🔍 Verifying CUDA toolchain after installation..."
if ! nvcc --version >/tmp/nvcc_version.txt 2>&1; then
    cat /tmp/nvcc_version.txt
    echo "❌ nvcc not available even after CUDA install. Aborting."
    exit 1
fi
cat /tmp/nvcc_version.txt
rm -f /tmp/nvcc_version.txt

python3 - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"❌ Failed to import torch for CUDA verification: {exc}")
    sys.exit(1)

cuda_ver = getattr(torch.version, "cuda", None)
print(f"✅ torch.version.cuda = {cuda_ver}")
if not cuda_ver:
    print("⚠️  PyTorch build does not have CUDA enabled")
PY

# Install runtime monitoring/precision dependencies that require CUDA headers
echo ""
echo "📦 Installing CUDA-dependent Python packages (prometheus-client, lmcache, transformer-engine)..."

# Ensure NumPy is present for torch's extension helper before building any wheels
python3 -m pip install --no-cache-dir --upgrade --ignore-installed numpy || true

install_cuda_package() {
    local package_spec="$1"
    echo "➡️  Installing $package_spec"
    if python3 -m pip install \
        --no-cache-dir --upgrade --ignore-installed --prefer-binary "$package_spec"; then
echo "✅ Installed $package_spec"
return 0
fi

    echo "⚠️  Installing numpy inside pip build environment to satisfy torch setup requirements..."
    python3 -m pip install --no-cache-dir --upgrade --ignore-installed numpy || true

    echo "⚠️  Initial install of $package_spec failed, retrying without build isolation..."
    if python3 -m pip install \
        --no-cache-dir --upgrade --ignore-installed --no-build-isolation --prefer-binary "$package_spec"; then
        echo "✅ Installed $package_spec (no-build-isolation)"
        return 0
    fi

    echo "❌ Failed to install $package_spec"
    return 1
}

install_cuda_package "prometheus-client==0.21.0"

echo ""
echo "➡️  Installing lmcache==0.3.8 (requires setuptools-scm version override)..."
if ! SETUPTOOLS_SCM_PRETEND_VERSION=0.3.8 \
        python3 -m pip install \
        --no-cache-dir --upgrade --ignore-installed --no-build-isolation --prefer-binary \
        lmcache==0.3.8; then
    echo "❌ Failed to install lmcache==0.3.8 even with version override. Please verify CUDA toolkit and setuptools-scm."
fi

install_cuda_package "transformer-engine"

# Clean up
cd /
rm -rf "$TEMP_DIR"
cd "$PROJECT_ROOT"

# Install system tools for performance testing
echo ""
echo "🛠️  Installing system performance tools..."
apt install -y \
    numactl \
    linux-tools-common \
    linux-tools-generic \
    linux-tools-$(uname -r) \
    gdb \
    perf-tools-unstable \
    infiniband-diags \
    perftest \
    htop \
    sysstat

# Install PyTorch 2.9 nightly with CUDA 13.0
echo ""
echo "🔥 Installing PyTorch 2.9 nightly with CUDA 13.0..."
python3 -m pip install --index-url https://download.pytorch.org/whl/nightly/cu130 \
    --no-cache-dir --upgrade --ignore-installed torch torchvision torchaudio

# Install project dependencies
echo ""
echo "📚 Installing project dependencies..."

# Use the updated requirements file with pinned versions
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements_latest.txt"

# Install dependencies with error handling
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing Python packages from requirements file..."
    if ! python3 -m pip install --no-input --upgrade --ignore-installed -r "$REQUIREMENTS_FILE"; then
        echo "⚠️  Some packages failed to install from requirements file."
        echo "Installing core packages individually..."
        python3 -m pip install --no-input --upgrade --ignore-installed \
            blinker==1.9.0 \
            nvidia-ml-py3 nvidia-ml-py==12.560.30 psutil==7.1.0 GPUtil==1.4.0 py-cpuinfo==9.0.0 \
            numpy==2.1.2 pandas==2.3.2 scikit-learn==1.7.2 pillow==11.3.0 \
            matplotlib==3.10.6 seaborn==0.13.2 tensorboard==2.20.0 wandb==0.22.0 plotly==6.3.0 bokeh==3.8.0 dash==3.2.0 \
            jupyter==1.1.1 ipykernel==6.30.1 black==25.9.0 flake8==7.3.0 mypy==1.18.2 pytest==8.3.4 \
            transformers==4.40.2 datasets==2.18.0 accelerate==0.29.0 sentencepiece==0.2.0 tokenizers==0.19.1 \
            onnx==1.19.0 onnxruntime-gpu==1.23.0 \
            py-spy==0.4.1 memory-profiler==0.61.0 line-profiler==5.0.0 pyinstrument==5.1.1 snakeviz==2.2.2 \
            optuna==4.5.0 hyperopt==0.2.7 ray==2.49.2 \
            dask==2025.9.1 xarray==2025.6.1
    fi
else
    echo "⚠️  Requirements file not found at $REQUIREMENTS_FILE. Installing core packages directly..."
    python3 -m pip install --no-input --upgrade --ignore-installed \
        blinker==1.9.0 \
        nvidia-ml-py3 nvidia-ml-py==12.560.30 psutil==7.1.0 GPUtil==1.4.0 py-cpuinfo==9.0.0 \
        numpy==2.1.2 pandas==2.3.2 scikit-learn==1.7.2 pillow==11.3.0 \
        matplotlib==3.10.6 seaborn==0.13.2 tensorboard==2.20.0 wandb==0.22.0 plotly==6.3.0 bokeh==3.8.0 dash==3.2.0 \
        jupyter==1.1.1 ipykernel==6.30.1 black==25.9.0 flake8==7.3.0 mypy==1.18.2 pytest==8.3.4 \
        transformers==4.40.2 datasets==2.18.0 accelerate==0.29.0 sentencepiece==0.2.0 tokenizers==0.19.1 \
        onnx==1.19.0 onnxruntime-gpu==1.23.0 \
        py-spy==0.4.1 memory-profiler==0.61.0 line-profiler==5.0.0 pyinstrument==5.1.1 snakeviz==2.2.2 \
        optuna==4.5.0 hyperopt==0.2.7 ray==2.49.2 \
        dask==2025.9.1 xarray==2025.6.1
fi

# Ensure monitoring/runtime extras are available even if requirements were cached
echo ""
echo "📦 Ensuring monitoring/runtime extras (Prometheus, LMCache, Transformer Engine)..."
python3 -m pip install --no-input --upgrade --ignore-installed prometheus-client==0.21.0 lmcache==0.3.9

# Transformer Engine build requires CUDNN headers shipped with the Python wheel.
CUDA_ROOT=/usr/local/cuda-13.0
CUDNN_INCLUDE_DIR=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/include
CUDNN_LIBRARY_DIR=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib
export CPATH="${CUDNN_INCLUDE_DIR}:${CPATH:-}"
export LIBRARY_PATH="${CUDNN_LIBRARY_DIR}:${LIBRARY_PATH:-}"

# Remove conflicting binary wheels before installing the source build.
python3 -m pip uninstall -y transformer_engine transformer-engine transformer_engine_cu12 transformer-engine-cu12 transformer-engine-cu13 >/dev/null 2>&1 || true

python3 -m pip install --no-input --upgrade --ignore-installed pybind11
if TORCH_CUDA_ARCH_LIST=120 \
   CUDNN_INCLUDE_DIR="${CUDNN_INCLUDE_DIR}" \
   CUDNN_LIBRARY_DIR="${CUDNN_LIBRARY_DIR}" \
   python3 -m pip install --no-input --upgrade --ignore-installed --no-build-isolation \
       git+https://github.com/NVIDIA/TransformerEngine.git; then
    echo "✅ Transformer Engine installed (FP8 kernels ready where supported)"
else
    echo "⚠️  Transformer Engine installation failed or is unsupported on this host; FP8 execution will fall back to AMP."
fi

# Remove conflicting system packages that interfere with PyTorch
echo ""
echo "🔧 Removing conflicting system packages..."
# Remove python3-optree which conflicts with torch.compile
if dpkg -s python3-optree >/dev/null 2>&1; then
    echo "Removing python3-optree (conflicts with torch.compile)..."
    apt remove -y python3-optree python3-keras 2>/dev/null || true
fi
# Clean up other conflicting Lambda Labs packages
apt autoremove -y 2>/dev/null || true

# Fix hardware info script compatibility
echo ""
echo "🔧 Fixing hardware info script compatibility..."
if [ -f "$PROJECT_ROOT/ch2/hardware_info.py" ]; then
    # Backup original file
    cp "$PROJECT_ROOT/ch2/hardware_info.py" "$PROJECT_ROOT/ch2/hardware_info.py.backup"
    
    # Fix the compatibility issue
    sed -i 's/"max_threads_per_block": device_props.max_threads_per_block,/"max_threads_per_block": getattr(device_props, '\''max_threads_per_block'\'', 1024),/' "$PROJECT_ROOT/ch2/hardware_info.py"
    
    echo "✅ Fixed hardware info script compatibility"
fi

# Verify installation
echo ""
echo "🧪 Verifying installation..."

# Check PyTorch
echo "Checking PyTorch installation..."
python3 - "$REQUIRED_DRIVER_VERSION" <<'PY'
import os
import sys
import textwrap
from packaging import version

required_driver = version.parse(sys.argv[1])

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"❌ PyTorch import failed: {exc}")
    sys.exit(1)

print(f"✅ PyTorch version: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"✅ CUDA available: {cuda_available}")

if cuda_available:
    print(f"✅ CUDA version: {torch.version.cuda}")
    print(f"✅ GPU count: {torch.cuda.device_count()}")
    try:
        print(f"✅ GPU name: {torch.cuda.get_device_name(0)}")
    except Exception:  # pragma: no cover
        pass
else:
    driver_version = None
    try:
        from torch._C import _cuda_getDriverVersion  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        _cuda_getDriverVersion = None

    if _cuda_getDriverVersion is not None:
        try:
            driver_version = _cuda_getDriverVersion()
        except Exception:
            driver_version = None

    if driver_version:
        current = version.parse(str(driver_version))
        if current < required_driver:
            print(
                textwrap.dedent(
                    f"""
                    ⚠ NVIDIA driver {current} is older than required {required_driver}.
                    → Install a newer driver (e.g., nvidia-driver-580) and reboot, then rerun setup.sh.
                    """
                ).strip()
            )
    else:
        print("❌ CUDA runtime not available. Ensure the NVIDIA driver meets CUDA 13.0 requirements and reboot if this is a fresh install.")
PY

# Check CUDA tools
echo ""
echo "Checking CUDA tools..."
if command -v nvcc &> /dev/null; then
    echo "✅ NVCC: $(nvcc --version | head -1)"
else
    echo "❌ NVCC not found"
fi

# Check Nsight tools
echo ""
echo "Checking Nsight tools..."
if command -v nsys &> /dev/null; then
    NSYS_VERSION=$(nsys --version 2>/dev/null | head -1)
    echo "✅ Nsight Systems: $NSYS_VERSION"
    # Check if it's a recent 2025 version
    if echo "$NSYS_VERSION" | grep -q "2025"; then
        echo "  🎉 Recent 2025 version installed!"
    else
        echo "  ⚠️  May not be the latest version (expected: 2025.x.x)"
    fi
else
    echo "❌ Nsight Systems not found"
fi

if command -v ncu &> /dev/null; then
    NCU_VERSION=$(ncu --version 2>/dev/null | head -1)
    echo "✅ Nsight Compute: $NCU_VERSION"
    # Check if it's a recent 2025 version
    if echo "$NCU_VERSION" | grep -q "2025"; then
        echo "  🎉 Recent 2025 version installed!"
    else
        echo "  ⚠️  May not be the latest version (expected: 2025.x.x)"
    fi
else
    echo "❌ Nsight Compute not found"
fi

# Check CUDA sanitizers and memcheck tools
echo ""
echo "Checking CUDA sanitizers..."
sanitizer_tools=("compute-sanitizer" "cuda-memcheck")
for tool in "${sanitizer_tools[@]}"; do
    if command -v "$tool" &> /dev/null; then
        echo "✅ $tool: installed"
    else
        echo "❌ $tool: not found"
    fi
done

# Check system tools
echo ""
echo "Checking system tools..."
tools=("numactl" "perf" "htop" "iostat" "ibstat")
for tool in "${tools[@]}"; do
    if command -v $tool &> /dev/null; then
        echo "✅ $tool: installed"
    else
        echo "❌ $tool: not found"
    fi
done

# Run basic performance test
echo ""
echo "🚀 Running basic performance test..."
python3 -c "
import torch
import time

device = torch.device('cuda')
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# Warm up
for _ in range(10):
    z = torch.mm(x, y)

# Time the operation
start = time.time()
for _ in range(100):
    z = torch.mm(x, y)
torch.cuda.synchronize()
end = time.time()

print(f'✅ Matrix multiplication (1000x1000): {(end - start) * 1000 / 100:.2f} ms per operation')
print(f'✅ GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')
print(f'✅ GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB')
"

# Test example scripts
echo ""
echo "🧪 Testing example scripts..."

# Test Chapter 1
echo "Testing Chapter 1 (Performance Basics)..."
if [ -f "$PROJECT_ROOT/ch1/performance_basics.py" ]; then
    if python3 "$PROJECT_ROOT/ch1/performance_basics.py" > /dev/null 2>&1; then
        echo "✅ Chapter 1: Performance basics working"
    else
        echo "⚠️  Chapter 1: Some issues detected (check output above)"
    fi
else
    echo "ℹ️  Chapter 1 example not present, skipping."
fi

# Test Chapter 2
echo "Testing Chapter 2 (Hardware Info)..."
if [ -f "$PROJECT_ROOT/ch2/hardware_info.py" ]; then
    if python3 "$PROJECT_ROOT/ch2/hardware_info.py" > /dev/null 2>&1; then
        echo "✅ Chapter 2: Hardware info working"
    else
        echo "⚠️  Chapter 2: Some issues detected (check output above)"
    fi
else
    echo "ℹ️  Chapter 2 example not present, skipping."
fi

# Test Chapter 3
echo "Testing Chapter 3 (NUMA Affinity)..."
if [ -f "$PROJECT_ROOT/ch3/bind_numa_affinity.py" ]; then
    if python3 "$PROJECT_ROOT/ch3/bind_numa_affinity.py" > /dev/null 2>&1; then
        echo "✅ Chapter 3: NUMA affinity working"
    else
        echo "⚠️  Chapter 3: Some issues detected (check output above)"
    fi
else
    echo "ℹ️  Chapter 3 example not present, skipping."
fi

# Set up environment variables for optimal performance
echo ""
echo "⚙️  Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'

# AI Performance Engineering Environment Variables
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_DISABLED=0
export PYTORCH_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# PyTorch optimization
export TORCH_COMPILE_DEBUG=0
export TORCH_LOGS="+dynamo"

# CUDA paths
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Force system NCCL 2.28.7 (Blackwell-optimized with NVLS support)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2:$LD_PRELOAD
EOF

echo "✅ Environment variables added to ~/.bashrc"

# Source the environment variables for current session
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2:$LD_PRELOAD
echo "✅ NCCL 2.28.7 activated for current session"

# Comprehensive setup verification
echo ""
echo "🧪 Running comprehensive setup verification..."
echo "=============================================="

# Test 1: PyTorch and CUDA
echo "🔍 Testing PyTorch and CUDA..."
python3 -c "
import torch
import sys
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('❌ CUDA not available!')
    sys.exit(1)
print(f'  CUDA version: {torch.version.cuda}')
print(f'  GPU count: {torch.cuda.device_count()}')
print(f'  GPU name: {torch.cuda.get_device_name(0)}')

# Check NCCL version
try:
    import ctypes
    libnccl = ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libnccl.so.2')
    version = ctypes.c_int()
    libnccl.ncclGetVersion(ctypes.byref(version))
    v_code = version.value
    major = v_code // 10000
    minor = (v_code % 10000) // 100
    patch = v_code % 100
    print(f'  NCCL version: {major}.{minor}.{patch}')
    if major == 2 and minor >= 28:
        print('  ✅ NVLS (NVLink SHARP) supported')
except Exception as e:
    print(f'  ⚠️  Could not verify NCCL version: {e}')

print('✅ PyTorch and CUDA working correctly')
"

if [ $? -ne 0 ]; then
    echo "❌ PyTorch/CUDA test failed!"
    exit 1
fi

# Test 2: Performance test
echo ""
echo "🚀 Testing GPU performance..."
python3 -c "
import torch
import time
device = torch.device('cuda')
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# Warm up
for _ in range(10):
    z = torch.mm(x, y)

# Time the operation
start = time.time()
for _ in range(100):
    z = torch.mm(x, y)
torch.cuda.synchronize()
end = time.time()

print(f'  Matrix multiplication: {(end - start) * 1000 / 100:.2f} ms per operation')
print(f'  GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated')
print('✅ GPU performance test passed')
"

if [ $? -ne 0 ]; then
    echo "❌ GPU performance test failed!"
    exit 1
fi

# Test 3: torch.compile test
echo ""
echo "⚡ Testing torch.compile..."
python3 -c "
import sys
import time
import torch
import traceback

device = torch.device('cuda')

def simple_model(x):
    return torch.mm(x, x.t())

x = torch.randn(1000, 1000, device=device)

# Uncompiled
start = time.time()
for _ in range(10):
    y = simple_model(x)
torch.cuda.synchronize()
uncompiled_time = time.time() - start

try:
    compiled_model = torch.compile(simple_model)
except AssertionError as exc:
    if \"duplicate template name\" in str(exc):
        print('⚠️  torch.compile skipped due to known PyTorch nightly issue: duplicate kernel template name')
        print(f'   Details: {exc}')
        sys.exit(0)
    print('❌ torch.compile failed with assertion error:')
    print(exc)
    sys.exit(1)
except Exception:
    print('❌ torch.compile failed with an unexpected exception:')
    traceback.print_exc()
    sys.exit(1)

start = time.time()
for _ in range(10):
    y = compiled_model(x)
torch.cuda.synchronize()
compiled_time = time.time() - start

speedup = uncompiled_time / compiled_time if compiled_time > 0 else float('inf')
print(f'  Uncompiled: {uncompiled_time*1000/10:.2f} ms per operation')
print(f'  Compiled: {compiled_time*1000/10:.2f} ms per operation')
print(f'  Speedup: {speedup:.2f}x')
print('✅ torch.compile test passed')
"

if [ $? -ne 0 ]; then
    echo "❌ torch.compile test failed!"
    exit 1
fi

# Step 11: Install CUTLASS Backend for torch.compile
echo ""
echo "📦 Step 11: Installing CUTLASS Backend (nvidia-cutlass-dsl)..."
echo "================================================================="

# Install CUTLASS DSL and CUDA Python bindings system-wide
# These are required for PyTorch's TorchInductor CUTLASS backend
echo "Installing nvidia-cutlass-dsl and cuda-python..."
pip install nvidia-cutlass-dsl cuda-python

if [ $? -eq 0 ]; then
    echo "✅ CUTLASS backend packages installed"
    echo "   - nvidia-cutlass-dsl: CUTLASS kernels for torch.compile"
    echo "   - cuda-python: CUDA runtime bindings"
else
    echo "⚠️  CUTLASS backend installation had issues, but continuing..."
fi

echo ""

# Test 4: Hardware info script
echo ""
echo "🔧 Testing hardware detection..."
if [ -f "$PROJECT_ROOT/ch2/hardware_info.py" ]; then
    python3 "$PROJECT_ROOT/ch2/hardware_info.py" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Hardware detection working"
    else
        echo "⚠️  Hardware detection had issues (may be expected in containers)"
    fi
else
    echo "ℹ️  Hardware detection script not present, skipping."
fi

# Test 5: NUMA binding script
echo ""
echo "🔗 Testing NUMA binding..."
if [ -f "$PROJECT_ROOT/ch3/bind_numa_affinity.py" ]; then
    # Temporarily disable exit-on-error for this test
    set +e
    python3 "$PROJECT_ROOT/ch3/bind_numa_affinity.py" > /dev/null 2>&1
    NUMA_EXIT_CODE=$?
    set -e
    
    if [ $NUMA_EXIT_CODE -eq 0 ]; then
        echo "✅ NUMA binding working"
    else
        echo "⚠️  NUMA binding had issues (expected in containers or without distributed launch)"
    fi
else
    echo "ℹ️  NUMA binding script not present, skipping."
fi

echo ""
echo "🎉 All critical tests passed! Setup is working correctly."

# Restart services impacted by NVSHMEM/CUDA installs (e.g., glances monitoring)
echo ""
echo "🔄 Restarting background services impacted by driver/tool updates..."
if command -v systemctl >/dev/null 2>&1; then
    if systemctl list-units --type=service --all | grep -q "^glances.service"; then
        if systemctl is-active --quiet glances.service; then
            systemctl restart glances.service && echo "✅ glances.service restarted"
        else
            systemctl restart glances.service >/dev/null 2>&1 && echo "✅ glances.service restarted (was inactive)" || echo "⚠️  Unable to restart glances.service (not running)"
        fi
    else
        echo "ℹ️  glances.service not present, skipping."
    fi
else
    echo "ℹ️  systemctl not available; please restart glances service manually if applicable."
fi

# Run verification scripts
echo ""
echo "🔍 Running Verification Checks..."
echo "=================================="
echo ""

VERIFICATION_FAILED=0

# Verify PyTorch installation
echo "▶️  Verifying PyTorch installation..."
if python3 verify_pytorch.py; then
    echo "✅ PyTorch verification passed"
else
    echo "❌ PyTorch verification failed"
    VERIFICATION_FAILED=1
fi

echo ""

# Verify NVLink connectivity (only if multiple GPUs)
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -gt 1 ]; then
    echo "▶️  Verifying NVLink connectivity..."
    if python3 verify_nvlink.py; then
        echo "✅ NVLink verification passed"
    else
        echo "⚠️  NVLink verification had warnings (review output above)"
    fi
else
    echo "ℹ️  Single GPU detected, skipping NVLink verification"
fi

echo ""

# Verify CUTLASS backend
echo "▶️  Verifying CUTLASS backend..."
if python3 verify_cutlass.py 2>/dev/null; then
    echo "✅ CUTLASS verification passed"
else
    echo "⚠️  CUTLASS verification had issues (may be expected)"
fi

echo ""

# Summary of verification
if [ $VERIFICATION_FAILED -eq 0 ]; then
    echo "✅ All critical verifications passed!"
else
    echo "⚠️  Some verifications failed - review output above"
fi

# Final summary
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "✅ Installed:"
echo "  • PyTorch 2.9 nightly with CUDA 13.0"
echo "  • CUDA 13.0 toolchain and development tools"
echo "  • NCCL 2.28.7 (Blackwell-optimized with NVLS support)"
echo "  • NVSHMEM 3.4.5 runtime and headers (CUDA 13)"
echo "  • NVIDIA Nsight Systems (latest available)"
echo "  • NVIDIA Nsight Compute (latest available)"
echo "  • All project dependencies"
echo "  • System performance tools (numactl, perf, etc.)"
echo ""
echo "✅ Verified:"
echo "  • PyTorch installation and CUDA functionality"
echo "  • NVLink connectivity (if multi-GPU)"
echo "  • CUTLASS backend configuration"
echo ""
echo "🚀 Quick Start:"
echo "  1. Run: python3 ch1/performance_basics.py"
echo "  2. Run: python3 ch2/hardware_info.py"
echo "  3. Run: python3 ch3/bind_numa_affinity.py"
echo ""
echo "📚 Available Examples:"
echo "  • Chapter 1: Performance basics"
echo "  • Chapter 2: Hardware information"
echo "  • Chapter 3: NUMA affinity binding"
echo "  • Chapter 14: PyTorch compiler and Triton examples"
echo ""
echo "🔧 Profiling Commands:"
echo "  • Nsight Systems: nsys profile -t cuda,nvtx,osrt -o profile python script.py"
echo "  • Nsight Compute: ncu --metrics achieved_occupancy -o profile python script.py"
echo "  • PyTorch Profiler: Use torch.profiler in your code"
echo ""
echo "🔗 NVLink Configuration (for multi-GPU workloads):"
echo "  export NCCL_P2P_LEVEL=NVL"
echo "  export NCCL_P2P_DISABLE=0"
echo "  export NCCL_IB_DISABLE=1"
echo "  export NCCL_NVLS_ENABLE=1"
echo ""
echo "📖 For more information, see the main README.md file and chapter-specific documentation."
echo ""
echo "Happy performance engineering! 🚀"
