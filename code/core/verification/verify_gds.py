#!/usr/bin/env python3
"""
GPUDirect Storage (GDS) Verification Script
===========================================

Verifies that GPUDirect Storage is properly installed and functional.

Checks:
1. nvidia-fs kernel module loaded
2. cuFile Python bindings available
3. NVMe drives present
4. GDS configuration files present
5. Basic cuFile operations work

Usage:
    python3 verify_gds.py
"""

import os
import subprocess
import sys
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def check_nvidia_fs_module():
    """Check if nvidia-fs kernel module is loaded."""
    print("Checking nvidia-fs kernel module...")
    try:
        result = subprocess.run(
            ["lsmod"], capture_output=True, text=True, check=True
        )
        if "nvidia_fs" in result.stdout:
            print("  [OK] nvidia-fs kernel module is loaded")
            return True
        else:
            print("  ERROR: nvidia-fs kernel module NOT loaded")
            print("  ℹ️  Load it with: sudo modprobe nvidia-fs")
            print("  ℹ️  Or run: sudo core/scripts/setup/load_gds_module.sh")
            return False
    except Exception as e:
        print(f"  ERROR: Error checking module: {e}")
        return False


def check_nvme_drives():
    """Check for NVMe drives."""
    print("\nChecking for NVMe drives...")
    try:
        result = subprocess.run(
            ["lsblk", "-o", "NAME,TYPE,SIZE,MOUNTPOINT,ROTA"],
            capture_output=True,
            text=True,
            check=True,
        )
        nvme_lines = [line for line in result.stdout.split("\n") if "nvme" in line.lower()]
        if nvme_lines:
            print(f"  [OK] Found {len(nvme_lines)} NVMe drive(s):")
            for line in nvme_lines[:5]:  # Show first 5
                print(f"     {line.strip()}")
            return True
        else:
            print("  WARNING: No NVMe drives found")
            print("  ℹ️  GDS requires NVMe storage for optimal performance")
            return False
    except Exception as e:
        print(f"  ERROR: Error checking NVMe drives: {e}")
        return False


def check_gds_files():
    """Check for GDS installation files."""
    print("\nChecking GDS installation...")
    gds_paths = [
        "/usr/local/cuda/gds",
        "/usr/local/cuda-13.0/gds",
        "/usr/lib/aarch64-linux-gnu/libcufile.so",
        "/usr/lib/x86_64-linux-gnu/libcufile.so",
    ]
    
    found = False
    for path in gds_paths:
        if os.path.exists(path):
            print(f"  [OK] Found: {path}")
            found = True
    
    if not found:
        print("  ERROR: GDS files not found")
        print("  ℹ️  Install with: apt install gds-tools-13-0")
        return False
    
    # Check for cufile.json config
    config_paths = [
        "/usr/local/cuda/gds/cufile.json",
        "/usr/local/cuda-13.0/gds/cufile.json",
    ]
    
    config_found = False
    for path in config_paths:
        if os.path.exists(path):
            print(f"  [OK] Config file: {path}")
            config_found = True
            break
    
    if not config_found:
        print("  WARNING: cufile.json not found (may use defaults)")
    
    return True


def check_cufile_bindings():
    """Check if cuFile Python bindings are available."""
    print("\nChecking cuFile Python bindings...")
    try:
        from cuda.bindings import cufile
        print("  [OK] cuFile Python bindings available")
        
        # Try to get version info
        try:
            import cuda
            print(f"  [OK] CUDA Python version: {cuda.__version__ if hasattr(cuda, '__version__') else 'unknown'}")
        except ImportError:
            pass  # cuda-python not installed
        
        return True
    except ImportError as e:
        print(f"  ERROR: cuFile bindings not available: {e}")
        print("  ℹ️  Install with: pip install cuda-python>=13.0")
        return False
    except OSError as e:
        if "nvidia-fs" in str(e).lower():
            print("  ERROR: cuFile bindings found but nvidia-fs module not loaded")
            print("  ℹ️  Load it with: sudo modprobe nvidia-fs")
        else:
            print(f"  ERROR: Error loading cuFile: {e}")
        return False


def check_gds_packages():
    """Check if GDS packages are installed."""
    print("\nChecking installed GDS packages...")
    try:
        result = subprocess.run(
            ["dpkg", "-l"], capture_output=True, text=True, check=True
        )
        gds_packages = [
            line for line in result.stdout.split("\n")
            if "gds" in line.lower() and line.startswith("ii")
        ]
        if gds_packages:
            print(f"  [OK] Found {len(gds_packages)} GDS package(s):")
            for pkg in gds_packages:
                # Extract package name (3rd column)
                parts = pkg.split()
                if len(parts) >= 3:
                    print(f"     {parts[1]}")
            return True
        else:
            print("  WARNING: No GDS packages found")
            return False
    except Exception as e:
        print(f"  WARNING: Could not check packages: {e}")
        return False


def test_basic_cufile():
    """Test basic cuFile operations."""
    print("\nTesting basic cuFile operations...")
    
    try:
        import torch
        from cuda.bindings import cufile
        
        print("  [OK] Imports successful")
        
        # Try to initialize cuFile
        try:
            cufile.driver_open()
            print("  [OK] cuFile driver opened successfully")
            cufile.driver_close()
            print("  [OK] cuFile driver closed successfully")
            return True
        except Exception as e:
            print(f"  ERROR: cuFile driver operations failed: {e}")
            if "nvidia-fs" in str(e).lower():
                print("  ℹ️  This usually means nvidia-fs module is not loaded")
                print("  ℹ️  Run: sudo modprobe nvidia-fs")
            return False
            
    except ImportError as e:
        print(f"  ERROR: Required libraries not available: {e}")
        return False
    except Exception as e:
        print(f"  ERROR: Error: {e}")
        return False


def main():
    """Run all GDS verification checks."""
    print_header("GPUDirect Storage (GDS) Verification")
    
    print("This script verifies GPUDirect Storage installation and functionality.\n")
    
    results = {
        "Kernel Module": check_nvidia_fs_module(),
        "NVMe Drives": check_nvme_drives(),
        "GDS Files": check_gds_files(),
        "GDS Packages": check_gds_packages(),
        "cuFile Bindings": check_cufile_bindings(),
        "Basic Operations": test_basic_cufile(),
    }
    
    # Summary
    print_header("Verification Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, result in results.items():
        status = "[OK] PASS" if result else "ERROR: FAIL"
        print(f"  {check:.<50} {status}")
    
    print(f"\n  Result: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n  🎉 All GDS checks passed! GPUDirect Storage is ready to use.")
        print(f"\n  Test it with: python3 ch05/gds_cufile_minimal.py")
        return 0
    elif results["Kernel Module"]:
        print("\n  WARNING: Some checks failed but nvidia-fs is loaded.")
        print("  You can try running GDS examples, though performance may be limited.")
        return 1
    else:
        print("\n  ERROR: Critical check failed: nvidia-fs kernel module not loaded")
        print("\n  To fix:")
        print("    1. Run: sudo modprobe nvidia-fs")
        print("    2. Or: sudo core/scripts/setup/load_gds_module.sh")
        print("    3. Re-run this script to verify")
        return 1


if __name__ == "__main__":
    sys.exit(main())
