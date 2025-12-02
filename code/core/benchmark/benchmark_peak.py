#!/usr/bin/env python3
"""
Peak Performance Benchmark
==========================
Measures peak hardware performance metrics:
- HBM memory bandwidth
- FP4 compute TFLOPS (if available)
- FP6 compute TFLOPS (if available)
- FP8 compute TFLOPS (if available)
- FP16 compute TFLOPS
- BF16 compute TFLOPS
- L2 cache bandwidth
- Shared memory (L1-equivalent) characteristics
- GPU hardware information (SMs, cache sizes, etc.)
- NVLink bandwidth (if multi-GPU available)
- torch.compile speedup

This script captures the actual peak performance of the hardware, which is then used
as baseline targets for validation in performance_targets.py.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Ensure repository root is on sys.path for local imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Suppress CUDA capability warnings for GB10 (12.1) - PyTorch supports up to 12.0
warnings.filterwarnings("ignore", message=".*Found GPU.*which is of cuda capability.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Minimum and Maximum cuda capability supported.*", category=UserWarning)
# Suppress TF32 API deprecation warnings
warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*TensorFloat32 tensor cores.*available but not enabled.*", category=UserWarning)
# Suppress FlashAttention kernel override warnings (harmless - happens when FlashAttention is imported multiple times)
warnings.filterwarnings("ignore", message=".*Overriding a previously registered kernel.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Warning only once for all operators.*", category=UserWarning)

import torch

from core.utils.compile_utils import enable_tf32

# Configure TF32 using new API (PyTorch 2.10+)
# Enable TF32 for optimal performance on Ampere+ GPUs using the shared helper
if torch.cuda.is_available():
    enable_tf32(matmul_precision="high", cudnn_precision="tf32", set_global_precision=True)
    # Explicitly set the supported matmul precision knob to avoid legacy API warnings
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision("high")

# FAIL FAST: Transformer Engine is REQUIRED
try:
    import transformer_engine.pytorch as te
    import transformer_engine.pytorch.constants as te_constants
except ImportError as e:
    raise ImportError(
        f"Transformer Engine is REQUIRED but not available: {e}\n"
        "Install with: pip install transformer-engine"
    ) from e

TE_AVAILABLE = True
FP8_AVAILABLE = hasattr(te, 'fp8_autocast')
FP4_AVAILABLE = hasattr(te_constants, 'NVFP4_BLOCK_SCALING_SIZE')
FP6_AVAILABLE = hasattr(te_constants, 'NVFP6_BLOCK_SCALING_SIZE')

if not FP8_AVAILABLE:
    raise RuntimeError("Transformer Engine FP8 support is REQUIRED but not available")


def maybe_force_triton_arch() -> None:
    """Force Triton to use a base arch when ptxas doesn't know micro-arch suffixes."""
    if os.environ.get("TRITON_CODEGEN_ARCH"):
        return

    if not torch.cuda.is_available():
        return

    major, minor = torch.cuda.get_device_capability()
    # Older/newer ptxas builds may not recognize micro-arch suffices like sm_121a.
    if (major, minor) == (12, 1):
        os.environ["TRITON_CODEGEN_ARCH"] = "sm_121"
        os.environ.setdefault("TRITON_GLOBAL_DEVICE_ARCH", "sm_121")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1")
        _install_ptxas_wrapper()


def _install_ptxas_wrapper() -> None:
    """Place a shim earlier in PATH that strips unsupported sm_121a -> sm_121."""
    real_ptxas = shutil.which("ptxas")
    if not real_ptxas:
        return

    wrapper_dir = Path(__file__).resolve().parent / ".ptxas_shim"
    wrapper_dir.mkdir(exist_ok=True)
    wrapper_path = wrapper_dir / "ptxas"

    script = """#!/usr/bin/env bash
set -euo pipefail
real_ptxas="{real_ptxas}"
args=()
ptx_file=""
for a in "$@"; do
  args+=("${{a//sm_121a/sm_121}}")
  if [[ -z "$ptx_file" && "$a" == *.ptx ]]; then
    ptx_file="$a"
  fi
done
# Rewrite PTX target if present
if [[ -n "$ptx_file" && -f "$ptx_file" ]]; then
  sed -i 's/sm_121a/sm_121/g' "$ptx_file"
fi
exec "$real_ptxas" "${{args[@]}}"
""".format(real_ptxas=real_ptxas)
    # Write only if missing or different to avoid chmod on every run.
    if not wrapper_path.exists() or wrapper_path.read_text() != script:
        wrapper_path.write_text(script)
        wrapper_path.chmod(0o755)

    # Prepend shim to PATH so Triton picks it up.
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if str(wrapper_dir) not in path_parts:
        os.environ["PATH"] = str(wrapper_dir) + os.pathsep + os.environ.get("PATH", "")

    # Hint Triton to use the shim instead of its bundled ptxas copy.
    os.environ.setdefault("TRITON_PTXAS_PATH", str(wrapper_path))


def get_ptxas_info() -> dict:
    """Collect ptxas path/version for debugging compiler issues."""
    path = shutil.which("ptxas")
    info = {"path": path, "version": None}
    if path:
        try:
            # ptxas prints version on stderr.
            result = subprocess.run(
                [path, "-v"], capture_output=True, text=True, check=False
            )
            output = (result.stdout or "") + (result.stderr or "")
            first_line = output.strip().splitlines()[0] if output.strip() else None
            info["version"] = first_line
        except Exception:
            info["version"] = "<unavailable>"
    return info


def measure_hbm_bandwidth(device: torch.device = None, size_gb: float = 4.0, iterations: int = 20) -> dict:
    """Measure HBM memory bandwidth."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - cannot measure HBM bandwidth")
    
    print(f"\nMeasuring HBM memory bandwidth...")
    print(f"  Test size: {size_gb} GB")
    print(f"  Iterations: {iterations}")
    
    try:
        size_bytes = int(size_gb * 1024**3)
        size_elements = size_bytes // 4  # float32 = 4 bytes
        
        # Allocate tensors
        x = torch.randn(size_elements, device=device, dtype=torch.float32)
        y = torch.empty_like(x)
        
        # Warmup
        for _ in range(10):
            y.copy_(x)
        torch.cuda.synchronize(device)
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            y.copy_(x)
        end_event.record()
        torch.cuda.synchronize(device)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        if elapsed_ms <= 0:
            raise RuntimeError("HBM bandwidth measurement returned non-positive elapsed time")
        elapsed_s = elapsed_ms / 1000.0
        bandwidth_gbs = (size_bytes * iterations / elapsed_s) / 1e9
        bandwidth_tbs = bandwidth_gbs / 1000.0
        
        # Theoretical peak for B200 is ~8.0 TB/s
        theoretical_tbs = 8.0
        utilization = (bandwidth_tbs / theoretical_tbs * 100) if theoretical_tbs > 0 else 0.0
        
        print(f"  Result: {bandwidth_tbs:.3f} TB/s ({bandwidth_gbs:.1f} GB/s)")
        print(f"  Utilization: {utilization:.1f}%")
        
        return {
            "peak_bandwidth_tbs": bandwidth_tbs,
            "peak_bandwidth_gbs": bandwidth_gbs,
            "peak_utilization_percent": utilization,
        }
    except Exception as e:
        raise RuntimeError(f"HBM bandwidth measurement failed: {e}") from e


def measure_fp16_compute(device: torch.device = None, matrix_size: int = 8192, iterations: int = 20) -> dict:
    """Measure FP16 compute TFLOPS."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - cannot measure compute TFLOPS")
    
    print(f"\nMeasuring FP16 compute performance...")
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    print(f"  Iterations: {iterations}")
    
    try:
        # Create large matrices for tensor core utilization
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
        c = torch.empty(matrix_size, matrix_size, device=device, dtype=torch.float16)
        
        # Warmup
        for _ in range(10):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize(device)
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            torch.mm(a, b, out=c)
        end_event.record()
        torch.cuda.synchronize(device)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_s = elapsed_ms / 1000.0
        
        # FLOPs = 2 * M * N * K
        flops_per_iteration = 2 * matrix_size * matrix_size * matrix_size
        total_flops = flops_per_iteration * iterations
        tflops = total_flops / (elapsed_s * 1e12)
        
        print(f"  Result: {tflops:.2f} TFLOPS")
        
        return {
            "peak_tflops": tflops,
            "matrix_size": matrix_size,
        }
    except Exception as e:
        raise RuntimeError(f"Compute measurement failed: {e}") from e


def measure_bf16_compute(device: torch.device = None, matrix_size: int = 8192, iterations: int = 20) -> dict:
    """Measure BF16 compute TFLOPS."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - cannot measure compute TFLOPS")
    
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
    if not is_bf16_supported():
        return {"peak_tflops": None, "error": "BF16 not supported on this GPU"}
    
    print(f"\nMeasuring BF16 compute performance...")
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    print(f"  Iterations: {iterations}")
    
    try:
        dtype = torch.bfloat16
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        c = torch.empty(matrix_size, matrix_size, device=device, dtype=dtype)
        
        for _ in range(10):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize(device)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            torch.mm(a, b, out=c)
        end_event.record()
        torch.cuda.synchronize(device)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_s = elapsed_ms / 1000.0
        
        flops_per_iteration = 2 * matrix_size * matrix_size * matrix_size
        total_flops = flops_per_iteration * iterations
        tflops = total_flops / (elapsed_s * 1e12)
        
        print(f"  Result: {tflops:.2f} TFLOPS")
        
        return {
            "peak_tflops": tflops,
            "matrix_size": matrix_size,
        }
    except Exception as e:
        raise RuntimeError(f"Compute measurement failed: {e}") from e


def measure_fp4_compute(device: torch.device = None, matrix_size: int = 8192, iterations: int = 20) -> dict:
    """Measure FP4 compute TFLOPS using Transformer Engine NVFP4 (if available)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - cannot measure compute TFLOPS")
    
    if not FP4_AVAILABLE:
        raise RuntimeError("Transformer Engine FP4 (NVFP4) is REQUIRED but not available")
    
    print(f"\nMeasuring FP4 compute performance...")
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    print(f"  Iterations: {iterations}")
    
    try:
        # Use Transformer Engine Linear layer (supports FP4 via fp8_autocast with FP4 recipe)
        from transformer_engine.pytorch import Linear as TELinear
        
        in_features = matrix_size
        out_features = matrix_size
        
        # Input tensor - create outside autocast
        x = torch.randn(1024, in_features, device=device, dtype=torch.float16)
        
        # Create TE linear layer - initialize on device first
        fp4_linear = TELinear(
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.float16,
        ).to(device)
        
        # Create FP4 recipe using DelayedScaling with NVFP4BlockScaling
        from transformer_engine.common.recipe import DelayedScaling, NVFP4BlockScaling
        
        try:
            # Create FP4 recipe with NVFP4BlockScaling
            fp4_recipe = DelayedScaling(
                float8_block_scaling=NVFP4BlockScaling()
            )
            print(f"  Created FP4 recipe: {type(fp4_recipe).__name__} with NVFP4BlockScaling")
        except (AttributeError, TypeError, ImportError) as e:
            error_msg = f"Failed to create FP4 recipe: {e}. FP4 measurement requires NVFP4BlockScaling support."
            print(f"  ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e
        
        # Initialize the layer INSIDE autocast context with FP4 recipe
        # This ensures parameters are properly initialized for FP4
        with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
            # Do initial forward pass to initialize FP4 weights properly
            _ = fp4_linear(x)
        
        torch.cuda.synchronize(device)
        
        # Warmup - FP4 uses fp8_autocast API with FP4 recipe
        with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
            for _ in range(10):
                _ = fp4_linear(x)
        torch.cuda.synchronize(device)
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
            for _ in range(iterations):
                _ = fp4_linear(x)
        end_event.record()
        torch.cuda.synchronize(device)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_s = elapsed_ms / 1000.0
        
        # FLOPs = 2 * batch_size * M * N
        batch_size = x.shape[0]
        flops_per_iteration = 2 * batch_size * in_features * out_features
        total_flops = flops_per_iteration * iterations
        tflops = total_flops / (elapsed_s * 1e12)
        
        print(f"  Result: {tflops:.2f} TFLOPS")
        
        return {
            "peak_tflops": tflops,
            "matrix_size": matrix_size,
        }
    except Exception as e:
        raise RuntimeError(f"FP4 measurement failed: {e}") from e


def measure_fp6_compute(device: torch.device = None, matrix_size: int = 8192, iterations: int = 20) -> dict:
    """Measure FP6 compute TFLOPS using Transformer Engine NVFP6 (if available)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - cannot measure compute TFLOPS")
    
    # FP6 is optional - skip if not available
    if not FP6_AVAILABLE:
        return {"peak_tflops": None, "error": "Transformer Engine FP6 (NVFP6) not available (optional)"}
    
    print(f"\nMeasuring FP6 compute performance...")
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    print(f"  Iterations: {iterations}")
    
    try:
        from transformer_engine.pytorch import Linear as TELinear
        
        in_features = matrix_size
        out_features = matrix_size
        x = torch.randn(1024, in_features, device=device, dtype=torch.float16)
        fp6_linear = TELinear(
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.float16,
        ).to(device)
        
        # Create FP6 recipe - check if NVFP6BlockScaling is available
        from transformer_engine.common.recipe import DelayedScaling
        
        # Check if NVFP6BlockScaling exists (may not be available in all TE versions)
        try:
            from transformer_engine.common.recipe import NVFP6BlockScaling
            fp6_recipe = DelayedScaling(
                float8_block_scaling=NVFP6BlockScaling()
            )
            print(f"  Created FP6 recipe: {type(fp6_recipe).__name__} with NVFP6BlockScaling")
        except (AttributeError, ImportError) as e:
            error_msg = f"FP6 (NVFP6) is not available in this Transformer Engine version: {e}. NVFP6BlockScaling not found."
            print(f"  Skipping FP6: {error_msg}")
            return {"peak_tflops": None, "error": error_msg}
        
        def _run(iterations_to_run: int) -> None:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp6_recipe):
                for _ in range(iterations_to_run):
                    _ = fp6_linear(x)
        
        _run(10)
        torch.cuda.synchronize(device)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        _run(iterations)
        end_event.record()
        torch.cuda.synchronize(device)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_s = elapsed_ms / 1000.0
        
        batch_size = x.shape[0]
        flops_per_iteration = 2 * batch_size * in_features * out_features
        total_flops = flops_per_iteration * iterations
        tflops = total_flops / (elapsed_s * 1e12)
        
        print(f"  Result: {tflops:.2f} TFLOPS")
        
        return {
            "peak_tflops": tflops,
            "matrix_size": matrix_size,
        }
    except Exception as e:
        raise RuntimeError(f"FP6 measurement failed: {e}") from e


def measure_fp8_compute(device: torch.device = None, matrix_size: int = 8192, iterations: int = 20) -> dict:
    """Measure FP8 compute TFLOPS using Transformer Engine (if available)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - cannot measure compute TFLOPS")
    
    if not FP8_AVAILABLE:
        raise RuntimeError("Transformer Engine FP8 is REQUIRED but not available")
    
    print(f"\nMeasuring FP8 compute performance...")
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    print(f"  Iterations: {iterations}")
    
    try:
        # Use Transformer Engine FP8 linear layer
        from transformer_engine.pytorch import Linear as Fp8Linear
        
        in_features = matrix_size
        out_features = matrix_size
        
        # Create FP8 linear layer
        fp8_linear = Fp8Linear(
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.float16,
        ).to(device)
        
        # Input tensor
        x = torch.randn(1024, in_features, device=device, dtype=torch.float16)
        
        # Warmup
        with te.fp8_autocast():
            for _ in range(10):
                _ = fp8_linear(x)
        torch.cuda.synchronize(device)
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with te.fp8_autocast():
            for _ in range(iterations):
                _ = fp8_linear(x)
        end_event.record()
        torch.cuda.synchronize(device)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_s = elapsed_ms / 1000.0
        
        # FLOPs = 2 * batch_size * M * N
        batch_size = x.shape[0]
        flops_per_iteration = 2 * batch_size * in_features * out_features
        total_flops = flops_per_iteration * iterations
        tflops = total_flops / (elapsed_s * 1e12)
        
        print(f"  Result: {tflops:.2f} TFLOPS")
        
        return {
            "peak_tflops": tflops,
            "matrix_size": matrix_size,
        }
    except Exception as e:
        raise RuntimeError(f"Compute measurement failed: {e}") from e


def measure_l2_cache_bandwidth(device: torch.device = None, size_mb: float = 50.0, iterations: int = 20) -> dict:
    """Measure L2 cache bandwidth by using data that fits in L2 cache."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"peak_bandwidth_gbs": None, "error": "CUDA not available"}
    
    # Get L2 cache size from device properties
    props = torch.cuda.get_device_properties(device)
    l2_cache_size_bytes = getattr(props, 'l2_cache_size', 0)
    l2_cache_size_mb = l2_cache_size_bytes / (1024**2)
    
    # Use a size that fits comfortably in L2 cache (75% of L2 size)
    if l2_cache_size_bytes > 0:
        test_size_mb = min(size_mb, l2_cache_size_mb * 0.75)
    else:
        test_size_mb = size_mb  # Fallback if L2 size unknown
    
    print(f"\nMeasuring L2 cache bandwidth...")
    print(f"  L2 cache size: {l2_cache_size_mb:.1f} MB" if l2_cache_size_bytes > 0 else "  L2 cache size: Unknown")
    print(f"  Test size: {test_size_mb:.1f} MB")
    print(f"  Iterations: {iterations}")
    
    try:
        size_bytes = int(test_size_mb * 1024**2)
        size_elements = size_bytes // 4  # float32 = 4 bytes
        
        # Allocate tensors that should fit in L2 cache
        x = torch.randn(size_elements, device=device, dtype=torch.float32)
        y = torch.empty_like(x)
        
        # Warmup to ensure data is in cache
        for _ in range(20):
            y.copy_(x)
        torch.cuda.synchronize(device)
        
        # Benchmark - small operations should hit L2 cache
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            y.copy_(x)
        end_event.record()
        torch.cuda.synchronize(device)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        if elapsed_ms <= 0:
            raise RuntimeError("L2 cache bandwidth measurement returned non-positive elapsed time")
        elapsed_s = elapsed_ms / 1000.0
        bandwidth_gbs = (size_bytes * iterations / elapsed_s) / 1e9
        
        print(f"  Result: {bandwidth_gbs:.1f} GB/s")
        
        return {
            "peak_bandwidth_gbs": bandwidth_gbs,
            "l2_cache_size_mb": l2_cache_size_mb,
            "test_size_mb": test_size_mb,
        }
    except Exception as e:
        raise RuntimeError(f"L2 cache bandwidth measurement failed: {e}") from e


def measure_shared_memory_info(device: torch.device = None) -> dict:
    """Capture shared memory characteristics (acts as L1 cache equivalent)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    print(f"\nCapturing shared memory (L1-equivalent) information...")
    
    try:
        props = torch.cuda.get_device_properties(device)
        
        shared_mem_per_block = getattr(props, 'shared_memory_per_block', 0) / 1024  # KB
        shared_mem_per_sm = getattr(props, 'shared_memory_per_multiprocessor', 0) / 1024  # KB
        num_sms = props.multi_processor_count
        
        print(f"  Shared memory per block: {shared_mem_per_block:.1f} KB")
        print(f"  Shared memory per SM: {shared_mem_per_sm:.1f} KB")
        print(f"  Total SMs: {num_sms}")
        print(f"  Total shared memory: {shared_mem_per_sm * num_sms / 1024:.1f} MB")
        
        return {
            "shared_memory_per_block_kb": shared_mem_per_block,
            "shared_memory_per_sm_kb": shared_mem_per_sm,
            "num_sms": num_sms,
            "total_shared_memory_mb": shared_mem_per_sm * num_sms / 1024,
        }
    except Exception as e:
        raise RuntimeError(f"Measurement failed: {e}") from e


def measure_nvlink_bandwidth(device: torch.device = None, iterations: int = 20) -> dict:
    """Measure NVLink bandwidth between GPUs (if multi-GPU available)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"peak_bandwidth_gbs": None, "error": "CUDA not available"}
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        return {"peak_bandwidth_gbs": None, "error": "Multi-GPU not available (requires 2+ GPUs)"}
    
    print(f"\nMeasuring NVLink bandwidth...")
    print(f"  GPU count: {gpu_count}")
    print(f"  Testing GPU 0 -> GPU 1")
    print(f"  Iterations: {iterations}")
    
    try:
        # Use ~1GB for bandwidth test
        size_mb = 1024
        size = size_mb * 1024 * 1024 // 4  # float32
        
        # Create tensors on different GPUs
        with torch.cuda.device(0):
            src = torch.randn(size, device='cuda:0')
        
        with torch.cuda.device(1):
            dst = torch.empty(size, device='cuda:1')
        
        # Warmup
        for _ in range(10):
            dst.copy_(src)
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            dst.copy_(src)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        bandwidth_gbs = (size_mb / 1024) / (elapsed_ms / 1000) * iterations
        
        print(f"  Result: {bandwidth_gbs:.2f} GB/s")
        
        # Clean up
        del src, dst
        torch.cuda.empty_cache()
        
        return {
            "peak_bandwidth_gbs": bandwidth_gbs,
            "gpu_count": gpu_count,
        }
    except Exception as e:
        raise RuntimeError(f"NVLink bandwidth measurement failed: {e}") from e


def capture_gpu_hardware_info(device: torch.device = None) -> dict:
    """Capture comprehensive GPU hardware information."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    print(f"\nCapturing GPU hardware information...")
    
    try:
        props = torch.cuda.get_device_properties(device)
        
        info = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": props.total_memory / (1024**3),
            "num_sms": props.multi_processor_count,
            "max_threads_per_block": getattr(props, 'max_threads_per_block', 1024),
            "max_threads_per_sm": props.max_threads_per_multi_processor,
            "warp_size": props.warp_size,
            "l2_cache_size_kb": getattr(props, 'l2_cache_size', 0) / 1024,
            "shared_memory_per_block_kb": getattr(props, 'shared_memory_per_block', 0) / 1024,
            "shared_memory_per_sm_kb": getattr(props, 'shared_memory_per_multiprocessor', 0) / 1024,
            "registers_per_block": getattr(props, 'registers_per_block', 0),
            "registers_per_sm": getattr(props, 'registers_per_multiprocessor', 0),
        }
        
        print(f"  GPU: {info['name']}")
        print(f"  Compute Capability: {info['compute_capability']}")
        print(f"  Total Memory: {info['total_memory_gb']:.2f} GB")
        print(f"  SMs: {info['num_sms']}")
        print(f"  L2 Cache: {info['l2_cache_size_kb']:.1f} KB")
        
        return info
    except Exception as e:
        raise RuntimeError(f"Measurement failed: {e}") from e


def measure_torch_compile_speedup(
    device: torch.device = None,
    matrix_size: int = 4096,
    iterations: int = 20,
    matrix_sizes: list = None,
) -> dict:
    """Measure torch.compile speedup across modes/settings and multiple matrix sizes."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"speedup": None, "error": "CUDA not available"}
  
    if matrix_sizes is None:
        matrix_sizes = [matrix_size]
    print(f"\nMeasuring torch.compile speedup...")
    print(f"  Matrix sizes: {', '.join(str(s) for s in matrix_sizes)}")
    print(f"  Iterations (per size): {iterations}")
    
    try:
        # Define a function that will benefit from compilation (more complex)
        def compute_fn(x, y):
            # Chain operations to benefit from fusion
            z = torch.mm(x, y)
            z = torch.relu(z)
            z = torch.mm(z, y.t())
            return z
        
        compile_configs = [
            {"name": "default", "kwargs": {"mode": "default"}},
            {"name": "default_fullgraph", "kwargs": {"mode": "default", "fullgraph": True}},
            {"name": "default_static", "kwargs": {"mode": "default", "fullgraph": True, "dynamic": False}},
            {"name": "reduce-overhead", "kwargs": {"mode": "reduce-overhead"}},
            {"name": "max-autotune", "kwargs": {"mode": "max-autotune"}},
        ]

        per_size_results = []
        for size in matrix_sizes:
            # Create test tensors
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)

            # Warmup eager (more iterations for stable timing)
            warmup_eager = min(10, iterations)
            for _ in range(warmup_eager):
                _ = compute_fn(a, b)
            torch.cuda.synchronize(device)

            # Benchmark eager
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(iterations):
                _ = compute_fn(a, b)
            end_event.record()
            torch.cuda.synchronize(device)
            eager_time_ms = start_event.elapsed_time(end_event)

            cfg_results = []
            for cfg in compile_configs:
                name = cfg["name"]
                kwargs = cfg["kwargs"]
                try:
                    compiled_fn = torch.compile(compute_fn, **kwargs)

                    # Warmup compiled (compilation happens here)
                    warmup_compiled = min(10, iterations)
                    for _ in range(warmup_compiled):
                        _ = compiled_fn(a, b)
                    torch.cuda.synchronize(device)

                    # Ensure compilation is complete
                    for _ in range(5):
                        _ = compiled_fn(a, b)
                    torch.cuda.synchronize(device)

                    # Benchmark compiled
                    start_event.record()
                    for _ in range(iterations):
                        _ = compiled_fn(a, b)
                    end_event.record()
                    torch.cuda.synchronize(device)
                    compiled_time_ms = start_event.elapsed_time(end_event)
                    speedup = eager_time_ms / compiled_time_ms if compiled_time_ms > 0 else 1.0
                    cfg_results.append(
                        {
                            "mode": name,
                            "compiled_time_ms": compiled_time_ms / iterations,
                            "speedup": speedup,
                            "compile_kwargs": kwargs,
                        }
                    )
                    print(f"  [size={size} | {name}] compiled time: {compiled_time_ms/iterations:.3f} ms/iter | speedup: {speedup:.2f}x")
                except Exception as e:
                    print(f"  [size={size} | {name}] compile failed: {e}")

            if not cfg_results:
                per_size_results.append(
                    {
                        "matrix_size": size,
                        "error": "No torch.compile modes succeeded",
                        "eager_time_ms": eager_time_ms / iterations,
                    }
                )
                continue

            best_cfg = max(cfg_results, key=lambda r: r["speedup"])
            per_size_results.append(
                {
                    "matrix_size": size,
                    "eager_time_ms": eager_time_ms / iterations,
                    "best_mode": best_cfg["mode"],
                    "best_speedup": best_cfg["speedup"],
                    "best_compiled_time_ms": best_cfg["compiled_time_ms"],
                    "modes_tried": cfg_results,
                }
            )
            print(f"  Best for size {size}: {best_cfg['mode']} | speedup: {best_cfg['speedup']:.2f}x")

        # Pick global best across sizes
        successful = [r for r in per_size_results if "best_speedup" in r]
        if not successful:
            raise RuntimeError("No torch.compile modes succeeded for any matrix size")

        global_best = max(successful, key=lambda r: r["best_speedup"])
        return {
            "speedup": global_best["best_speedup"],
            "eager_time_ms": global_best["eager_time_ms"],
            "compiled_time_ms": global_best["best_compiled_time_ms"],
            "mode": global_best["best_mode"],
            "matrix_size": global_best["matrix_size"],
            "per_size": per_size_results,
        }
    except Exception as e:
        raise RuntimeError(f"torch.compile speedup measurement failed: {e}") from e


def run_peak_benchmarks(output_dir: Path = None) -> dict:
    """Run peak performance benchmarks."""
    if output_dir is None:
        output_dir = Path.cwd()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Cannot run peak performance benchmarks.")
        return {"error": "CUDA not available"}

    maybe_force_triton_arch()
    device = torch.device("cuda")
    ptxas_info = get_ptxas_info()

    print("="*70)
    print("Peak Performance Benchmark")
    print("="*70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Transformer Engine Available: {TE_AVAILABLE}")
    print(f"FP4 Available: {FP4_AVAILABLE}")
    print(f"FP6 Available: {FP6_AVAILABLE}")
    print(f"FP8 Available: {FP8_AVAILABLE}")
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    print(f"BF16 Supported: {bf16_supported}")
    print(f"TRITON_CODEGEN_ARCH: {os.environ.get('TRITON_CODEGEN_ARCH')} (auto-set for cc 12.1 to avoid sm_121a)")
    print(f"ptxas: {ptxas_info['path']} | version: {ptxas_info['version']}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "gpu_name": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "te_available": TE_AVAILABLE,
        "fp4_available": FP4_AVAILABLE,
        "fp6_available": FP6_AVAILABLE,
        "fp8_available": FP8_AVAILABLE,
        "bf16_supported": bf16_supported,
    }
    
    # Use 20 iterations for all measurements
    iterations = 20
    
    # Measure HBM memory bandwidth
    results["hbm"] = measure_hbm_bandwidth(device, iterations=iterations)
    
    # Measure FP4 compute (REQUIRED - no fallback)
    results["fp4_compute"] = measure_fp4_compute(device, iterations=iterations)
    
    # Measure FP6 compute (OPTIONAL - skip if not available)
    try:
        results["fp6_compute"] = measure_fp6_compute(device, iterations=iterations)
    except RuntimeError as e:
        if "not available" in str(e).lower() or "NVFP6" in str(e):
            results["fp6_compute"] = {"peak_tflops": None, "error": str(e)}
        else:
            raise  # Re-raise if it's a real error, not just "not available"
    
    # Measure FP8 compute (REQUIRED - no fallback)
    results["fp8_compute"] = measure_fp8_compute(device, iterations=iterations)
    
    # Measure FP16 compute
    results["fp16_compute"] = measure_fp16_compute(device, iterations=iterations)
    
    # Measure BF16 compute (if supported)
    results["bf16_compute"] = measure_bf16_compute(device, iterations=iterations)
    
    # Measure L2 cache bandwidth
    results["l2_cache"] = measure_l2_cache_bandwidth(device, iterations=iterations)
    
    # Capture shared memory info (L1-equivalent)
    results["shared_memory"] = measure_shared_memory_info(device)
    
    # Capture comprehensive GPU hardware info
    results["gpu_hardware"] = capture_gpu_hardware_info(device)
    
    # Measure NVLink bandwidth (if multi-GPU)
    results["nvlink"] = measure_nvlink_bandwidth(device, iterations=iterations)
    
    # Measure torch.compile speedup (sweep modes/settings and sizes)
    results["torch_compile"] = measure_torch_compile_speedup(
        device,
        iterations=iterations,
        matrix_sizes=[4096, 8192],
    )
    
    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    if results["hbm"].get("peak_bandwidth_tbs"):
        print(f"HBM Memory Bandwidth: {results['hbm']['peak_bandwidth_tbs']:.3f} TB/s")
    
    # REQUIRED measurements - if they failed, exceptions were already raised
    if results["fp4_compute"].get("peak_tflops"):
        print(f"FP4 Compute: {results['fp4_compute']['peak_tflops']:.2f} TFLOPS")
    else:
        raise RuntimeError("FP4 compute measurement failed - this should not happen (exception should have been raised)")
    
    # FP6 is optional - skip if not available
    if results["fp6_compute"].get("peak_tflops"):
        print(f"FP6 Compute: {results['fp6_compute']['peak_tflops']:.2f} TFLOPS")
    elif results["fp6_compute"].get("error"):
        print(f"FP6 Compute: Not available ({results['fp6_compute']['error']})")
    else:
        print("FP6 Compute: Not available (optional)")
    
    if results["fp8_compute"].get("peak_tflops"):
        print(f"FP8 Compute: {results['fp8_compute']['peak_tflops']:.2f} TFLOPS")
    else:
        raise RuntimeError("FP8 compute measurement failed - this should not happen (exception should have been raised)")
    
    if results["fp16_compute"].get("peak_tflops"):
        print(f"FP16 Compute: {results['fp16_compute']['peak_tflops']:.2f} TFLOPS")
    
    if results["bf16_compute"].get("peak_tflops"):
        print(f"BF16 Compute: {results['bf16_compute']['peak_tflops']:.2f} TFLOPS")
    elif results["bf16_compute"].get("error"):
        print(f"BF16 Compute: Not available ({results['bf16_compute']['error']})")
    
    if results["l2_cache"].get("peak_bandwidth_gbs"):
        print(f"L2 Cache Bandwidth: {results['l2_cache']['peak_bandwidth_gbs']:.1f} GB/s")
    
    if results["nvlink"].get("peak_bandwidth_gbs"):
        print(f"NVLink Bandwidth (GPU 0->1): {results['nvlink']['peak_bandwidth_gbs']:.2f} GB/s")
    elif results["nvlink"].get("error"):
        print(f"NVLink: Not available ({results['nvlink']['error']})")
    
    if results["torch_compile"].get("speedup"):
        print(f"torch.compile Speedup: {results['torch_compile']['speedup']:.2f}x")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_peak_results_{timestamp}.json"
    output_path = output_dir / filename
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Measure peak hardware performance")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: current directory)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        results = run_peak_benchmarks(output_dir)
        if "error" in results and results["error"] == "CUDA not available":
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
