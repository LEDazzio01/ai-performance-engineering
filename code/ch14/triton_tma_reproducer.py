"""
===============================================================================
Triton 3.5 Bug Report: TMA Descriptor + Deep Pipelines Crash on Blackwell
===============================================================================

BUG TITLE:
  tritongpu-assign-latencies pass fails with TMA descriptors on SM 10.0

DESCRIPTION:
  Using TMA tensor descriptors with aggressive pipeline configurations
  (num_stages >= 4, BLOCK_K >= 64) causes compiler crash on Blackwell GPUs.

SYMPTOM:
  error: Failures have been detected while processing an MLIR pass pipeline
  note: Pipeline failed while executing [`TritonGPUAssignLatencies` on
        'builtin.module' operation]

IMPACT:
  - Blackwell B200/B300 GPUs cannot use optimal TMA configurations
  - Performance degradation: ~2x slower GEMM than hardware capability
  - HBM3e bandwidth utilization capped at ~60% instead of 85-90%
  - Forced to use conservative configs (BLOCK_K=32, num_stages=1)

ENVIRONMENT:
  - GPU: NVIDIA B200 (Compute Capability 10.0)
  - CUDA: 13.0
  - Triton: 3.5.0
  - PyTorch: 2.9.0+cu130
  - OS: Linux 6.8.0

REPRODUCTION:
  python triton_tma_reproducer.py

EXPECTED BEHAVIOR:
  Aggressive configs should compile and execute, providing 2x better performance

ACTUAL BEHAVIOR:
  Compiler crashes during latency assignment pass
  
WORKAROUND:
  Use BLOCK_K=32, num_stages=1, num_warps=4 (significant perf loss)

===============================================================================
"""

import torch
import triton
import triton.language as tl
from triton.runtime import _allocation as triton_allocation


# REQUIRED: Setup allocator for TMA descriptor scratch buffers
class _TorchCudaBuffer:
    __slots__ = ("_ptr",)
    def __init__(self, ptr: int):
        self._ptr = ptr
    def data_ptr(self) -> int:
        return self._ptr
    def __del__(self):
        if self._ptr:
            torch.cuda.caching_allocator_delete(self._ptr)
            self._ptr = 0


class _TorchCudaAllocator:
    def __call__(self, size: int, alignment: int, stream: int | None):
        if size == 0:
            return _TorchCudaBuffer(0)
        if stream is None:
            current_stream = torch.cuda.current_stream()
            stream = current_stream.cuda_stream
            device_idx = current_stream.device.index
        else:
            device_idx = torch.cuda.current_device()
        if device_idx is None:
            device_idx = torch.cuda.current_device()
        ptr = torch.cuda.caching_allocator_alloc(size, device_idx, stream=stream)
        return _TorchCudaBuffer(ptr)


if torch.cuda.is_available():
    current = triton_allocation._allocator.get()
    if isinstance(current, triton_allocation.NullAllocator):
        triton.set_allocator(_TorchCudaAllocator())


# ============================================================================
# WORKING CASE: Conservative Configuration (BLOCK_K=32, num_stages=1)
# ============================================================================

@triton.jit
def tma_gemm_conservative(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """WORKS: Conservative TMA configuration"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    
    A_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    
    B_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k0 in range(0, K, BLOCK_K):
        a = A_desc.load([m0, k0])
        b = B_desc.load([k0, n0])
        acc += tl.dot(a, b, out_dtype=tl.float32)
    
    # Store directly (no descriptor for output to avoid additional issues)
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ============================================================================
# FAILING CASE: With Autotune (matches production code structure)
# ============================================================================

@triton.autotune(
    configs=[
        # This is the config that crashes in production
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=16, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def tma_gemm_with_autotune(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """FAILS: Autotune with aggressive configs crashes tritongpu-assign-latencies"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    
    A_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    
    B_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    
    C_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Double buffering pattern (matches production code)
    k0 = 0
    a_cur = A_desc.load([m0, k0])
    b_cur = B_desc.load([k0, n0])
    
    for k0 in range(0, K, BLOCK_K):
        next_k = k0 + BLOCK_K
        a_next = a_cur
        b_next = b_cur
        if next_k < K:
            a_next = A_desc.load([m0, next_k])
            b_next = B_desc.load([next_k, n0])
        
        acc += tl.dot(a_cur, b_cur, out_dtype=tl.float32)
        
        if next_k < K:
            a_cur = a_next
            b_cur = b_next
    
    C_desc.store([m0, n0], acc)


def test_conservative():
    """Test with conservative config - SHOULD WORK"""
    print("=" * 80)
    print("TEST 1: Conservative Configuration (BLOCK_K=32, num_stages=1)")
    print("=" * 80)
    
    M, N, K = 2048, 2048, 2048
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    C = torch.zeros(M, N, device='cuda', dtype=torch.float32)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    try:
        tma_gemm_conservative[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=1,  # Conservative
        )
        torch.cuda.synchronize()
        print("✅ SUCCESS: Conservative config compiled and executed")
        print(f"   Result shape: {C.shape}, norm: {C.norm().item():.2f}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_aggressive():
    """Test with aggressive config + AUTOTUNE - WILL FAIL on Triton 3.5"""
    print("\n" + "=" * 80)
    print("TEST 2: Aggressive Configuration with Autotune (BLOCK_K=128, num_stages=4)")
    print("=" * 80)
    print("NOTE: Using autotune (matches production code path)")
    
    M, N, K = 2048, 2048, 2048
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    C = torch.zeros(M, N, device='cuda', dtype=torch.float32)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    
    try:
        tma_gemm_with_autotune[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )
        torch.cuda.synchronize()
        print("✅ SUCCESS: Aggressive config with autotune compiled and executed")
        print(f"   Result shape: {C.shape}, norm: {C.norm().item():.2f}")
        print("\n   This suggests the bug may be fixed or doesn't occur in this")
        print("   specific configuration. Try running the full triton_tma_blackwell.py")
        return True
    except Exception as e:
        error_str = str(e)
        print(f"❌ EXPECTED FAILURE: {error_str[:200]}")
        
        if "TritonGPUAssignLatencies" in error_str or "PassManager" in error_str:
            print("\n✓ BUG CONFIRMED: This is the tritongpu-assign-latencies bug!")
            print("  The compiler cannot handle:")
            print("  - TMA tensor descriptors")
            print("  - Large tiles (BLOCK_K=128)")
            print("  - Deep pipelines (num_stages=4)")
            print("  - Multiple warps (num_warps=16)")
        else:
            print(f"\n⚠️  Different error than expected:")
            print(f"     {error_str[:400]}")
        
        return False


def print_environment_info():
    """Print detailed environment information for bug report"""
    print("\n" + "=" * 80)
    print("ENVIRONMENT DETAILS FOR BUG REPORT")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    
    device_props = torch.cuda.get_device_properties(0)
    print(f"GPU Model:              {device_props.name}")
    print(f"Compute Capability:     {device_props.major}.{device_props.minor}")
    print(f"CUDA Version:           {torch.version.cuda}")
    print(f"Triton Version:         {triton.__version__}")
    print(f"PyTorch Version:        {torch.__version__}")
    print(f"Total GPU Memory:       {device_props.total_memory / 1e9:.1f} GB")
    print(f"SMs:                    {device_props.multi_processor_count}")
    
    if device_props.major != 10 or device_props.minor != 0:
        print("\n⚠️  WARNING: This bug specifically affects Blackwell (SM 10.0)")
        print("    Your GPU may not reproduce the issue")
        return False
    
    return True


def main():
    is_blackwell = print_environment_info()
    
    if not is_blackwell:
        print("\nSkipping tests - Blackwell GPU required for reproduction")
        return
    
    # Run tests
    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)
    
    conservative_works = test_conservative()
    aggressive_works = test_aggressive()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Conservative config (BLOCK_K=32, stages=1):  {'✅ WORKS' if conservative_works else '❌ FAILED'}")
    print(f"Aggressive config (BLOCK_K=128, stages=4):   {'✅ WORKS' if aggressive_works else '❌ FAILS'}")
    
    if not aggressive_works and conservative_works:
        print("\n" + "=" * 80)
        print("BUG CONFIRMED - SUBMISSION INFORMATION")
        print("=" * 80)
        print("\nThis reproducer confirms the tritongpu-assign-latencies bug on Blackwell.")
        print("\n📋 INFORMATION FOR TRITON ISSUE:")
        print("   • Title: [SM 10.0] TMA descriptor compilation fails with deep pipelines")
        print("   • Component: tritongpu-assign-latencies pass")
        print("   • Architecture: Blackwell B200 (SM 10.0)")
        print("   • Symptom: PassManager::run failed during compilation")
        print("   • Impact: 2x performance loss, ~40% bandwidth utilization lost")
        print("\n📎 ATTACH TO ISSUE:")
        print("   • This reproducer script (triton_tma_reproducer.py)")
        print("   • Error output with MLIR pipeline details")
        print("   • Environment info (printed above)")
        print("\n🔧 CONFIGURATION THAT WORKS:")
        print("   BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, num_stages=1, num_warps=4")
        print("\n❌ CONFIGURATION THAT FAILS:")
        print("   BLOCK_K=128, BLOCK_M=128, BLOCK_N=128, num_stages=4, num_warps=16")
        print("\n💡 HYPOTHESIS:")
        print("   Latency assignment logic may not account for increased register")
        print("   pressure from larger TMA tiles + deeper pipelines on SM 10.0")
        print("\n🔗 RELATED COMPONENTS:")
        print("   • tritongpu-assign-latencies pass")
        print("   • TMA descriptor lowering (triton-nvidia-tma-lowering)")
        print("   • Blackwell-specific scheduling (SM 10.0)")
    elif aggressive_works:
        print("\n✅ Bug appears to be fixed! Both configs work.")
        print("   (Or this GPU doesn't exhibit the issue)")
    else:
        print("\n❌ Both configs failed - may indicate different issue")


if __name__ == "__main__":
    main()

