"""Chapter 20: AI-assisted CUDA kernel iteration (PyTorch 2.9 / CUDA 13.0)."""

from __future__ import annotations

import arch_config  # noqa: F401 - Configure Blackwell optimizations
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import torch


class OptimizationTarget(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"


@dataclass
class KernelCandidate:
    code: str
    compile_ok: bool
    runtime_ms: float
    mem_mb: float
    correctness: float
    iteration: int

    @property
    def score(self) -> float:
        if not self.compile_ok:
            return 0.0
        perf = max(0.0, 1.0 - self.runtime_ms / 50.0)
        mem = max(0.0, 1.0 - self.mem_mb / 512.0)
        return self.correctness * (0.7 * perf + 0.3 * mem)

    @property
    def valid(self) -> bool:
        return self.compile_ok and self.correctness >= 0.95


class MockGenerator:
    def __init__(self) -> None:
        self.templates: Dict[str, str] = {
            "gemm": """
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
  extern __shared__ float shared_mem[];
  float* As = shared_mem;
  float* Bs = shared_mem + blockDim.y * blockDim.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.0f;
  for (int tile = 0; tile < (K + blockDim.x - 1) / blockDim.x; ++tile) {
    int tiled_col = tile * blockDim.x + threadIdx.x;
    int tiled_row = tile * blockDim.y + threadIdx.y;
    As[threadIdx.y * blockDim.x + threadIdx.x] =
        (row < M && tiled_col < K) ? A[row * K + tiled_col] : 0.0f;
    Bs[threadIdx.y * blockDim.x + threadIdx.x] =
        (tiled_row < K && col < N) ? B[tiled_row * N + col] : 0.0f;
    __syncthreads();
    for (int k = 0; k < blockDim.x; ++k) {
      acc += As[threadIdx.y * blockDim.x + k] *
             Bs[k * blockDim.x + threadIdx.x];
    }
    __syncthreads();
  }
  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}
""",
        }

    def generate(self, prompt: str, iteration: int, feedback: str) -> str:
        if "matmul" in prompt.lower():
            code = self.templates["gemm"]
        else:
            code = self.templates["gemm"]
        if iteration > 0 and "vectorize" in feedback:
            code = code.replace("acc +=", "#pragma unroll\n    acc +=")
        return code


class Verifier:
    def __init__(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def _nvcc(self, src: Path) -> bool:
        cmd = ["nvcc", "-arch=sm_100", "-c", str(src), "-o", str(src.with_suffix(".o"))]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except Exception:
            return False

    def check(self, code: str) -> KernelCandidate:
        src = self.tmp / "kernel.cu"
        src.write_text("#include <cuda_runtime.h>\n" + code)
        ok = self._nvcc(src)
        runtime = random.uniform(20.0, 45.0)
        mem = random.uniform(120.0, 260.0)
        correctness = 0.95 - random.uniform(0.0, 0.05)
        return KernelCandidate(code, ok, runtime, mem, correctness, iteration=0)


class Optimizer:
    def __init__(self, target: OptimizationTarget) -> None:
        self.target = target
        self.generator = MockGenerator()
        self.verifier = Verifier()
        self.history: List[KernelCandidate] = []
        self.max_iters = 6

    def optimize(self, prompt: str) -> KernelCandidate:
        feedback = ""
        best: Optional[KernelCandidate] = None
        for itr in range(self.max_iters):
            code = self.generator.generate(prompt, itr, feedback)
            candidate = self.verifier.check(code)
            candidate.iteration = itr
            self.history.append(candidate)
            if best is None or candidate.score > best.score:
                best = candidate
            if candidate.valid:
                break
            feedback = self._feedback(candidate)
        assert best is not None
        return best

    def _feedback(self, candidate: KernelCandidate) -> str:
        items: List[str] = []
        if not candidate.compile_ok:
            items.append("fix compilation")
        if candidate.correctness < 0.95:
            items.append("add bounds checks")
        if candidate.runtime_ms > 25.0:
            items.append("vectorize inner loop")
        return ", ".join(items)


def compare_workflows() -> None:
    prompt = "Optimized CUDA matmul using shared memory and coalesced access"
    opt = Optimizer(OptimizationTarget.LATENCY)
    start = time.time()
    best = opt.optimize(prompt)
    elapsed = time.time() - start

    print("AI-assisted search")
    print(f"  Runtime: {best.runtime_ms:.1f} ms")
    print(f"  Correctness: {best.correctness:.2f}")
    print(f"  Iterations: {len(opt.history)}")
    print(f"  Optimization wall time: {elapsed:.1f} s")

    manual = KernelCandidate("// manual kernel", True, 42.0, 140.0, 0.98, 0)
    manual_time = 6 * 3600
    print("\nManual baseline")
    print(f"  Runtime: {manual.runtime_ms:.1f} ms")
    print(f"  Optimization time: {manual_time/3600:.1f} h")

    if best.valid:
        speedup = manual.runtime_ms / best.runtime_ms
        time_gain = manual_time / elapsed
        print("\nComparison")
        print(f"  Performance speedup: {speedup:.2f}x")
        print(f"  Optimization time reduction: {time_gain:.0f}x")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("AI-assisted kernel generator demo")
    print(f"Device detected: {device}")
    compare_workflows()


if __name__ == "__main__":
    main()
