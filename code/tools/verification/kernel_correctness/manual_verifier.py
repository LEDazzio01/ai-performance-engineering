"""Manual kernel verification using traditional testing approaches.

This module provides testing-based verification utilities that demonstrate
the inherent limitations of manual testing:

- Incomplete coverage of input space
- No formal memory safety guarantees
- No thread safety proofs
- Relies on human-identified edge cases

Use this as a baseline to compare against formal verification approaches.
"""

from __future__ import annotations

import torch
from typing import List, Tuple, Dict, Any, Callable, Optional
from dataclasses import dataclass, field


def get_dtype_tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    """Get appropriate rtol, atol for a given dtype.
    
    Returns dtype-aware tolerances:
    - FP32: tight (1e-5, 1e-5)
    - FP16: medium (1e-3, 1e-3)
    - BF16: loose (1e-2, 1e-2)
    - FP8: very loose (5e-2, 5e-2)
    - Default: (1e-4, 1e-4)
    """
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    elif dtype == torch.bfloat16:
        return 1e-2, 1e-2
    elif hasattr(torch, 'float8_e4m3fn') and dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return 5e-2, 5e-2
    else:
        return 1e-4, 1e-4


@dataclass
class TestResult:
    """Result of a single verification test."""
    name: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "errors": self.errors,
        }


@dataclass 
class VerificationResult:
    """Complete verification result from manual testing."""
    random_tests: TestResult
    edge_case_tests: TestResult
    boundary_tests: TestResult
    
    @property
    def all_passed(self) -> bool:
        return (
            self.random_tests.passed and 
            self.edge_case_tests.passed and 
            self.boundary_tests.passed
        )
    
    @property
    def categories_passed(self) -> int:
        return sum([
            self.random_tests.passed,
            self.edge_case_tests.passed,
            self.boundary_tests.passed,
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "random_tests": self.random_tests.to_dict(),
            "edge_case_tests": self.edge_case_tests.to_dict(),
            "boundary_tests": self.boundary_tests.to_dict(),
            "all_passed": self.all_passed,
            "categories_passed": self.categories_passed,
        }


class ManualKernelVerifier:
    """Manual kernel verification using traditional testing.
    
    This class provides testing-based verification with significant limitations:
    
    1. **Incomplete coverage**: Random sampling misses corner cases
    2. **No formal proofs**: Can't prove memory or thread safety
    3. **Human effort**: Edge cases must be manually identified
    4. **Doesn't scale**: Complex kernels need exponentially more tests
    
    Example:
        >>> verifier = ManualKernelVerifier(device="cuda")
        >>> result = verifier.verify(my_kernel, reference_fn, shape=(1024, 1024))
        >>> print(f"Tests passed: {result.all_passed}")
        >>> print(f"Categories: {result.categories_passed}/3")
    
    Key insight: Even with all tests passing, you only know "no bugs found."
    You cannot prove "no bugs exist."
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize verifier with target device.
        
        Args:
            device: CUDA device string (e.g., "cuda", "cuda:0")
        """
        self.device = device
        self._last_result: Optional[VerificationResult] = None
    
    def verify(
        self,
        kernel_fn: Callable[[torch.Tensor], torch.Tensor],
        reference_fn: Callable[[torch.Tensor], torch.Tensor],
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        num_random_tests: int = 20,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> VerificationResult:
        """Run complete verification suite.
        
        Args:
            kernel_fn: The kernel function to verify
            reference_fn: Reference implementation to compare against
            shape: Input tensor shape for testing
            dtype: Tensor data type
            num_random_tests: Number of random input tests
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            
        Returns:
            VerificationResult with all test outcomes
        """
        random_result = self._run_random_tests(
            kernel_fn, reference_fn, shape, dtype, num_random_tests, rtol, atol
        )
        edge_result = self._run_edge_case_tests(
            kernel_fn, reference_fn, shape, dtype
        )
        boundary_result = self._run_boundary_tests(
            kernel_fn, reference_fn, shape, dtype
        )
        
        self._last_result = VerificationResult(
            random_tests=random_result,
            edge_case_tests=edge_result,
            boundary_tests=boundary_result,
        )
        return self._last_result
    
    def _run_random_tests(
        self,
        kernel_fn: Callable,
        reference_fn: Callable,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        num_tests: int,
        rtol: float,
        atol: float,
    ) -> TestResult:
        """Test with random inputs - incomplete coverage.
        
        Limitations:
        - Random sampling misses corner cases
        - No guarantee of covering important code paths
        - May miss numerical edge cases (denormals, inf, nan)
        """
        errors = []
        
        for i in range(num_tests):
            x = torch.randn(*shape, device=self.device, dtype=dtype)
            
            try:
                kernel_out = kernel_fn(x)
                ref_out = reference_fn(x)
                
                if not torch.allclose(kernel_out, ref_out, rtol=rtol, atol=atol):
                    max_diff = (kernel_out - ref_out).abs().max().item()
                    errors.append(f"Test {i}: max diff = {max_diff}")
            except Exception as e:
                errors.append(f"Test {i}: Exception - {e}")
        
        return TestResult(
            name="random_tests",
            passed=len(errors) == 0,
            errors=errors,
        )
    
    def _run_edge_case_tests(
        self,
        kernel_fn: Callable,
        reference_fn: Callable,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> TestResult:
        """Test with manually-identified edge cases.
        
        Limitations:
        - Human must identify all edge cases (error-prone)
        - Easy to miss domain-specific corner cases
        - Doesn't catch all numerical stability issues
        """
        errors = []
        
        # Core edge cases - with dtype-appropriate values
        # FP16 max is ~65504, BF16 is similar to FP32
        large_val = 1e6 if dtype in [torch.float32, torch.bfloat16] else 6e4
        small_val = 1e-6 if dtype in [torch.float32, torch.bfloat16] else 1e-4
        
        edge_cases = [
            ("zeros", torch.zeros(*shape, device=self.device, dtype=dtype)),
            ("ones", torch.ones(*shape, device=self.device, dtype=dtype)),
            ("large", torch.full(shape, large_val, device=self.device, dtype=dtype)),
            ("small", torch.full(shape, small_val, device=self.device, dtype=dtype)),
            ("negative", torch.full(shape, -1.0, device=self.device, dtype=dtype)),
        ]
        
        # Numerical stability edge cases
        edge_cases.extend([
            # Powers of 2 - common FP edge cases
            ("power_of_2", torch.full(shape, 2.0, device=self.device, dtype=dtype)),
            ("inv_power_of_2", torch.full(shape, 0.5, device=self.device, dtype=dtype)),
            # Near-zero denormalized range (dtype-specific)
            ("near_zero_pos", torch.full(shape, 1e-38 if dtype == torch.float32 else (1e-7 if dtype == torch.bfloat16 else 6e-8), device=self.device, dtype=dtype)),
            ("near_zero_neg", torch.full(shape, -1e-38 if dtype == torch.float32 else (-1e-7 if dtype == torch.bfloat16 else -6e-8), device=self.device, dtype=dtype)),
            # Mixed patterns - important for reductions
            ("alternating", torch.tensor([1.0, -1.0] * (shape[0] // 2 if len(shape) > 0 else 1), device=self.device, dtype=dtype).view(*shape[:1], -1).expand(*shape) if shape else torch.tensor([1.0], device=self.device, dtype=dtype)),
        ])
        
        # Gradient-like monotonic sequence (catches off-by-one errors)
        try:
            total_elements = 1
            for s in shape:
                total_elements *= s
            gradient = torch.linspace(-1, 1, total_elements, device=self.device, dtype=dtype).view(*shape)
            edge_cases.append(("gradient", gradient))
        except Exception:
            pass  # Shape may not be compatible
        
        # Sparse-like pattern (mostly zeros with a few non-zeros)
        try:
            sparse = torch.zeros(*shape, device=self.device, dtype=dtype)
            sparse.view(-1)[::max(1, sparse.numel() // 10)] = 1.0
            edge_cases.append(("sparse", sparse))
        except Exception:
            pass
        
        # Add inf/nan only for float types
        if dtype in [torch.float32, torch.float16, torch.bfloat16]:
            edge_cases.extend([
                ("inf", torch.full(shape, float('inf'), device=self.device, dtype=dtype)),
                ("neg_inf", torch.full(shape, float('-inf'), device=self.device, dtype=dtype)),
                ("nan", torch.full(shape, float('nan'), device=self.device, dtype=dtype)),
                # Mixed inf/finite - common source of bugs
                ("mixed_inf", torch.cat([
                    torch.full((shape[0] // 2,) + shape[1:] if len(shape) > 1 else (shape[0] // 2,), float('inf'), device=self.device, dtype=dtype),
                    torch.ones((shape[0] - shape[0] // 2,) + shape[1:] if len(shape) > 1 else (shape[0] - shape[0] // 2,), device=self.device, dtype=dtype)
                ], dim=0) if shape and shape[0] >= 2 else torch.full(shape, float('inf'), device=self.device, dtype=dtype)),
            ])
        
        for name, x in edge_cases:
            try:
                kernel_out = kernel_fn(x)
                ref_out = reference_fn(x)
                
                # Special handling for NaN
                if name == "nan":
                    if not (torch.isnan(kernel_out).all() == torch.isnan(ref_out).all()):
                        errors.append(f"Edge case '{name}': NaN handling mismatch")
                    continue
                
                # Allow NaN outputs for inf inputs
                if name in ["inf", "neg_inf", "mixed_inf"]:
                    continue
                
                rtol, atol = get_dtype_tolerances(dtype)
                if not torch.allclose(kernel_out.float(), ref_out.float(), rtol=rtol, atol=atol, equal_nan=True):
                    max_diff = (kernel_out.float() - ref_out.float()).abs().max().item()
                    errors.append(f"Edge case '{name}': max diff = {max_diff:.6f} (rtol={rtol})")
            except Exception as e:
                errors.append(f"Edge case '{name}': Exception - {e}")
        
        return TestResult(
            name="edge_case_tests",
            passed=len(errors) == 0,
            errors=errors,
        )
    
    def _run_boundary_tests(
        self,
        kernel_fn: Callable,
        reference_fn: Callable,
        base_shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> TestResult:
        """Test boundary conditions for shapes.
        
        Limitations:
        - Doesn't prove correctness for all shapes
        - May miss subtle alignment issues
        - Thread block boundary issues may not surface
        """
        errors = []
        
        # Test various shapes - comprehensive boundary conditions
        test_shapes = []
        
        # 1. Around the base shape (+/- 1)
        for dim_idx in range(len(base_shape)):
            for offset in [-1, 0, 1]:
                shape = list(base_shape)
                shape[dim_idx] = max(1, shape[dim_idx] + offset)
                test_shapes.append(tuple(shape))
        
        # 2. Single element
        test_shapes.append((1,) * len(base_shape))
        
        # 3. Powers of 2 (common CUDA block sizes)
        for power in [5, 7, 8, 9, 10]:  # 32, 128, 256, 512, 1024
            size = 2 ** power
            if len(base_shape) == 1:
                test_shapes.append((size,))
            elif len(base_shape) == 2:
                test_shapes.append((size, size))
            elif len(base_shape) >= 3:
                test_shapes.append((size,) + base_shape[1:])
        
        # 4. Non-power-of-2 (catches alignment issues)
        for size in [31, 33, 63, 65, 127, 129, 255, 257]:
            if len(base_shape) == 1:
                test_shapes.append((size,))
            elif len(base_shape) >= 2:
                test_shapes.append((size, base_shape[1]) if len(base_shape) > 1 else (size,))
        
        # 5. Prime sizes (worst case for many algorithms)
        for size in [17, 31, 127, 257]:
            if len(base_shape) == 1:
                test_shapes.append((size,))
            elif len(base_shape) >= 2:
                test_shapes.append((size, size))
        
        # Remove duplicates while preserving order
        test_shapes = list(dict.fromkeys(test_shapes))
        
        for shape in test_shapes:
            try:
                x = torch.randn(*shape, device=self.device, dtype=dtype)
                kernel_out = kernel_fn(x)
                ref_out = reference_fn(x)
                
                rtol, atol = get_dtype_tolerances(dtype)
                if not torch.allclose(kernel_out.float(), ref_out.float(), rtol=rtol, atol=atol):
                    max_diff = (kernel_out.float() - ref_out.float()).abs().max().item()
                    errors.append(f"Shape {shape}: max diff = {max_diff:.6f} (rtol={rtol})")
            except Exception as e:
                errors.append(f"Shape {shape}: Exception - {e}")
        
        return TestResult(
            name="boundary_tests",
            passed=len(errors) == 0,
            errors=errors,
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get verification metrics for benchmark integration.
        
        Returns metrics compatible with the benchmark harness.
        """
        if not self._last_result:
            return {
                "verification_approach": "manual",
                "tests_run": 0,
                "coverage_guaranteed": False,
                "formal_proof": False,
            }
        
        return {
            "verification_approach": "manual",
            "total_test_categories": 3,
            "categories_passed": self._last_result.categories_passed,
            "num_random_tests": 20,
            "num_edge_cases": 8,
            "num_boundary_tests": 7,
            "all_passed": self._last_result.all_passed,
            "coverage_guaranteed": False,  # Key limitation
            "memory_safety_proven": False,
            "thread_safety_proven": False,
            "formal_proof": False,
        }

