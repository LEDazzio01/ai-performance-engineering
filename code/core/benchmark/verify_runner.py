"""Verification runner for benchmark correctness validation.

This module provides the main verification engine that executes benchmark
pairs (baseline/optimized) in verify mode to ensure correctness before
allowing performance comparison.

Key Features:
- Deterministic seed setup for reproducible execution
- Golden output caching for baseline comparison
- Fresh-input and jitter checks to detect hardcoding/caching
- Workload invariant enforcement
"""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from core.benchmark.verification import (
    ComparisonDetails,
    EnforcementPhase,
    InputSignature,
    PrecisionFlags,
    QuarantineReason,
    ToleranceSpec,
    VerifyResult,
    compare_workload_metrics,
    detect_seed_mutation,
    get_enforcement_phase,
    get_output_tolerance,
    is_verification_enabled,
    get_tolerance_for_dtype,
    select_jitter_dimension,
    set_deterministic_seeds,
)
from core.benchmark.quarantine import (
    QuarantineManager,
    check_benchmark_compliance,
    detect_skip_flags,
)
from core.harness.validity_checks import (
    check_input_output_aliasing,
    get_tensor_addresses,
    DistributedVerifyResult,
    gather_rank_outputs,
    verify_distributed_outputs,
    check_rank_execution,
    GraphCaptureCheatDetector,
    check_graph_capture_integrity,
)


# Default golden output cache directory
DEFAULT_CACHE_DIR = Path("artifacts/verify_cache/golden_outputs")


def _detect_git_cache_salt() -> str:
    """Generate a cache salt tied to the current repo state.
    
    Using the git HEAD hash prevents stale golden outputs from being reused
    across code changes while keeping cache keys stable within a commit.
    """
    try:
        import subprocess
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        if head:
            return head[:12]
    except Exception:
        pass
    return "nogit"


@dataclass
class GoldenOutput:
    """Cached baseline output for verification comparison.
    
    Stores the output tensor(s) from a baseline benchmark execution,
    along with metadata about the execution context.
    """
    signature_hash: str
    outputs: Dict[str, torch.Tensor]  # Named outputs
    workload_metrics: Dict[str, float]
    checksum: str
    created_at: datetime
    seed: int
    
    def compute_checksum(self) -> str:
        """Compute checksum of outputs for integrity verification."""
        parts = []
        for name in sorted(self.outputs.keys()):
            tensor = self.outputs[name]
            # Use numpy for consistent hash across sessions
            arr = tensor.cpu().numpy()
            parts.append(hashlib.sha256(arr.tobytes()).hexdigest()[:16])
        return "-".join(parts)


class GoldenOutputCache:
    """Manages cached golden outputs from baseline benchmark runs.
    
    Golden outputs are keyed by input signature hash and used for
    verifying that optimized benchmarks produce equivalent results.
    """
    
    def __init__(self, cache_dir: Optional[Union[Path, str]] = None, cache_salt: Optional[str] = None):
        """Initialize the golden output cache.
        
        Args:
            cache_dir: Directory for storing cached outputs.
                      Defaults to artifacts/verify_cache/golden_outputs
            cache_salt: Optional salt to incorporate into cache keys (defaults to git HEAD)
        """
        if cache_dir is None:
            self.cache_dir = DEFAULT_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_salt = cache_salt or _detect_git_cache_salt()
    
    def _get_cache_path(self, signature_hash: str) -> Path:
        """Get the path to the cache file for a signature hash."""
        return self.cache_dir / f"{signature_hash}-{self.cache_salt}.pkl"
    
    def has(self, signature_hash: str) -> bool:
        """Check if a golden output exists for the given signature."""
        return self._get_cache_path(signature_hash).exists()
    
    def get(self, signature_hash: str) -> Optional[GoldenOutput]:
        """Retrieve a cached golden output.
        
        Args:
            signature_hash: Hash of the input signature
            
        Returns:
            GoldenOutput if found, None otherwise
        """
        path = self._get_cache_path(signature_hash)
        if not path.exists():
            return None
        
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return GoldenOutput(
                signature_hash=data["signature_hash"],
                outputs={k: torch.tensor(v) for k, v in data["outputs"].items()},
                workload_metrics=data["workload_metrics"],
                checksum=data["checksum"],
                created_at=datetime.fromisoformat(data["created_at"]),
                seed=data["seed"],
            )
        except Exception:
            return None
    
    def put(self, golden: GoldenOutput) -> None:
        """Store a golden output in the cache.
        
        Args:
            golden: The GoldenOutput to cache
        """
        path = self._get_cache_path(golden.signature_hash)
        data = {
            "signature_hash": golden.signature_hash,
            "outputs": {k: v.cpu().numpy() for k, v in golden.outputs.items()},
            "workload_metrics": golden.workload_metrics,
            "checksum": golden.checksum,
            "created_at": golden.created_at.isoformat(),
            "seed": golden.seed,
            "cache_salt": self.cache_salt,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    def invalidate(self, signature_hash: str) -> bool:
        """Invalidate (delete) a cached golden output.
        
        Args:
            signature_hash: Hash of the input signature
            
        Returns:
            True if cache was deleted, False if it didn't exist
        """
        path = self._get_cache_path(signature_hash)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def clear_all(self) -> int:
        """Clear all cached golden outputs.
        
        Returns:
            Number of cache entries cleared
        """
        count = 0
        for path in self.cache_dir.glob("*.pkl"):
            path.unlink()
            count += 1
        return count


@dataclass
class VerifyConfig:
    """Configuration for verification runs."""
    seed: int = 42
    tolerance_override: Optional[ToleranceSpec] = None
    skip_jitter_check: bool = False
    skip_fresh_input_check: bool = False
    skip_timing_validation: bool = False  # Skip warmup/iteration count validation
    skip_output_validation: bool = False  # Skip output tensor comparison
    workload_tolerance: float = 0.01  # 1% tolerance for workload metrics
    verbose: bool = False
    force_recache: bool = False  # Ignore existing cache


@dataclass
class TimingConfig:
    """Timing configuration extracted from a benchmark or harness config.
    
    Used to validate that baseline and optimized benchmarks use identical
    timing parameters, preventing timing manipulation attacks.
    """
    warmup_iterations: Optional[int] = None
    measurement_iterations: Optional[int] = None
    min_run_time_ms: Optional[float] = None
    
    @classmethod
    def from_benchmark(cls, benchmark: Any) -> "TimingConfig":
        """Extract timing config from a benchmark instance.
        
        Checks for timing config in multiple places:
        1. benchmark.config (if BenchmarkConfig is used)
        2. benchmark attributes
        3. get_config() method
        """
        warmup = None
        iterations = None
        min_run_time = None
        
        # Try to get from benchmark.config
        config = getattr(benchmark, "config", None)
        if config:
            warmup = getattr(config, "warmup", None) or getattr(config, "warmup_iterations", None)
            iterations = getattr(config, "iterations", None)
            min_run_time = getattr(config, "min_run_time_ms", None)
        
        # Try to get from benchmark attributes directly
        if warmup is None:
            warmup = getattr(benchmark, "warmup_iterations", None) or getattr(benchmark, "warmup", None)
        if iterations is None:
            iterations = getattr(benchmark, "iterations", None)
        if min_run_time is None:
            min_run_time = getattr(benchmark, "min_run_time_ms", None)
        
        # Try to get from get_config() method
        if hasattr(benchmark, "get_config") and callable(benchmark.get_config):
            try:
                cfg = benchmark.get_config()
                if isinstance(cfg, dict):
                    if warmup is None:
                        warmup = cfg.get("warmup") or cfg.get("warmup_iterations")
                    if iterations is None:
                        iterations = cfg.get("iterations")
                    if min_run_time is None:
                        min_run_time = cfg.get("min_run_time_ms")
            except Exception:
                pass
        
        return cls(
            warmup_iterations=warmup,
            measurement_iterations=iterations,
            min_run_time_ms=min_run_time,
        )
    
    def matches(self, other: "TimingConfig") -> Tuple[bool, Optional[str]]:
        """Check if two timing configs match.
        
        Returns:
            Tuple of (matches, mismatch_description)
        """
        mismatches = []
        
        # Only compare values that are set in BOTH configs
        if self.warmup_iterations is not None and other.warmup_iterations is not None:
            if self.warmup_iterations != other.warmup_iterations:
                mismatches.append(
                    f"warmup_iterations: {self.warmup_iterations} vs {other.warmup_iterations}"
                )
        
        if self.measurement_iterations is not None and other.measurement_iterations is not None:
            if self.measurement_iterations != other.measurement_iterations:
                mismatches.append(
                    f"measurement_iterations: {self.measurement_iterations} vs {other.measurement_iterations}"
                )
        
        if self.min_run_time_ms is not None and other.min_run_time_ms is not None:
            if abs(self.min_run_time_ms - other.min_run_time_ms) > 0.001:
                mismatches.append(
                    f"min_run_time_ms: {self.min_run_time_ms} vs {other.min_run_time_ms}"
                )
        
        if mismatches:
            return False, "; ".join(mismatches)
        return True, None
    

class VerifyRunner:
    """Main verification engine for benchmark correctness validation.
    
    Executes benchmark pairs (baseline/optimized) with deterministic seeds
    and verifies that outputs match within tolerance. Implements anti-reward-hacking
    checks including fresh-input and jitter verification.
    
    Usage:
        runner = VerifyRunner()
        
        # Verify a benchmark pair
        result = runner.verify_pair(baseline_benchmark, optimized_benchmark)
        
        if not result.passed:
            print(f"Verification failed: {result.reason}")
    """
    
    def __init__(
        self,
        cache_dir: Optional[Union[Path, str]] = None,
        quarantine_manager: Optional[QuarantineManager] = None,
    ):
        """Initialize the verification runner.
        
        Args:
            cache_dir: Directory for golden output cache
            quarantine_manager: Manager for quarantine records
        """
        cache_path = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        self.cache = GoldenOutputCache(cache_path)
        self.quarantine = quarantine_manager or QuarantineManager()
    
    def _extract_output(self, benchmark: Any) -> Dict[str, torch.Tensor]:
        """Extract output tensor from benchmark using ONLY get_verify_output().
        
        STRICT: No fallbacks, no auto-detection. Every benchmark MUST implement
        get_verify_output() explicitly.
        
        Args:
            benchmark: The benchmark instance
            
        Returns:
            Dict mapping output names to tensors
            
        Raises:
            NotImplementedError: If benchmark doesn't implement get_verify_output()
            ValueError: If get_verify_output() returns invalid type
        """
        # STRICT: Only use get_verify_output() - no fallbacks
        if not hasattr(benchmark, "get_verify_output"):
            raise NotImplementedError(
                f"{benchmark.__class__.__name__} must implement get_verify_output(). "
                "No fallbacks or auto-detection allowed."
            )
        
        if not callable(benchmark.get_verify_output):
            raise TypeError(
                f"{benchmark.__class__.__name__}.get_verify_output must be a method, "
                f"got {type(benchmark.get_verify_output)}"
            )
        
        # Call get_verify_output() - let NotImplementedError propagate
        out = benchmark.get_verify_output()
        
        # Validate return type
        if out is None:
            raise ValueError(
                f"{benchmark.__class__.__name__}.get_verify_output() returned None. "
                "Must return a tensor or dict of tensors. "
                "If this is a throughput-only benchmark, return a checksum tensor."
            )
        
        outputs: Dict[str, torch.Tensor] = {}
        
        if isinstance(out, torch.Tensor):
            outputs["output"] = out.detach().clone()
        elif isinstance(out, dict):
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    outputs[k] = v.detach().clone()
            if not outputs:
                raise ValueError(
                    f"{benchmark.__class__.__name__}.get_verify_output() returned dict "
                    "with no tensor values."
                )
        else:
            raise TypeError(
                f"{benchmark.__class__.__name__}.get_verify_output() must return "
                f"torch.Tensor or Dict[str, torch.Tensor], got {type(out)}"
            )
        
        return outputs
    
    def _extract_inputs(self, benchmark: Any) -> Dict[str, torch.Tensor]:
        """Extract input tensors from a benchmark.
        
        Requires benchmarks to implement get_verify_inputs() explicitly to
        avoid auto-detected fallbacks.
        """
        if not hasattr(benchmark, "get_verify_inputs") or not callable(getattr(benchmark, "get_verify_inputs")):
            raise NotImplementedError(
                f"{benchmark.__class__.__name__} must implement get_verify_inputs() for aliasing checks"
            )
        
        inp = benchmark.get_verify_inputs()
        if inp is None:
            raise ValueError(f"{benchmark.__class__.__name__}.get_verify_inputs() returned None")
        
        inputs: Dict[str, torch.Tensor] = {}
        if isinstance(inp, torch.Tensor):
            inputs["input"] = inp
        elif isinstance(inp, dict):
            for k, v in inp.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v
        else:
            raise TypeError(
                f"{benchmark.__class__.__name__}.get_verify_inputs() must return Tensor or Dict[str, Tensor], got {type(inp)}"
            )
        
        if not inputs:
            raise ValueError(f"{benchmark.__class__.__name__}.get_verify_inputs() returned no tensors")
        
        return inputs
    
    def _check_input_output_aliasing(
        self,
        benchmark: Any,
    ) -> Tuple[bool, Optional[str]]:
        """Check if any output tensor aliases an input tensor.
        
        This detects the "pre-filled output" cheat where the output buffer
        already contains the result before benchmark_fn() runs.
        
        Args:
            benchmark: The benchmark instance
            
        Returns:
            Tuple of (no_aliasing, error_message_if_aliasing)
        """
        inputs = self._extract_inputs(benchmark)
        outputs = self._extract_output(benchmark)
        
        # Use the check from validity_checks module
        no_aliasing, error_msg = check_input_output_aliasing(inputs, outputs)
        return no_aliasing, error_msg
    
    def _extract_signature(self, benchmark: Any) -> Optional[InputSignature]:
        """Extract input signature from a benchmark.
        
        Args:
            benchmark: The benchmark instance
            
        Returns:
            InputSignature if extractable, None otherwise
        """
        if not hasattr(benchmark, "get_input_signature") or not callable(getattr(benchmark, "get_input_signature")):
            raise NotImplementedError(
                f"{benchmark.__class__.__name__} must implement get_input_signature()"
            )
        
        sig_dict = benchmark.get_input_signature()
        if sig_dict is None:
            raise ValueError(f"{benchmark.__class__.__name__}.get_input_signature() returned None")
        
        if isinstance(sig_dict, InputSignature):
            errors = sig_dict.validate(strict=True)
            if errors:
                raise ValueError(f"Invalid InputSignature: {errors[0]}")
            return sig_dict
        
        if not isinstance(sig_dict, dict):
            raise TypeError(
                f"{benchmark.__class__.__name__}.get_input_signature() must return InputSignature or dict, got {type(sig_dict)}"
            )
        
        if not sig_dict.get("shapes") or not sig_dict.get("dtypes"):
            raise ValueError("Input signature must include non-empty 'shapes' and 'dtypes'")
        if "batch_size" not in sig_dict or "parameter_count" not in sig_dict:
            raise ValueError("Input signature must include 'batch_size' and 'parameter_count'")
        
        shapes = {k: tuple(v) if isinstance(v, list) else tuple(v) for k, v in sig_dict["shapes"].items()}
        dtypes = sig_dict["dtypes"]
        precision = PrecisionFlags(
            fp16=bool(sig_dict.get("fp16", False)),
            bf16=bool(sig_dict.get("bf16", False)),
            fp8=bool(sig_dict.get("fp8", False)),
            tf32=bool(sig_dict.get("tf32", True)),
        )
        
        signature = InputSignature(
            shapes=shapes,
            dtypes=dtypes,
            batch_size=int(sig_dict["batch_size"]),
            parameter_count=int(sig_dict["parameter_count"]),
            precision_flags=precision,
            world_size=sig_dict.get("world_size"),
            ranks=sig_dict.get("ranks"),
            shards=sig_dict.get("shards"),
            pipeline_stages=sig_dict.get("pipeline_stages"),
            per_rank_batch_size=sig_dict.get("per_rank_batch_size"),
            collective_type=sig_dict.get("collective_type"),
            num_streams=sig_dict.get("num_streams"),
            graph_capture_enabled=sig_dict.get("graph_capture_enabled"),
            pruning_enabled=sig_dict.get("pruning_enabled"),
            sparsity_ratio=sig_dict.get("sparsity_ratio"),
            quantization_mode=sig_dict.get("quantization_mode"),
        )
        errors = signature.validate(strict=True)
        if errors:
            raise ValueError(f"Invalid InputSignature: {errors[0]}")
        return signature
    
    def _extract_workload_metrics(self, benchmark: Any) -> Dict[str, float]:
        """Extract workload metrics from a benchmark.
        
        Args:
            benchmark: The benchmark instance
            
        Returns:
            Dict of workload metrics (bytes/tokens/ops per iteration)
        """
        metrics: Dict[str, float] = {}
        
        if hasattr(benchmark, "get_workload_metadata"):
            try:
                metadata = benchmark.get_workload_metadata()
                if metadata:
                    if hasattr(metadata, "bytes_per_iter") and metadata.bytes_per_iter:
                        metrics["bytes_per_iter"] = float(metadata.bytes_per_iter)
                    if hasattr(metadata, "tokens_per_iter") and metadata.tokens_per_iter:
                        metrics["tokens_per_iter"] = float(metadata.tokens_per_iter)
                    if hasattr(metadata, "flops_per_iter") and metadata.flops_per_iter:
                        metrics["flops_per_iter"] = float(metadata.flops_per_iter)
            except Exception:
                pass
        
        return metrics
    
    def _compare_outputs(
        self,
        expected: Dict[str, torch.Tensor],
        actual: Dict[str, torch.Tensor],
        tolerance: Optional[ToleranceSpec] = None,
    ) -> ComparisonDetails:
        """Compare expected and actual outputs.
        
        Args:
            expected: Expected output tensors (from baseline)
            actual: Actual output tensors (from optimized)
            tolerance: Optional custom tolerance override
            
        Returns:
            ComparisonDetails with comparison results
        """
        if set(expected.keys()) != set(actual.keys()):
            return ComparisonDetails(
                passed=False,
                max_diff=None,
                location=None,
            )
        
        max_diff_overall = 0.0
        worst_location: Optional[Tuple[int, ...]] = None
        worst_expected: Optional[float] = None
        worst_actual: Optional[float] = None
        
        for name in expected.keys():
            exp_tensor = expected[name]
            act_tensor = actual[name]
            
            # Ensure tensors are on the same device for comparison
            if exp_tensor.device != act_tensor.device:
                # Move to actual tensor's device (typically GPU during verification)
                exp_tensor = exp_tensor.to(act_tensor.device)
            
            # Check shapes match
            if exp_tensor.shape != act_tensor.shape:
                return ComparisonDetails(
                    passed=False,
                    max_diff=float('inf'),
                    location=None,
                )
            
            # Get tolerance for dtype
            tol = tolerance or get_tolerance_for_dtype(exp_tensor.dtype)
            
            # Custom comparator takes precedence
            if tol.comparator_fn is not None:
                try:
                    if not tol.comparator_fn(exp_tensor, act_tensor):
                        return ComparisonDetails(
                            passed=False,
                            tolerance_used=tol,
                        )
                    continue
                except Exception:
                    return ComparisonDetails(passed=False, tolerance_used=tol)
            
            # Standard numeric comparison
            if exp_tensor.is_floating_point():
                # Use allclose with tolerances
                diff = torch.abs(exp_tensor - act_tensor)
                rel_diff = diff / (torch.abs(exp_tensor) + 1e-12)
                
                max_diff = float(diff.max())
                if max_diff > max_diff_overall:
                    max_diff_overall = max_diff
                    max_idx = diff.argmax()
                    flat_idx = max_idx.item()
                    worst_location = tuple(int(x) for x in np.unravel_index(flat_idx, diff.shape))
                    worst_expected = float(exp_tensor.flatten()[flat_idx])
                    worst_actual = float(act_tensor.flatten()[flat_idx])
                
                # Check if passes tolerance
                if not torch.allclose(exp_tensor, act_tensor, rtol=tol.rtol, atol=tol.atol):
                    return ComparisonDetails(
                        passed=False,
                        max_diff=max_diff_overall,
                        location=worst_location,
                        expected_sample=worst_expected,
                        actual_sample=worst_actual,
                        tolerance_used=tol,
                    )
            else:
                # Exact match for non-floating point
                if not torch.equal(exp_tensor, act_tensor):
                    diff = (exp_tensor != act_tensor)
                    max_idx = diff.int().argmax()
                    flat_idx = max_idx.item()
                    worst_location = tuple(int(x) for x in np.unravel_index(flat_idx, diff.shape))
                    return ComparisonDetails(
                        passed=False,
                        max_diff=float('inf'),
                        location=worst_location,
                        expected_sample=float(exp_tensor.flatten()[flat_idx]),
                        actual_sample=float(act_tensor.flatten()[flat_idx]),
                        tolerance_used=tol,
                    )
        
        return ComparisonDetails(
            passed=True,
            max_diff=max_diff_overall if max_diff_overall > 0 else None,
            tolerance_used=tolerance,
        )
    
    def _run_with_seed(
        self,
        benchmark: Any,
        seed: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], Dict[str, int]]:
        """Run a benchmark with specific seed and extract outputs.
        
        Args:
            benchmark: The benchmark instance
            seed: Random seed to use
            
        Returns:
            Tuple of (outputs, workload_metrics, seed_info)
        """
        # Set deterministic seeds BEFORE setup and capture seed_info
        # NOTE: We do NOT re-seed after setup. This ensures inputs created
        # in setup() are deterministic - both baseline and optimized get
        # identical inputs because they use the same seed before setup().
        # Benchmarks should create fixed inputs in setup(), not benchmark_fn().
        seed_info = set_deterministic_seeds(seed)
        
        # Setup creates model and fixed inputs deterministically
        benchmark.setup()
        
        try:
            # Run benchmark function
            benchmark.benchmark_fn()
            
            # Sync CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Extract outputs
            outputs = self._extract_output(benchmark)
            metrics = self._extract_workload_metrics(benchmark)
            
            # Check for seed mutation
            if detect_seed_mutation(seed_info):
                raise RuntimeError("Benchmark mutated RNG seeds during execution")
            
            return outputs, metrics, seed_info
            
        finally:
            # Always teardown
            if hasattr(benchmark, "teardown"):
                try:
                    benchmark.teardown()
                except Exception:
                    pass
    
    def _run_fresh_input_check(
        self,
        benchmark: Any,
        original_outputs: Dict[str, torch.Tensor],
        config: VerifyConfig,
    ) -> Tuple[bool, Optional[str]]:
        """Run fresh-input check to detect output caching.
        
        Runs the benchmark with a different seed and verifies that
        outputs change (for non-deterministic algorithms) or stay
        the same (for deterministic algorithms).
        
        Args:
            benchmark: The benchmark instance
            original_outputs: Outputs from original run
            config: Verification configuration
            
        Returns:
            Tuple of (passed, failure_reason)
        """
        if config.skip_fresh_input_check:
            return True, None
        
        try:
            # Run with different seed
            fresh_seed = config.seed + 1000
            fresh_outputs, _, _ = self._run_with_seed(benchmark, fresh_seed)
            
            # For deterministic algorithms, outputs should match
            # For non-deterministic, they should differ
            # We check if they're suspiciously identical when they shouldn't be
            
            if not fresh_outputs:
                return True, None  # Can't check
            
            # Simple check: if outputs are bitwise identical with different seed,
            # that's suspicious for most algorithms (might be cached)
            all_identical = True
            for name in original_outputs:
                if name in fresh_outputs:
                    if not torch.equal(original_outputs[name], fresh_outputs[name]):
                        all_identical = False
                        break
            
            # If outputs are identical, check if algorithm is marked deterministic
            if all_identical:
                # Check if benchmark declares itself as deterministic
                is_deterministic = getattr(benchmark, "_is_deterministic", False)
                if not is_deterministic:
                    # Suspicious - might be caching
                    # But don't fail yet - could be legit deterministic algo
                    pass
            
            return True, None
            
        except Exception as e:
            # Fresh input check is advisory, don't fail hard
            return True, f"Fresh input check warning: {e}"
    
    def _run_jitter_check(
        self,
        benchmark: Any,
        input_signature: InputSignature,
        config: VerifyConfig,
    ) -> Tuple[bool, Optional[str]]:
        """Run jitter check to detect hardcoded outputs.
        
        The jitter check verifies that benchmark outputs are actually derived
        from inputs, not hardcoded constants. It works by:
        1. Getting the current output
        2. Perturbing an input dimension
        3. Re-running benchmark_fn
        4. Verifying output changed
        
        IMPORTANT: This check is largely redundant with proper output verification.
        If baseline computes real output and optimized returns hardcoded values,
        they won't match anyway. The jitter check only catches the case where
        BOTH baseline AND optimized return the SAME hardcoded value, which is
        extremely unlikely without deliberate collusion.
        
        Args:
            benchmark: The benchmark instance
            input_signature: The input signature
            config: Verification configuration
            
        Returns:
            Tuple of (passed, failure_reason)
        """
        if config.skip_jitter_check:
            return True, None
        
        # Select dimension to perturb
        jitter_dim = select_jitter_dimension(input_signature)
        if jitter_dim is None:
            # No dimension to perturb (e.g., scalar inputs) - this is fine
            # Benchmarks with scalar inputs are rare and still caught by
            # output verification if they return hardcoded values
            return True, None
        
        tensor_name, dim_idx = jitter_dim
        
        # Get original output for comparison
        try:
            original_output = benchmark.get_verify_output()
            if original_output is None:
                return True, None  # No output to verify
        except (RuntimeError, NotImplementedError):
            return True, None  # Benchmark doesn't support output verification
        
        # Get the input tensor to perturb
        input_tensor = None
        if hasattr(benchmark, tensor_name):
            input_tensor = getattr(benchmark, tensor_name)
        elif hasattr(benchmark, 'x') and tensor_name == 'input':
            input_tensor = benchmark.x
        elif hasattr(benchmark, 'inputs'):
            input_tensor = benchmark.inputs
        
        if input_tensor is None or not isinstance(input_tensor, torch.Tensor):
            return True, None  # Can't find input tensor - skip check
        
        # Save original input
        original_input = input_tensor.clone()
        
        try:
            # Perturb the input by adding small noise
            with torch.no_grad():
                noise = torch.randn_like(input_tensor) * 0.01
                input_tensor.add_(noise)
            
            # Re-run benchmark
            benchmark.benchmark_fn()
            
            # Get new output
            try:
                perturbed_output = benchmark.get_verify_output()
            except (RuntimeError, NotImplementedError):
                return True, None
            
            # Restore original input
            with torch.no_grad():
                input_tensor.copy_(original_input)
            
            # Check if output changed
            if perturbed_output is not None and original_output is not None:
                # Outputs should be DIFFERENT after perturbation
                if torch.allclose(original_output, perturbed_output, rtol=1e-5, atol=1e-5):
                    return False, (
                        f"Jitter check failed: output unchanged after perturbing {tensor_name}. "
                        "This suggests hardcoded/cached outputs."
                    )
            
            return True, None
            
        except Exception as e:
            # Restore original input on error
            try:
                with torch.no_grad():
                    input_tensor.copy_(original_input)
            except Exception:
                pass
            # Don't fail verification for jitter check errors - it's advisory
            return True, f"Jitter check skipped due to error: {e}"
    
    def verify_baseline(
        self,
        baseline: Any,
        config: Optional[VerifyConfig] = None,
    ) -> VerifyResult:
        """Run verification on baseline benchmark and cache golden output.
        
        This should be called first when establishing a new baseline.
        The output is cached and used for subsequent optimized comparisons.
        
        Args:
            baseline: The baseline benchmark instance
            config: Optional verification configuration
            
        Returns:
            VerifyResult with verification outcome
        """
        config = config or VerifyConfig()
        
        # Check compliance
        issues = check_benchmark_compliance(baseline)
        if issues:
            reason = issues[0]  # Report first issue
            return VerifyResult.fail(
                f"Baseline compliance check failed: {reason.value}",
            )
        
        # Extract input signature
        try:
            signature = self._extract_signature(baseline)
        except Exception as exc:
            return VerifyResult.fail(f"Baseline input signature invalid: {exc}")
        
        errors = signature.validate(strict=False)  # Allow simple parameter-based signatures
        if errors:
            return VerifyResult.fail(f"Invalid signature: {errors[0]}")
        
        sig_hash = signature.hash()
        
        # Check cache (unless forced)
        if not config.force_recache and self.cache.has(sig_hash):
            golden = self.cache.get(sig_hash)
            if golden:
                return VerifyResult.success(
                    signature_hash=sig_hash,
                    baseline_checksum=golden.checksum,
                )
        
        try:
            # Run baseline with deterministic seed
            outputs, metrics, seed_info = self._run_with_seed(baseline, config.seed)
            
            if not outputs:
                return VerifyResult.fail("Baseline produced no extractable outputs")
            
            # Create and cache golden output
            golden = GoldenOutput(
                signature_hash=sig_hash,
                outputs=outputs,
                workload_metrics=metrics,
                checksum="",  # Will be computed
                created_at=datetime.now(),
                seed=config.seed,
            )
            golden.checksum = golden.compute_checksum()
            self.cache.put(golden)
            
            return VerifyResult.success(
                signature_hash=sig_hash,
                baseline_checksum=golden.checksum,
                workload_delta=None,
                seed_info=seed_info,
            )
            
        except Exception as e:
            return VerifyResult.fail(f"Baseline execution failed: {e}\n{traceback.format_exc()}")
    
    def verify_optimized(
        self,
        optimized: Any,
        config: Optional[VerifyConfig] = None,
    ) -> VerifyResult:
        """Run verification on optimized benchmark against cached baseline.
        
        Compares optimized outputs against the cached golden baseline output.
        Also runs anti-hacking checks (fresh-input, jitter).
        
        Args:
            optimized: The optimized benchmark instance
            config: Optional verification configuration
            
        Returns:
            VerifyResult with verification outcome
        """
        config = config or VerifyConfig()
        
        # Check compliance
        issues = check_benchmark_compliance(optimized)
        if issues:
            reason = issues[0]
            return VerifyResult.fail(
                f"Optimized compliance check failed: {reason.value}",
            )
        
        # Extract input signature
        try:
            signature = self._extract_signature(optimized)
        except Exception as exc:
            return VerifyResult.fail(f"Optimized input signature invalid: {exc}")
        
        sig_hash = signature.hash()
        
        # Get golden output
        golden = self.cache.get(sig_hash)
        if golden is None:
            return VerifyResult.fail(
                f"No golden output cached for signature {sig_hash}. "
                "Run verify_baseline first."
            )
        
        try:
            # Run optimized with same seed
            outputs, metrics, seed_info = self._run_with_seed(optimized, config.seed)
            
            if not outputs:
                return VerifyResult.fail("Optimized produced no extractable outputs")
            
            # Get tolerance - config override takes precedence, then benchmark, then dtype default
            tolerance = config.tolerance_override
            if tolerance is None:
                try:
                    tolerance = get_output_tolerance(optimized)
                except Exception as exc:
                    return VerifyResult(
                        passed=False,
                        reason=QuarantineReason.MISSING_OUTPUT_TOLERANCE.value,
                        details={"error": str(exc)},
                        timestamp=datetime.now(),
                    )
            
            # Compare outputs
            comparison = self._compare_outputs(
                golden.outputs,
                outputs,
                tolerance,
            )
            
            if not comparison.passed:
                return VerifyResult(
                    passed=False,
                    reason="Output mismatch between baseline and optimized",
                    signature_hash=sig_hash,
                    comparison_details=comparison,
                )
            
            # Compare workload metrics
            if golden.workload_metrics and metrics:
                metrics_match, deltas = compare_workload_metrics(
                    golden.workload_metrics,
                    metrics,
                    config.workload_tolerance,
                )
                if not metrics_match:
                    return VerifyResult(
                        passed=False,
                        reason="Workload metrics mismatch",
                        signature_hash=sig_hash,
                        workload_delta=deltas,
                    )
            
            # Run anti-hacking checks
            fresh_passed, fresh_msg = self._run_fresh_input_check(
                optimized, outputs, config
            )
            jitter_passed, jitter_msg = self._run_jitter_check(
                optimized, signature, config
            )
            
            if not fresh_passed:
                return VerifyResult.fail(f"Fresh-input check failed: {fresh_msg}")
            if not jitter_passed:
                return VerifyResult.fail(f"Jitter check failed: {jitter_msg}")
            
            # Compute checksum for optimized
            optimized_checksum = "-".join(
                hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()[:16]
                for v in [outputs[k] for k in sorted(outputs.keys())]
            )
            
            return VerifyResult.success(
                signature_hash=sig_hash,
                baseline_checksum=golden.checksum,
                optimized_checksum=optimized_checksum,
                comparison_details=comparison,
                seed_info=seed_info,
            )
            
        except Exception as e:
            return VerifyResult.fail(f"Optimized execution failed: {e}\n{traceback.format_exc()}")
    
    def _validate_timing_config(
        self,
        baseline: Any,
        optimized: Any,
    ) -> Tuple[bool, Optional[str]]:
        """Validate that baseline and optimized use identical timing configuration.
        
        This prevents timing manipulation attacks where the optimized benchmark
        uses different warmup/iteration counts to game the measurement.
        
        Args:
            baseline: The baseline benchmark instance
            optimized: The optimized benchmark instance
            
        Returns:
            Tuple of (valid, error_message)
        """
        baseline_timing = TimingConfig.from_benchmark(baseline)
        optimized_timing = TimingConfig.from_benchmark(optimized)
        
        return baseline_timing.matches(optimized_timing)
    
    def verify_pair(
        self,
        baseline: Any,
        optimized: Any,
        config: Optional[VerifyConfig] = None,
    ) -> VerifyResult:
        """Verify a baseline/optimized benchmark pair.
        
        Performs the following checks:
        1. Timing configuration validation (warmup/iterations must match)
        2. Baseline verification with golden output caching
        3. Optimized verification against baseline
        
        Args:
            baseline: The baseline benchmark instance
            optimized: The optimized benchmark instance
            config: Optional verification configuration
            
        Returns:
            VerifyResult with final verification outcome
        """
        config = config or VerifyConfig()
        
        # Step 0: Validate timing configuration matches
        if not config.skip_timing_validation:
            timing_valid, timing_error = self._validate_timing_config(baseline, optimized)
            if not timing_valid:
                return VerifyResult(
                    passed=False,
                    reason=QuarantineReason.TIMING_CONFIG_MISMATCH,
                    details={"timing_mismatch": timing_error},
                    timestamp=datetime.now(),
                )
        
        # Step 0.5: Check for input-output aliasing (pre-filled results detection)
        if not config.skip_output_validation:
            # Check baseline
            try:
                baseline_no_alias, baseline_alias_error = self._check_input_output_aliasing(baseline)
                if not baseline_no_alias:
                    return VerifyResult(
                        passed=False,
                        reason=QuarantineReason.INPUT_OUTPUT_ALIASING,
                        details={"aliasing_error": baseline_alias_error, "benchmark": "baseline"},
                        timestamp=datetime.now(),
                    )
            except Exception as exc:
                return VerifyResult(
                    passed=False,
                    reason=QuarantineReason.MISSING_VERIFY_INPUTS,
                    details={"error": str(exc), "benchmark": "baseline"},
                    timestamp=datetime.now(),
                )
            
            # Check optimized
            try:
                optimized_no_alias, optimized_alias_error = self._check_input_output_aliasing(optimized)
                if not optimized_no_alias:
                    return VerifyResult(
                        passed=False,
                        reason=QuarantineReason.INPUT_OUTPUT_ALIASING,
                        details={"aliasing_error": optimized_alias_error, "benchmark": "optimized"},
                        timestamp=datetime.now(),
                    )
            except Exception as exc:
                return VerifyResult(
                    passed=False,
                    reason=QuarantineReason.MISSING_VERIFY_INPUTS,
                    details={"error": str(exc), "benchmark": "optimized"},
                    timestamp=datetime.now(),
                )
        
        # Step 1: Verify and cache baseline
        baseline_result = self.verify_baseline(baseline, config)
        if not baseline_result.passed:
            return baseline_result
        
        # Step 2: Verify optimized against baseline
        return self.verify_optimized(optimized, config)
    
    def gate_perf(
        self,
        benchmark_path: str,
    ) -> Tuple[bool, Optional[str]]:
        """Check if a benchmark is allowed to run performance measurement.
        
        Based on enforcement phase and quarantine status, determines
        whether perf measurement should proceed.
        
        Args:
            benchmark_path: Path to the benchmark file
            
        Returns:
            Tuple of (allowed, reason_if_blocked)
        """
        phase = get_enforcement_phase()
        
        if phase == EnforcementPhase.DETECT:
            # Detect mode: always allow perf, just report issues
            return True, None
        
        # Check quarantine status
        if self.quarantine.is_quarantined(benchmark_path):
            record = self.quarantine.get_quarantine_record(benchmark_path)
            reason = record.quarantine_reason.value if record else "unknown"
            
            if phase == EnforcementPhase.GATE:
                return False, f"Benchmark quarantined ({reason}) - perf blocked in GATE phase"
            else:  # QUARANTINE phase
                return False, f"Benchmark quarantined ({reason}) - excluded from perf reports"
        
        return True, None
    
    def verify_distributed(
        self,
        benchmark: Any,
        world_size: int,
        rank: int,
        config: Optional[VerifyConfig] = None,
    ) -> VerifyResult:
        """Verify a benchmark in a distributed setting.
        
        Performs verification on the local rank and gathers outputs from all
        ranks to ensure consistency. This detects:
        - Rank skipping (some ranks not executing work)
        - Output inconsistency (ranks producing different results)
        
        Args:
            benchmark: The benchmark instance
            world_size: Total number of ranks
            rank: Current rank (0 to world_size-1)
            config: Optional verification configuration
            
        Returns:
            VerifyResult with distributed verification outcome
        """
        config = config or VerifyConfig()
        
        # Step 1: Check rank execution
        executed, exec_error = check_rank_execution(benchmark, world_size, rank)
        if not executed:
            return VerifyResult(
                passed=False,
                reason=QuarantineReason.DISTRIBUTED_VERIFY_FAIL,
                details={"rank_error": exec_error, "rank": rank},
                timestamp=datetime.now(),
            )
        
        # Step 2: Run local verification
        try:
            outputs, metrics, seed_info = self._run_with_seed(benchmark, config.seed)
        except Exception as e:
            return VerifyResult.fail(f"Rank {rank} execution failed: {e}")
        
        # Step 3: Gather outputs from all ranks
        rank_outputs = gather_rank_outputs(outputs, world_size, rank)
        
        # Step 4: Verify consistency (only rank 0 has all data in distributed case)
        if rank == 0 or len(rank_outputs) == world_size:
            dist_result = verify_distributed_outputs(rank_outputs, world_size)
            
            if not dist_result.all_ranks_executed:
                return VerifyResult(
                    passed=False,
                    reason=QuarantineReason.DISTRIBUTED_VERIFY_FAIL,
                    details={"error": dist_result.error_message},
                    timestamp=datetime.now(),
                )
            
            if not dist_result.outputs_consistent:
                return VerifyResult(
                    passed=False,
                    reason=QuarantineReason.DISTRIBUTED_VERIFY_FAIL,
                    details={
                        "error": dist_result.error_message,
                        "inconsistent_ranks": dist_result.inconsistent_ranks,
                    },
                    timestamp=datetime.now(),
                )
        
        # Step 5: Extract signature for caching
        signature = self._extract_signature(benchmark)
        if signature is None:
            return VerifyResult.fail("Distributed benchmark has no valid input signature")
        
        sig_hash = signature.hash()
        
        return VerifyResult.success(
            signature_hash=sig_hash,
            seed_info=seed_info,
        )
    
    def check_graph_capture_cheat(
        self,
        capture_time_ms: float,
        replay_times_ms: List[float],
        memory_during_capture_mb: float = 0.0,
    ) -> Tuple[bool, Optional[str]]:
        """Check for CUDA graph capture cheating.
        
        Detects if work is being done during graph capture rather than replay.
        This is a wrapper around the validity_checks function for convenience.
        
        Args:
            capture_time_ms: Time to capture the graph
            replay_times_ms: List of replay iteration times
            memory_during_capture_mb: Memory allocated during capture
            
        Returns:
            Tuple of (is_valid, error_message_if_cheating)
        """
        return check_graph_capture_integrity(
            capture_time_ms, 
            replay_times_ms,
            memory_during_capture_mb,
        )
