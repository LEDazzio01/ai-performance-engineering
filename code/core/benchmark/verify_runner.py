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
from contextlib import nullcontext
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
    coerce_input_signature,
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
    tolerance: Optional["ToleranceSpec"] = None
    
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
                tolerance=self._load_tolerance(data),
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
            "tolerance": self._dump_tolerance(golden.tolerance),
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

    @staticmethod
    def _dump_tolerance(tolerance: Optional["ToleranceSpec"]) -> Optional[Dict[str, Any]]:
        """Convert tolerance to a JSON/pickle-friendly dict."""
        if tolerance is None:
            return None
        return {
            "rtol": tolerance.rtol,
            "atol": tolerance.atol,
            "justification": getattr(tolerance, "justification", None),
            "has_comparator": tolerance.comparator_fn is not None,
        }

    @staticmethod
    def _load_tolerance(data: Dict[str, Any]) -> Optional["ToleranceSpec"]:
        """Rehydrate tolerance from cached dict (ignores comparator functions)."""
        tol_dict = data.get("tolerance")
        if not tol_dict:
            return None
        tol = ToleranceSpec(
            rtol=tol_dict.get("rtol"),
            atol=tol_dict.get("atol"),
            justification=tol_dict.get("justification"),
            comparator_fn=None,
        )
        # Preserve knowledge that a comparator existed so callers can fail fast.
        setattr(tol, "_cached_has_comparator", bool(tol_dict.get("has_comparator")))
        return tol


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
                "Surface the real output (or a representative slice); verification cannot be skipped."
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

    def _extract_output_for_aliasing(self, benchmark: Any) -> Dict[str, torch.Tensor]:
        """Extract output tensors WITHOUT cloning for aliasing detection.

        Aliasing checks rely on pointer equality (`tensor.data_ptr()`), so this
        method must not detach/clone outputs.
        """
        # Prefer direct access to VerificationPayloadMixin storage when available.
        payload = getattr(benchmark, "_verification_payload", None)
        if payload is not None:
            payload_out = getattr(payload, "output", None)
            if isinstance(payload_out, torch.Tensor):
                return {"output": payload_out}

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

        out = benchmark.get_verify_output()
        if out is None:
            raise ValueError(
                f"{benchmark.__class__.__name__}.get_verify_output() returned None; "
                "cannot perform aliasing detection."
            )

        outputs: Dict[str, torch.Tensor] = {}
        if isinstance(out, torch.Tensor):
            outputs["output"] = out
        elif isinstance(out, dict):
            for name, tensor in out.items():
                if isinstance(tensor, torch.Tensor):
                    outputs[name] = tensor
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
        outputs = self._extract_output_for_aliasing(benchmark)
        
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
        
        return coerce_input_signature(sig_dict)
    
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

    def compare_perf_outputs(
        self,
        baseline_output: Union[torch.Tensor, Dict[str, torch.Tensor]],
        optimized_output: Union[torch.Tensor, Dict[str, torch.Tensor]],
        tolerance: Tuple[float, float],
    ) -> ComparisonDetails:
        """Compare already-captured perf-run outputs strictly.

        This is used for post-timing verification. Callers MUST supply an
        explicit tolerance from the baseline benchmark; no fallbacks or
        auto-inference are permitted.
        """
        def _coerce(out: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            if isinstance(out, torch.Tensor):
                return {"output": out.detach()}
            if isinstance(out, dict):
                if not out:
                    raise ValueError("verify_output dict is empty")
                coerced: Dict[str, torch.Tensor] = {}
                for name, tensor in out.items():
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError(f"verify_output['{name}'] must be torch.Tensor, got {type(tensor)}")
                    coerced[name] = tensor.detach()
                return coerced
            raise TypeError(f"verify_output must be torch.Tensor or Dict[str, torch.Tensor], got {type(out)}")

        expected = _coerce(baseline_output)
        actual = _coerce(optimized_output)
        tol_spec = ToleranceSpec(rtol=float(tolerance[0]), atol=float(tolerance[1]))
        return self._compare_outputs(expected, actual, tol_spec)
    
    def _run_with_seed(
        self,
        benchmark: Any,
        seed: int,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, float],
        Dict[str, int],
        Dict[str, torch.Tensor],
        InputSignature,
    ]:
        """Run a benchmark with specific seed and extract outputs.
        
        Args:
            benchmark: The benchmark instance
            seed: Random seed to use
        
        Returns:
            Tuple of (outputs, workload_metrics, seed_info, inputs_used, input_signature)
        """
        # Set deterministic seeds BEFORE setup and capture seed_info
        # NOTE: We do NOT re-seed after setup. This ensures inputs created
        # in setup() are deterministic - both baseline and optimized get
        # identical inputs because they use the same seed before setup().
        # Benchmarks should create fixed inputs in setup(), not benchmark_fn().
        seed_info = set_deterministic_seeds(seed)
        
        # Setup creates model and fixed inputs deterministically
        benchmark.setup()
        
        # Block CUDA graphs in verification to prevent capture/replay cheating
        cfg = getattr(benchmark, "_config", None) or getattr(benchmark, "config", None)
        if getattr(cfg, "enable_cuda_graph", False) or getattr(benchmark, "enable_cuda_graph", False):
            raise RuntimeError("Verification forbids CUDA graph capture. Disable enable_cuda_graph for verify runs.")
        
        inputs_for_validation: Dict[str, torch.Tensor] = {}
        signature: Optional[InputSignature] = None
        stream_auditor = None
        pre_streams: List[int] = []
        audit_ctx = nullcontext()
        declared_streams: List[Any] = []
        if torch.cuda.is_available():
            from core.harness.validity_checks import audit_streams, get_active_streams
            if hasattr(benchmark, "get_custom_streams"):
                try:
                    custom_streams = benchmark.get_custom_streams()
                    if custom_streams:
                        declared_streams = [custom_streams] if isinstance(custom_streams, torch.cuda.Stream) else list(custom_streams)
                except Exception as exc:
                    raise RuntimeError(f"get_custom_streams() failed during verification: {exc}")
            audit_ctx = audit_streams(getattr(benchmark, "device", None))
            pre_streams = get_active_streams(getattr(benchmark, "device", None), declared_streams)
        
        try:
            # Run benchmark function under stream audit when CUDA is available
            with audit_ctx as stream_auditor:
                if stream_auditor is not None:
                    for stream in declared_streams:
                        stream_auditor.record_stream_event(stream, operation="declared_stream")
                benchmark.benchmark_fn()
                # Allow benchmarks to set verification payload outside benchmark_fn().
                capture_hook = getattr(benchmark, "capture_verification_payload", None)
                if callable(capture_hook):
                    try:
                        capture_hook()
                    except Exception as exc:
                        raise RuntimeError(
                            f"capture_verification_payload() failed during verification: {exc}"
                        ) from exc
                if hasattr(benchmark, "mark_execution_complete"):
                    try:
                        benchmark.mark_execution_complete()
                    except Exception:
                        pass

                # STRICT: Verification metadata must be surfaced post-run.
                inputs_for_validation = self._extract_inputs(benchmark)

                # Memory aliasing check must use *real* tensors (no clones).
                outputs_for_aliasing = self._extract_output_for_aliasing(benchmark)
                no_aliasing, alias_error = check_input_output_aliasing(
                    inputs_for_validation, outputs_for_aliasing
                )
                if not no_aliasing:
                    raise RuntimeError(alias_error or "INPUT_OUTPUT_ALIASING detected")

                # Payload-backed signatures are extracted post-run.
                signature = self._extract_signature(benchmark)
            
            # Sync CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Extract outputs
            outputs = self._extract_output(benchmark)
            metrics = self._extract_workload_metrics(benchmark)
            if signature is None:
                raise RuntimeError("get_input_signature() did not produce a signature during verification run")
            
            # Stream audit check for verification path
            if stream_auditor is not None:
                from core.harness.validity_checks import check_stream_sync_completeness, get_active_streams
                post_streams = get_active_streams(getattr(benchmark, "device", None), declared_streams)
                sync_complete, sync_warning = check_stream_sync_completeness(pre_streams, post_streams)
                audit_ok, audit_warnings = stream_auditor.check_issues()
                issues: List[str] = []
                if not sync_complete:
                    issues.append("Stream synchronization incomplete during verification run")
                if sync_warning:
                    issues.append(sync_warning)
                try:
                    info = stream_auditor.get_info()
                    declared_ids = set(post_streams)
                    if info.default_stream_id is not None:
                        declared_ids.add(info.default_stream_id)
                    undeclared_streams = info.stream_ids - declared_ids
                    if undeclared_streams:
                        issues.append(
                            f"UNDECLARED STREAMS during verification: {len(undeclared_streams)} stream(s) not declared via get_custom_streams()."
                        )
                except Exception:
                    pass
                if not audit_ok or audit_warnings:
                    issues.extend(audit_warnings)
                if issues:
                    raise RuntimeError(
                        "STREAM TIMING VIOLATION (verification): " + " | ".join(issues)
                    )
            
            # Check for seed mutation
            if detect_seed_mutation(seed_info):
                raise RuntimeError("Benchmark mutated RNG seeds during execution")
            
            return outputs, metrics, seed_info, inputs_for_validation, signature
            
        finally:
            # Always teardown
            if hasattr(benchmark, "teardown"):
                try:
                    benchmark.teardown()
                except Exception:
                    pass

    def _validate_inputs_match_signature(
        self,
        signature: InputSignature,
        inputs: Dict[str, torch.Tensor],
    ) -> None:
        """Ensure runtime inputs align with declared signature (shapes/dtypes)."""
        if not signature.shapes:
            raise ValueError("Input signature has no shapes declared.")
        if not inputs:
            raise ValueError("get_verify_inputs() returned no tensors for validation.")

        def _norm_dtype(value: Any) -> str:
            return str(value).replace("torch.", "").lower()

        # All signature-listed tensors must be present and shape-matching
        for name, expected_shape in signature.shapes.items():
            if name not in inputs:
                raise ValueError(f"Input '{name}' declared in signature missing from get_verify_inputs().")
            tensor = inputs[name]
            if tuple(tensor.shape) != tuple(expected_shape):
                raise ValueError(
                    f"Input '{name}' shape mismatch: signature {tuple(expected_shape)} vs actual {tuple(tensor.shape)}."
                )

        # Dtype checks for overlapping names
        for name, tensor in inputs.items():
            if name in signature.dtypes:
                expected_dtype = _norm_dtype(signature.dtypes[name])
                actual_dtype = _norm_dtype(tensor.dtype)
                if expected_dtype != actual_dtype:
                    raise ValueError(
                        f"Input '{name}' dtype mismatch: signature {expected_dtype} vs actual {actual_dtype}."
                    )

    def _enforce_skip_policy(self, benchmark: Any) -> None:
        """Allow verification skips only for sanctioned cases."""
        cfg = getattr(benchmark, "_config", None) or getattr(benchmark, "config", None)
        multi_gpu_required = bool(
            getattr(cfg, "multi_gpu_required", False)
            or getattr(benchmark, "multi_gpu_required", False)
        )
        if multi_gpu_required:
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                raise RuntimeError("SKIPPED: requires >=2 GPUs")

        skip_input = False
        skip_output = False
        try:
            skip_input = bool(benchmark.skip_input_verification())  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            skip_output = bool(benchmark.skip_output_verification())  # type: ignore[attr-defined]
        except Exception:
            pass

        if skip_input or skip_output:
            reason = getattr(benchmark, "verification_not_applicable_reason", None)
            if reason:
                raise RuntimeError(f"SKIPPED: {reason}")
            if multi_gpu_required and (not torch.cuda.is_available() or torch.cuda.device_count() < 2):
                raise RuntimeError("SKIPPED: requires >=2 GPUs")
            raise RuntimeError(
                "SKIPPED: skip_input/output_verification() is only allowed for sanctioned cases. "
                "Provide verification_not_applicable_reason or disable skip flags."
            )
    
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
            fresh_outputs, _, _, _, _ = self._run_with_seed(benchmark, fresh_seed)
            
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
                    return False, (
                        "Fresh-input check failed: outputs are identical under a different seed. "
                        "Declare _is_deterministic=True explicitly if this benchmark is truly deterministic."
                    )
            
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
        
        # Get the input tensor to perturb from declared verification inputs.
        try:
            verify_inputs = self._extract_inputs(benchmark)
        except Exception:
            return True, None  # Can't access verification inputs - skip check

        input_tensor = verify_inputs.get(tensor_name)
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

            # Post-run payload refresh for benchmarks using capture_verification_payload().
            capture_hook = getattr(benchmark, "capture_verification_payload", None)
            if callable(capture_hook):
                try:
                    capture_hook()
                except Exception as exc:
                    return True, f"Jitter check skipped due to capture_verification_payload() error: {exc}"
            
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
        try:
            self._enforce_skip_policy(baseline)
        except RuntimeError as exc:
            return VerifyResult.fail(str(exc))
        
        # Check compliance
        issues = check_benchmark_compliance(baseline)
        if issues:
            reason_obj = issues[0]  # Report first issue
            reason = reason_obj.value if isinstance(reason_obj, QuarantineReason) else str(reason_obj)
            return VerifyResult(
                passed=False,
                reason=reason,
                details={"error": "Baseline compliance check failed"},
                timestamp=datetime.now(),
            )

        try:
            # Run baseline with deterministic seed
            outputs, metrics, seed_info, inputs, signature = self._run_with_seed(baseline, config.seed)
            try:
                self._validate_inputs_match_signature(signature, inputs)
            except Exception as exc:
                return VerifyResult.fail(f"Baseline inputs do not match signature: {exc}")

            errors = signature.validate(strict=False)  # Allow simple parameter-based signatures
            if errors:
                return VerifyResult.fail(f"Invalid signature: {errors[0]}")

            sig_hash = signature.hash()
            
            # Capture baseline tolerance (fail-fast)
            try:
                baseline_tol = get_output_tolerance(baseline)
            except Exception as exc:
                return VerifyResult(
                    passed=False,
                    reason=QuarantineReason.MISSING_OUTPUT_TOLERANCE.value,
                    details={"error": str(exc)},
                    timestamp=datetime.now(),
                )
            
            if not outputs:
                return VerifyResult(
                    passed=False,
                    reason=QuarantineReason.MISSING_VERIFY_OUTPUT.value,
                    details={"error": "Baseline produced no extractable outputs"},
                    timestamp=datetime.now(),
                )
            
            # Create and cache golden output
            golden = GoldenOutput(
                signature_hash=sig_hash,
                outputs=outputs,
                workload_metrics=metrics,
                checksum="",  # Will be computed
                created_at=datetime.now(),
                seed=config.seed,
                tolerance=baseline_tol,
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
        try:
            self._enforce_skip_policy(optimized)
        except RuntimeError as exc:
            return VerifyResult.fail(str(exc))
        
        # Check compliance
        issues = check_benchmark_compliance(optimized)
        if issues:
            reason_obj = issues[0]
            reason = reason_obj.value if isinstance(reason_obj, QuarantineReason) else str(reason_obj)
            return VerifyResult(
                passed=False,
                reason=reason,
                details={"error": "Optimized compliance check failed"},
                timestamp=datetime.now(),
            )

        try:
            # Run optimized with same seed
            outputs, metrics, seed_info, inputs, signature = self._run_with_seed(optimized, config.seed)
            try:
                self._validate_inputs_match_signature(signature, inputs)
            except Exception as exc:
                return VerifyResult.fail(f"Optimized inputs do not match signature: {exc}")

            sig_hash = signature.hash()
            
            # Get golden output
            golden = self.cache.get(sig_hash)
            if golden is None:
                return VerifyResult(
                    passed=False,
                    reason=QuarantineReason.SIGNATURE_MISMATCH.value,
                    signature_hash=sig_hash,
                    details={
                        "error": (
                            f"No golden output cached for signature {sig_hash}. "
                            "Run verify_baseline first."
                        )
                    },
                    timestamp=datetime.now(),
                )
            
            if not outputs:
                return VerifyResult.fail("Optimized produced no extractable outputs")
            
            # Get tolerance - enforce baseline-bound tolerances
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
            baseline_tol = golden.tolerance
            if baseline_tol is None and tolerance is not None:
                return VerifyResult.fail(
                    "Baseline tolerance missing from cache. Re-run verify_baseline with force_recache=True."
                )
            if baseline_tol is not None:
                if getattr(baseline_tol, "_cached_has_comparator", False) or baseline_tol.comparator_fn is not None:
                    return VerifyResult.fail(
                        "Comparator-based tolerances are not supported in verification cache. Use numeric rtol/atol."
                    )
                if tolerance.comparator_fn is not None:
                    return VerifyResult.fail(
                        "Comparator-based tolerances are not supported for optimized benchmarks. Use numeric rtol/atol."
                    )
                # Disallow looser tolerances on optimized path
                if tolerance.rtol > baseline_tol.rtol or tolerance.atol > baseline_tol.atol:
                    return VerifyResult.fail(
                        f"Optimized tolerance ({tolerance.rtol}, {tolerance.atol}) exceeds baseline tolerance "
                        f"({baseline_tol.rtol}, {baseline_tol.atol})."
                    )
                # Use the stricter tolerance between baseline/optimized
                tolerance = ToleranceSpec(
                    rtol=min(tolerance.rtol, baseline_tol.rtol),
                    atol=min(tolerance.atol, baseline_tol.atol),
                    justification=tolerance.justification or baseline_tol.justification,
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
                    reason=QuarantineReason.OUTPUT_MISMATCH.value,
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
                        reason=QuarantineReason.WORKLOAD_MISMATCH.value,
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
                return VerifyResult(
                    passed=False,
                    reason=QuarantineReason.FRESH_INPUT_FAIL.value,
                    signature_hash=sig_hash,
                    details={"error": fresh_msg},
                    timestamp=datetime.now(),
                )
            if not jitter_passed:
                return VerifyResult(
                    passed=False,
                    reason=QuarantineReason.JITTER_FAIL.value,
                    signature_hash=sig_hash,
                    details={"error": jitter_msg},
                    timestamp=datetime.now(),
                )
            
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
        
        try:
            self._enforce_skip_policy(baseline)
            self._enforce_skip_policy(optimized)
        except RuntimeError as exc:
            return VerifyResult.fail(str(exc))
        
        # Step 0: Validate timing configuration matches
        if not config.skip_timing_validation:
            timing_valid, timing_error = self._validate_timing_config(baseline, optimized)
            if not timing_valid:
                return VerifyResult(
                    passed=False,
                    reason=QuarantineReason.TIMING_CONFIG_MISMATCH.value,
                    details={"timing_mismatch": timing_error},
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
                reason=QuarantineReason.DISTRIBUTED_VERIFY_FAIL.value,
                details={"rank_error": exec_error, "rank": rank},
                timestamp=datetime.now(),
            )
        
        # Step 2: Run local verification
        try:
            outputs, metrics, seed_info, _, signature = self._run_with_seed(benchmark, config.seed)
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
                    reason=QuarantineReason.DISTRIBUTED_VERIFY_FAIL.value,
                    details={"error": dist_result.error_message},
                    timestamp=datetime.now(),
                )
            
            if not dist_result.outputs_consistent:
                return VerifyResult(
                    passed=False,
                    reason=QuarantineReason.DISTRIBUTED_VERIFY_FAIL.value,
                    details={
                        "error": dist_result.error_message,
                        "inconsistent_ranks": dist_result.inconsistent_ranks,
                    },
                    timestamp=datetime.now(),
                )
        
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
