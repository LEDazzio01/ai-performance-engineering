"""Benchmark Validity Checks.

This module provides utilities to detect and prevent benchmark validity issues
that could lead to misleading performance measurements.

Categories covered:
- Memory manipulation (aliasing, pre-allocation)
- Work relocation (setup pre-computation, lazy evaluation)
- Environment issues (GPU state, GC interference)
- Compilation issues (cache, recompilation)

Reference:
    See docs/remaining_protections.md for the full list of 93 validity issues.
"""

from __future__ import annotations

import gc
import hashlib
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import torch
except ImportError:
    torch = None  # type: ignore


# =============================================================================
# GPU State Monitoring
# =============================================================================

@dataclass
class GPUState:
    """Snapshot of GPU state for validity checking."""
    device_index: int
    device_name: str
    temperature_c: Optional[float] = None
    clock_mhz: Optional[int] = None
    memory_clock_mhz: Optional[int] = None
    power_draw_w: Optional[float] = None
    memory_used_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None
    throttle_reason: Optional[str] = None


def capture_gpu_state(device_index: int = 0) -> GPUState:
    """Capture current GPU state for comparison.
    
    Uses pynvml if available, falls back to basic info otherwise.
    """
    if torch is None or not torch.cuda.is_available():
        return GPUState(device_index=device_index, device_name="N/A")
    
    props = torch.cuda.get_device_properties(device_index)
    state = GPUState(
        device_index=device_index,
        device_name=props.name,
    )
    
    # Try to get detailed info via pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        
        # Temperature
        try:
            state.temperature_c = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            pass
        
        # Clock speeds
        try:
            state.clock_mhz = pynvml.nvmlDeviceGetClockInfo(
                handle, pynvml.NVML_CLOCK_SM
            )
            state.memory_clock_mhz = pynvml.nvmlDeviceGetClockInfo(
                handle, pynvml.NVML_CLOCK_MEM
            )
        except Exception:
            pass
        
        # Power
        try:
            state.power_draw_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            pass
        
        # Throttle reason
        try:
            throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            if throttle != 0:
                reasons = []
                if throttle & 0x1:
                    reasons.append("GpuIdle")
                if throttle & 0x2:
                    reasons.append("ApplicationsClocks")
                if throttle & 0x4:
                    reasons.append("SwPowerCap")
                if throttle & 0x8:
                    reasons.append("HwSlowdown")
                if throttle & 0x10:
                    reasons.append("SyncBoost")
                if throttle & 0x20:
                    reasons.append("SwThermalSlowdown")
                if throttle & 0x40:
                    reasons.append("HwThermalSlowdown")
                if throttle & 0x80:
                    reasons.append("HwPowerBrakeSlowdown")
                state.throttle_reason = ",".join(reasons) if reasons else None
        except Exception:
            pass
        
        pynvml.nvmlShutdown()
    except ImportError:
        pass  # pynvml not available
    except Exception:
        pass  # pynvml error
    
    # Memory info from PyTorch
    try:
        state.memory_used_mb = torch.cuda.memory_allocated(device_index) / (1024 * 1024)
        state.memory_total_mb = props.total_memory / (1024 * 1024)
    except Exception:
        pass
    
    return state


def check_gpu_state_consistency(before: GPUState, after: GPUState, 
                                  temp_threshold: float = 10.0,
                                  clock_threshold_pct: float = 10.0) -> Tuple[bool, List[str]]:
    """Check if GPU state changed significantly during benchmark.
    
    Returns:
        Tuple of (is_consistent, list_of_warnings)
    """
    warnings_list: List[str] = []
    
    # Temperature increase
    if before.temperature_c is not None and after.temperature_c is not None:
        temp_delta = after.temperature_c - before.temperature_c
        if temp_delta > temp_threshold:
            warnings_list.append(
                f"GPU temperature increased {temp_delta:.1f}°C during benchmark "
                f"({before.temperature_c}°C → {after.temperature_c}°C)"
            )
    
    # Clock speed drop
    if before.clock_mhz is not None and after.clock_mhz is not None:
        clock_drop_pct = (before.clock_mhz - after.clock_mhz) / before.clock_mhz * 100
        if clock_drop_pct > clock_threshold_pct:
            warnings_list.append(
                f"GPU clock dropped {clock_drop_pct:.1f}% during benchmark "
                f"({before.clock_mhz}MHz → {after.clock_mhz}MHz) - possible thermal throttling"
            )
    
    # Throttling detected
    if after.throttle_reason and "Thermal" in after.throttle_reason:
        warnings_list.append(f"GPU throttling detected: {after.throttle_reason}")
    
    return len(warnings_list) == 0, warnings_list


# =============================================================================
# Memory Validity Checks
# =============================================================================

def get_tensor_addresses(tensors: Dict[str, Any]) -> Dict[str, int]:
    """Get memory addresses of tensors for aliasing detection.
    
    Args:
        tensors: Dict of tensor name -> tensor
        
    Returns:
        Dict of tensor name -> data_ptr address
    """
    addresses = {}
    for name, tensor in tensors.items():
        if torch is not None and isinstance(tensor, torch.Tensor):
            addresses[name] = tensor.data_ptr()
    return addresses


def check_input_output_aliasing(input_tensors: Dict[str, Any],
                                  output_tensors: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Check if any output tensor aliases an input tensor.
    
    This detects the "pre-filled output" cheat where the output buffer
    already contains the result before benchmark_fn() runs.
    
    Returns:
        Tuple of (no_aliasing, error_message_if_aliasing)
    """
    input_addrs = get_tensor_addresses(input_tensors)
    output_addrs = get_tensor_addresses(output_tensors)
    
    for out_name, out_addr in output_addrs.items():
        for in_name, in_addr in input_addrs.items():
            if out_addr == in_addr:
                return False, (
                    f"OUTPUT ALIASING DETECTED: Output '{out_name}' has same memory address "
                    f"as input '{in_name}' (0x{out_addr:x}). This may indicate pre-filled results."
                )
    
    return True, None


def reset_cuda_memory_pool(device: Optional[Any] = None) -> None:
    """Reset CUDA memory pool to prevent memory reuse gaming.
    
    This ensures each benchmark run starts with a clean memory state,
    preventing cached allocations from skewing timing.
    """
    if torch is None or not torch.cuda.is_available():
        return
    
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    
    # Also reset the memory stats
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats(device)
    if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
        torch.cuda.reset_accumulated_memory_stats(device)


# =============================================================================
# Setup Pre-computation Detection
# =============================================================================

def hash_tensors(tensors: Dict[str, Any]) -> str:
    """Compute a hash of tensor contents for change detection.
    
    Used to detect if setup() pre-computes results by modifying tensors.
    """
    if torch is None:
        return "no_torch"
    
    hasher = hashlib.sha256()
    for name in sorted(tensors.keys()):
        tensor = tensors[name]
        if isinstance(tensor, torch.Tensor):
            # Hash tensor metadata and content
            hasher.update(name.encode())
            hasher.update(str(tensor.shape).encode())
            hasher.update(str(tensor.dtype).encode())
            hasher.update(str(tensor.device).encode())
            # Sample content (full hash would be too slow)
            if tensor.numel() > 0:
                flat = tensor.flatten()
                # Hash first, last, and middle elements
                samples = [flat[0].item(), flat[-1].item()]
                if tensor.numel() > 2:
                    samples.append(flat[tensor.numel() // 2].item())
                hasher.update(str(samples).encode())
    
    return hasher.hexdigest()[:16]


def check_setup_precomputation(get_outputs_fn, setup_fn) -> Tuple[bool, Optional[str]]:
    """Check if setup() pre-computes results.
    
    Args:
        get_outputs_fn: Function that returns dict of output tensors
        setup_fn: The benchmark's setup() function
        
    Returns:
        Tuple of (no_precomputation, error_message_if_detected)
    """
    # Get outputs before setup
    try:
        outputs_before = get_outputs_fn()
        hash_before = hash_tensors(outputs_before)
    except Exception:
        # Can't check if outputs not available
        return True, None
    
    # Run setup
    setup_fn()
    
    # Get outputs after setup
    try:
        outputs_after = get_outputs_fn()
        hash_after = hash_tensors(outputs_after)
    except Exception:
        return True, None
    
    if hash_before != hash_after:
        return False, (
            "SETUP PRE-COMPUTATION DETECTED: Output tensors changed during setup(). "
            f"Hash before: {hash_before}, after: {hash_after}. "
            "setup() should only initialize inputs, not compute outputs."
        )
    
    return True, None


# =============================================================================
# Garbage Collection Control
# =============================================================================

@contextmanager
def gc_disabled():
    """Context manager to disable garbage collection during timing.
    
    Prevents GC from interfering with timing measurements.
    
    Example:
        with gc_disabled():
            # GC won't run during this block
            results = benchmark_fn()
    """
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if gc_was_enabled:
            gc.enable()


# =============================================================================
# torch.compile Cache Management
# =============================================================================

def clear_compile_cache() -> bool:
    """Clear torch.compile/dynamo cache for consistent compilation.
    
    Returns:
        True if cache was cleared, False if not available
    """
    if torch is None:
        return False
    
    try:
        # Reset dynamo
        if hasattr(torch, '_dynamo'):
            torch._dynamo.reset()
        
        # Reset inductor cache if available
        if hasattr(torch, '_inductor') and hasattr(torch._inductor, 'codecache'):
            if hasattr(torch._inductor.codecache, 'clear'):
                torch._inductor.codecache.clear()
        
        return True
    except Exception:
        return False


def get_compile_state() -> Dict[str, Any]:
    """Get current torch.compile state for consistency checking."""
    state = {
        "dynamo_available": False,
        "compile_count": 0,
        "cache_entries": 0,
    }
    
    if torch is None:
        return state
    
    try:
        if hasattr(torch, '_dynamo'):
            state["dynamo_available"] = True
            if hasattr(torch._dynamo, 'utils') and hasattr(torch._dynamo.utils, 'counters'):
                counters = torch._dynamo.utils.counters
                state["compile_count"] = counters.get("compile", {}).get("calls", 0)
    except Exception:
        pass
    
    return state


# =============================================================================
# Lazy Evaluation Detection
# =============================================================================

def force_tensor_evaluation(tensors: Dict[str, Any]) -> None:
    """Force evaluation of potentially lazy tensors.
    
    This prevents the "lazy evaluation skip" cheat where unevaluated
    tensors are returned without doing actual computation.
    """
    if torch is None:
        return
    
    for name, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            # Sync ensures any pending operations complete
            if tensor.is_cuda:
                torch.cuda.synchronize(tensor.device)
            # .item() on a small tensor forces full evaluation
            # (we use a single element to minimize overhead)
            if tensor.numel() > 0:
                try:
                    _ = tensor.flatten()[0].item()
                except Exception:
                    pass  # Some tensors may not support .item()


# =============================================================================
# Environment Validation
# =============================================================================

def validate_environment() -> Tuple[bool, List[str]]:
    """Validate benchmark environment is suitable.
    
    Checks for common issues that could affect benchmark validity.
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings_list: List[str] = []
    
    if torch is None or not torch.cuda.is_available():
        warnings_list.append("CUDA not available")
        return False, warnings_list
    
    # Check device count
    device_count = torch.cuda.device_count()
    if device_count > 1:
        warnings_list.append(
            f"Multiple GPUs detected ({device_count}). Ensure benchmarks use "
            "consistent device placement."
        )
    
    # Check CUDA version consistency
    try:
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        # Just log, don't warn (for manifest)
    except Exception:
        pass
    
    # Check for debug mode
    if hasattr(torch, 'autograd') and hasattr(torch.autograd, 'grad_mode'):
        if torch.is_grad_enabled():
            # Grad enabled is normal, but in benchmarks we often want it disabled
            pass
    
    # Check for torch.compile backend consistency
    try:
        if hasattr(torch, '_dynamo'):
            config = getattr(torch._dynamo, 'config', None)
            if config and hasattr(config, 'suppress_errors') and config.suppress_errors:
                warnings_list.append(
                    "torch._dynamo.config.suppress_errors=True may hide compilation issues"
                )
    except Exception:
        pass
    
    return len(warnings_list) == 0, warnings_list


# =============================================================================
# Memory Allocation Tracking
# =============================================================================

@dataclass
class MemoryAllocationSnapshot:
    """Snapshot of CUDA memory allocations for tracking patterns."""
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    num_allocs: int
    num_frees: int
    
    @classmethod
    def capture(cls, device: Optional[Any] = None) -> "MemoryAllocationSnapshot":
        """Capture current memory allocation state."""
        if torch is None or not torch.cuda.is_available():
            return cls(0.0, 0.0, 0.0, 0, 0)
        
        stats = torch.cuda.memory_stats(device)
        return cls(
            allocated_mb=torch.cuda.memory_allocated(device) / (1024 * 1024),
            reserved_mb=torch.cuda.memory_reserved(device) / (1024 * 1024),
            max_allocated_mb=torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            num_allocs=stats.get("num_alloc_retries", 0) + stats.get("allocation.all.current", 0),
            num_frees=stats.get("num_ooms", 0),
        )


class MemoryAllocationTracker:
    """Track memory allocations during benchmark execution.
    
    Uses PyTorch's memory hooks to track allocations and detect
    suspicious patterns like pre-allocated outputs or memory leaks.
    
    Usage:
        tracker = MemoryAllocationTracker(device)
        tracker.start()
        # ... run benchmark ...
        tracker.stop()
        issues = tracker.check_patterns()
    """
    
    def __init__(self, device: Optional[Any] = None):
        self.device = device
        self.allocations: List[Dict[str, Any]] = []
        self.frees: List[Dict[str, Any]] = []
        self._hook_handle = None
        self.start_snapshot: Optional[MemoryAllocationSnapshot] = None
        self.end_snapshot: Optional[MemoryAllocationSnapshot] = None
    
    def start(self) -> None:
        """Start tracking memory allocations."""
        if torch is None or not torch.cuda.is_available():
            return
        
        self.allocations = []
        self.frees = []
        self.start_snapshot = MemoryAllocationSnapshot.capture(self.device)
        
        # Use memory snapshot for tracking (simpler than hooks)
        torch.cuda.reset_peak_memory_stats(self.device)
    
    def stop(self) -> None:
        """Stop tracking and capture final state."""
        if torch is None or not torch.cuda.is_available():
            return
        
        self.end_snapshot = MemoryAllocationSnapshot.capture(self.device)
    
    def check_patterns(
        self,
        max_memory_increase_mb: float = 100.0,
        max_peak_vs_end_ratio: float = 2.0,
    ) -> Tuple[bool, List[str]]:
        """Check for suspicious memory allocation patterns.
        
        Args:
            max_memory_increase_mb: Maximum allowed memory increase during benchmark
            max_peak_vs_end_ratio: Maximum allowed ratio of peak to final memory
            
        Returns:
            Tuple of (no_issues, list_of_warnings)
        """
        warnings_list: List[str] = []
        
        if self.start_snapshot is None or self.end_snapshot is None:
            return True, warnings_list
        
        # Check for memory increase (potential leak)
        memory_increase = self.end_snapshot.allocated_mb - self.start_snapshot.allocated_mb
        if memory_increase > max_memory_increase_mb:
            warnings_list.append(
                f"MEMORY INCREASE: Allocated memory increased by {memory_increase:.1f}MB "
                f"during benchmark (threshold: {max_memory_increase_mb}MB). "
                "This may indicate a memory leak or pre-allocated outputs."
            )
        
        # Check peak vs end ratio (could indicate temporary large allocations)
        if self.end_snapshot.allocated_mb > 0:
            peak_ratio = self.end_snapshot.max_allocated_mb / self.end_snapshot.allocated_mb
            if peak_ratio > max_peak_vs_end_ratio:
                warnings_list.append(
                    f"MEMORY PEAK SPIKE: Peak memory ({self.end_snapshot.max_allocated_mb:.1f}MB) "
                    f"was {peak_ratio:.1f}x final memory ({self.end_snapshot.allocated_mb:.1f}MB). "
                    "This may indicate inefficient memory usage."
                )
        
        # Check for potential pre-allocation (high starting memory that doesn't grow)
        if self.start_snapshot.allocated_mb > 100 and memory_increase < 10:
            warnings_list.append(
                f"POTENTIAL PRE-ALLOCATION: High initial memory ({self.start_snapshot.allocated_mb:.1f}MB) "
                f"with minimal growth ({memory_increase:.1f}MB). "
                "Verify outputs aren't pre-computed in setup()."
            )
        
        return len(warnings_list) == 0, warnings_list
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics summary."""
        if self.start_snapshot is None or self.end_snapshot is None:
            return {}
        
        return {
            "start_allocated_mb": self.start_snapshot.allocated_mb,
            "end_allocated_mb": self.end_snapshot.allocated_mb,
            "peak_allocated_mb": self.end_snapshot.max_allocated_mb,
            "memory_increase_mb": self.end_snapshot.allocated_mb - self.start_snapshot.allocated_mb,
        }


@contextmanager
def track_memory_allocations(device: Optional[Any] = None):
    """Context manager for memory allocation tracking.
    
    Example:
        with track_memory_allocations() as tracker:
            # ... run benchmark ...
        no_issues, warnings = tracker.check_patterns()
    """
    tracker = MemoryAllocationTracker(device)
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()


# =============================================================================
# CUDA Stream Auditing
# =============================================================================

@dataclass
class StreamUsageInfo:
    """Information about CUDA stream usage during benchmark."""
    default_stream_ops: int = 0
    custom_streams_detected: int = 0
    stream_ids: Set[int] = None
    sync_operations: int = 0
    unsync_warning: bool = False
    
    def __post_init__(self):
        if self.stream_ids is None:
            self.stream_ids = set()


class StreamAuditor:
    """Audits CUDA stream usage during benchmark execution.
    
    This helps detect multi-stream timing exploits (Locus/KernelBench 2025)
    where work is launched on non-default streams while the timer only
    measures the default stream.
    
    Usage:
        auditor = StreamAuditor()
        auditor.start()
        # ... run benchmark ...
        auditor.stop()
        info = auditor.get_info()
        warnings = auditor.check_issues()
    """
    
    def __init__(self, device: Optional[Any] = None):
        self.device = device
        self._start_time: Optional[float] = None
        self._stream_events: List[Dict[str, Any]] = []
        self._default_stream_id: Optional[int] = None
        self._observed_streams: Set[int] = set()
        self._sync_count: int = 0
        self._orig_stream_cls = None
        self._orig_synchronize = None
    
    def start(self) -> None:
        """Start stream auditing."""
        if torch is None or not torch.cuda.is_available():
            return
        
        import time
        self._start_time = time.perf_counter()
        self._stream_events = []
        self._observed_streams = set()
        self._sync_count = 0
        
        # Record default stream ID
        default_stream = torch.cuda.current_stream(self.device)
        self._default_stream_id = default_stream.cuda_stream
        self._observed_streams.add(self._default_stream_id)
        
        # Monkeypatch torch.cuda.Stream to record new stream creation
        try:
            self._orig_stream_cls = torch.cuda.Stream
            auditor = self
            
            class _AuditedStream(torch.cuda.Stream):  # type: ignore[misc]
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    auditor.record_stream_event(self, operation="stream_create")
            
            torch.cuda.Stream = _AuditedStream  # type: ignore[assignment]
        except Exception:
            self._orig_stream_cls = None
        
        # Monkeypatch synchronize to capture syncs
        try:
            self._orig_synchronize = torch.cuda.synchronize
            auditor = self
            
            def _audited_synchronize(*args, **kwargs):
                auditor.record_sync("device")
                return auditor._orig_synchronize(*args, **kwargs)
            
            torch.cuda.synchronize = _audited_synchronize  # type: ignore[assignment]
        except Exception:
            self._orig_synchronize = None
    
    def record_stream_event(self, stream: Any, operation: str = "kernel") -> None:
        """Record a stream event for auditing.
        
        Call this manually when a custom stream is used, or use hooks.
        """
        if torch is None or not torch.cuda.is_available():
            return
        
        import time
        stream_id = stream.cuda_stream if hasattr(stream, 'cuda_stream') else id(stream)
        self._observed_streams.add(stream_id)
        
        self._stream_events.append({
            "time": time.perf_counter() - (self._start_time or 0),
            "stream_id": stream_id,
            "operation": operation,
            "is_default": stream_id == self._default_stream_id,
        })
    
    def record_sync(self, sync_type: str = "device") -> None:
        """Record a synchronization event."""
        self._sync_count += 1
        
        import time
        self._stream_events.append({
            "time": time.perf_counter() - (self._start_time or 0),
            "stream_id": None,
            "operation": f"sync_{sync_type}",
            "is_default": None,
        })
    
    def stop(self) -> None:
        """Stop stream auditing."""
        if torch is None or not torch.cuda.is_available():
            return
        if self._orig_stream_cls is not None:
            try:
                torch.cuda.Stream = self._orig_stream_cls  # type: ignore[assignment]
            except Exception:
                pass
        if self._orig_synchronize is not None:
            try:
                torch.cuda.synchronize = self._orig_synchronize  # type: ignore[assignment]
            except Exception:
                pass
    
    def get_info(self) -> StreamUsageInfo:
        """Get stream usage information."""
        info = StreamUsageInfo()
        
        if self._default_stream_id is not None:
            info.default_stream_ops = sum(
                1 for e in self._stream_events 
                if e["is_default"] is True and e["operation"] != "sync_device"
            )
            
        info.custom_streams_detected = len(self._observed_streams) - 1  # Exclude default
        info.stream_ids = self._observed_streams.copy()
        info.sync_operations = self._sync_count
        
        # Check if custom streams were used without sync
        if info.custom_streams_detected > 0 and self._sync_count == 0:
            info.unsync_warning = True
        
        return info
    
    def check_issues(self) -> Tuple[bool, List[str]]:
        """Check for stream usage issues.
        
        Returns:
            Tuple of (no_issues, list_of_warnings)
        """
        warnings_list: List[str] = []
        info = self.get_info()
        
        # Warning: custom streams without synchronization
        if info.unsync_warning:
            warnings_list.append(
                f"STREAM SYNC WARNING: {info.custom_streams_detected} custom stream(s) "
                "detected but no device synchronization was recorded. "
                "This could allow work to escape timing measurement. "
                "Use torch.cuda.synchronize() for accurate multi-stream timing."
            )
        
        # Warning: many custom streams (unusual for benchmarks)
        if info.custom_streams_detected > 2:
            warnings_list.append(
                f"MULTI-STREAM WARNING: {info.custom_streams_detected} custom streams detected. "
                "Unusual for standard benchmarks - verify all streams are properly synchronized."
            )
        
        return len(warnings_list) == 0, warnings_list


@contextmanager
def audit_streams(device: Optional[Any] = None):
    """Context manager for CUDA stream auditing.
    
    Example:
        with audit_streams() as auditor:
            # ... run benchmark ...
        no_issues, warnings = auditor.check_issues()
    """
    auditor = StreamAuditor(device)
    auditor.start()
    try:
        yield auditor
    finally:
        auditor.stop()


def get_active_streams(device: Optional[Any] = None) -> List[int]:
    """Get list of active CUDA streams on a device.
    
    Note: This uses internal PyTorch APIs and may not be comprehensive.
    """
    if torch is None or not torch.cuda.is_available():
        return []
    
    streams = []
    
    # Get default stream
    default_stream = torch.cuda.current_stream(device)
    streams.append(default_stream.cuda_stream)
    
    return streams


def check_stream_sync_completeness(
    pre_streams: List[int],
    post_streams: List[int],
) -> Tuple[bool, Optional[str]]:
    """Check if stream synchronization was complete.
    
    Compares streams before and after an operation to detect
    potential timing issues.
    
    Args:
        pre_streams: Stream IDs before operation
        post_streams: Stream IDs after operation
        
    Returns:
        Tuple of (sync_complete, warning_message)
    """
    if not pre_streams or not post_streams:
        return True, None
    
    # If new streams appeared, that's suspicious
    new_streams = set(post_streams) - set(pre_streams)
    if new_streams:
        return False, (
            f"NEW STREAMS WARNING: {len(new_streams)} new stream(s) created during benchmark. "
            "Verify all streams are synchronized before timing ends."
        )
    
    return True, None


# =============================================================================
# Distributed Verification
# =============================================================================

@dataclass
class DistributedVerifyResult:
    """Result of distributed verification across ranks."""
    all_ranks_executed: bool
    outputs_consistent: bool
    rank_outputs: Dict[int, Any]  # rank -> output hash
    inconsistent_ranks: List[int]
    error_message: Optional[str] = None


def gather_rank_outputs(
    local_output: Any,
    world_size: int = 1,
    rank: int = 0,
) -> Dict[int, str]:
    """Gather output hashes from all ranks for consistency verification.
    
    In distributed settings, this uses torch.distributed to gather output
    checksums from all ranks to ensure they computed consistent results.
    
    Args:
        local_output: The local rank's output tensor(s)
        world_size: Total number of ranks
        rank: Current rank
        
    Returns:
        Dict mapping rank -> output hash (only complete on rank 0)
    """
    if torch is None:
        return {0: "no_torch"}
    
    # Compute local hash
    if isinstance(local_output, torch.Tensor):
        local_hash = hashlib.sha256(local_output.cpu().numpy().tobytes()).hexdigest()[:16]
    elif isinstance(local_output, dict):
        hasher = hashlib.sha256()
        for k in sorted(local_output.keys()):
            v = local_output[k]
            if isinstance(v, torch.Tensor):
                hasher.update(k.encode())
                hasher.update(v.cpu().numpy().tobytes())
        local_hash = hasher.hexdigest()[:16]
    else:
        local_hash = hashlib.sha256(str(local_output).encode()).hexdigest()[:16]
    
    # Single-GPU case
    if world_size == 1:
        return {0: local_hash}
    
    # Multi-GPU: gather hashes from all ranks
    try:
        import torch.distributed as dist
        
        if not dist.is_initialized():
            return {rank: local_hash}
        
        # Gather all hashes to rank 0
        all_hashes: List[str] = [None] * world_size  # type: ignore
        
        # Use all_gather_object for string data
        if hasattr(dist, 'all_gather_object'):
            dist.all_gather_object(all_hashes, local_hash)
            return {i: h for i, h in enumerate(all_hashes) if h is not None}
        else:
            # Fallback: just return local
            return {rank: local_hash}
            
    except Exception:
        # Distributed not available or failed
        return {rank: local_hash}


def verify_distributed_outputs(
    rank_outputs: Dict[int, str],
    expected_world_size: int,
) -> DistributedVerifyResult:
    """Verify that all ranks executed and produced consistent outputs.
    
    This detects:
    - Rank skipping: Some ranks didn't execute work
    - Inconsistent outputs: Ranks produced different results (when they shouldn't)
    
    Args:
        rank_outputs: Dict mapping rank -> output hash
        expected_world_size: Expected number of ranks
        
    Returns:
        DistributedVerifyResult with verification outcome
    """
    result = DistributedVerifyResult(
        all_ranks_executed=True,
        outputs_consistent=True,
        rank_outputs=rank_outputs,
        inconsistent_ranks=[],
    )
    
    # Check all ranks reported
    if len(rank_outputs) != expected_world_size:
        result.all_ranks_executed = False
        missing_ranks = set(range(expected_world_size)) - set(rank_outputs.keys())
        result.error_message = f"RANK SKIPPING: Missing outputs from ranks {sorted(missing_ranks)}"
        return result
    
    # Check outputs are consistent
    unique_hashes = set(rank_outputs.values())
    if len(unique_hashes) > 1:
        result.outputs_consistent = False
        
        # Find the majority hash (assumed correct)
        hash_counts: Dict[str, int] = {}
        for h in rank_outputs.values():
            hash_counts[h] = hash_counts.get(h, 0) + 1
        majority_hash = max(hash_counts, key=lambda x: hash_counts[x])
        
        # Find inconsistent ranks
        result.inconsistent_ranks = [
            r for r, h in rank_outputs.items() if h != majority_hash
        ]
        result.error_message = (
            f"OUTPUT INCONSISTENCY: Ranks {result.inconsistent_ranks} produced different outputs. "
            f"Majority hash: {majority_hash}, inconsistent: "
            f"{[rank_outputs[r] for r in result.inconsistent_ranks]}"
        )
    
    return result


def check_rank_execution(
    benchmark: Any,
    world_size: int,
    rank: int,
) -> Tuple[bool, Optional[str]]:
    """Check if a benchmark properly executed on the current rank.
    
    This is a lightweight check to detect rank skipping by verifying
    that the benchmark actually performed computation on this rank.
    
    Args:
        benchmark: The benchmark instance
        world_size: Total number of ranks
        rank: Current rank
        
    Returns:
        Tuple of (executed, error_message)
    """
    # Check for explicit rank skip flags
    if hasattr(benchmark, '_skip_rank') and benchmark._skip_rank:
        return False, f"Rank {rank} has _skip_rank=True"
    
    # Check if benchmark has output (indicates execution)
    if hasattr(benchmark, 'get_verify_output'):
        try:
            output = benchmark.get_verify_output()
            if output is None:
                return False, f"Rank {rank} produced no output"
            return True, None
        except NotImplementedError:
            # Benchmark doesn't implement output, can't verify
            return True, None
        except Exception as e:
            return False, f"Rank {rank} failed to get output: {e}"
    
    return True, None


# =============================================================================
# CUDA Graph Capture Cheat Detection
# =============================================================================

@dataclass
class GraphCaptureState:
    """State tracking for CUDA graph capture detection."""
    capturing: bool = False
    capture_start_time: Optional[float] = None
    capture_end_time: Optional[float] = None
    kernels_during_capture: int = 0
    memory_allocated_during_capture: float = 0.0
    work_detected_during_capture: bool = False


class GraphCaptureCheatDetector:
    """Detects if work is being done during CUDA graph capture instead of replay.
    
    The "graph capture cheat" is when a benchmark does actual computation
    during the graph capture phase (which happens once) rather than during
    graph replay (which is what's being timed). This makes the benchmark
    appear faster than it actually is.
    
    Detection Strategy:
    1. Track memory allocations during capture vs replay
    2. Track GPU utilization during capture
    3. Compare capture time to replay time (capture should be similar or longer)
    
    Usage:
        detector = GraphCaptureCheatDetector()
        
        # During capture
        detector.start_capture()
        with torch.cuda.graph(g):
            benchmark_fn()  # Capture the work
        detector.end_capture()
        
        # During replay (timed)
        detector.start_replay()
        g.replay()
        detector.end_replay()
        
        # Check for cheating
        is_cheating, reason = detector.check_for_cheat()
    """
    
    def __init__(self, device: Optional[Any] = None):
        self.device = device
        self.capture_state: Optional[GraphCaptureState] = None
        self.replay_times: List[float] = []
        self._capture_start_memory: float = 0.0
        self._replay_start_memory: float = 0.0
    
    def start_capture(self) -> None:
        """Mark the start of CUDA graph capture."""
        import time
        
        self.capture_state = GraphCaptureState(capturing=True)
        self.capture_state.capture_start_time = time.perf_counter()
        
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            self._capture_start_memory = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
    
    def end_capture(self) -> None:
        """Mark the end of CUDA graph capture."""
        import time
        
        if self.capture_state is None:
            return
        
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        
        self.capture_state.capture_end_time = time.perf_counter()
        self.capture_state.capturing = False
        
        if torch is not None and torch.cuda.is_available():
            capture_end_memory = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            self.capture_state.memory_allocated_during_capture = (
                capture_end_memory - self._capture_start_memory
            )
    
    def start_replay(self) -> None:
        """Mark the start of a CUDA graph replay iteration."""
        if torch is not None and torch.cuda.is_available():
            self._replay_start_memory = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
    
    def end_replay(self, replay_time_ms: float) -> None:
        """Record a replay iteration time.
        
        Args:
            replay_time_ms: Time for this replay iteration in milliseconds
        """
        self.replay_times.append(replay_time_ms)
    
    def check_for_cheat(
        self,
        capture_replay_ratio_threshold: float = 10.0,
        memory_threshold_mb: float = 100.0,
    ) -> Tuple[bool, Optional[str]]:
        """Check if the benchmark is cheating via graph capture.
        
        Signs of cheating:
        1. Capture time >> replay time (work done during capture, not replay)
        2. Large memory allocation during capture
        3. Replay times are suspiciously fast (near-zero)
        
        Args:
            capture_replay_ratio_threshold: Max allowed ratio of capture/replay time
            memory_threshold_mb: Memory allocation during capture that triggers warning
            
        Returns:
            Tuple of (is_cheating, reason_if_cheating)
        """
        if self.capture_state is None:
            return False, None
        
        warnings_list: List[str] = []
        
        # Check 1: Capture/replay time ratio
        if (self.capture_state.capture_start_time is not None and 
            self.capture_state.capture_end_time is not None and
            self.replay_times):
            
            capture_time_ms = (
                self.capture_state.capture_end_time - 
                self.capture_state.capture_start_time
            ) * 1000
            
            avg_replay_time_ms = sum(self.replay_times) / len(self.replay_times)
            
            if avg_replay_time_ms > 0:
                ratio = capture_time_ms / avg_replay_time_ms
                if ratio > capture_replay_ratio_threshold:
                    warnings_list.append(
                        f"GRAPH CAPTURE CHEAT SUSPECTED: Capture time ({capture_time_ms:.2f}ms) "
                        f"is {ratio:.1f}x longer than replay time ({avg_replay_time_ms:.2f}ms). "
                        "This suggests work is being done during capture, not replay."
                    )
        
        # Check 2: Memory allocation during capture
        if self.capture_state.memory_allocated_during_capture > memory_threshold_mb:
            warnings_list.append(
                f"SUSPICIOUS MEMORY ALLOCATION: {self.capture_state.memory_allocated_during_capture:.1f}MB "
                f"allocated during graph capture. This may indicate computation during capture."
            )
        
        # Check 3: Near-zero replay times
        if self.replay_times:
            min_replay = min(self.replay_times)
            if min_replay < 0.001:  # Less than 1 microsecond
                warnings_list.append(
                    f"SUSPICIOUS REPLAY TIME: Minimum replay time ({min_replay*1000:.3f}μs) "
                    "is near-zero. Graph may be empty or doing no actual work."
                )
        
        if warnings_list:
            return True, " | ".join(warnings_list)
        
        return False, None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get capture/replay statistics."""
        if self.capture_state is None:
            return {}
        
        capture_time_ms = 0.0
        if (self.capture_state.capture_start_time is not None and 
            self.capture_state.capture_end_time is not None):
            capture_time_ms = (
                self.capture_state.capture_end_time - 
                self.capture_state.capture_start_time
            ) * 1000
        
        return {
            "capture_time_ms": capture_time_ms,
            "capture_memory_mb": self.capture_state.memory_allocated_during_capture,
            "replay_count": len(self.replay_times),
            "avg_replay_time_ms": sum(self.replay_times) / len(self.replay_times) if self.replay_times else 0,
            "min_replay_time_ms": min(self.replay_times) if self.replay_times else 0,
            "max_replay_time_ms": max(self.replay_times) if self.replay_times else 0,
        }


@contextmanager
def detect_graph_capture_cheat(device: Optional[Any] = None):
    """Context manager for graph capture cheat detection.
    
    Example:
        with detect_graph_capture_cheat() as detector:
            detector.start_capture()
            with torch.cuda.graph(g):
                fn()
            detector.end_capture()
            
            for _ in range(10):
                detector.start_replay()
                g.replay()
                # Record replay time manually
                detector.end_replay(elapsed_ms)
        
        is_cheating, reason = detector.check_for_cheat()
    """
    detector = GraphCaptureCheatDetector(device)
    try:
        yield detector
    finally:
        pass  # No cleanup needed


def check_graph_capture_integrity(
    capture_time_ms: float,
    replay_times_ms: List[float],
    memory_during_capture_mb: float = 0.0,
) -> Tuple[bool, Optional[str]]:
    """Quick check for graph capture integrity without full detector.
    
    Args:
        capture_time_ms: Time to capture the graph
        replay_times_ms: List of replay iteration times
        memory_during_capture_mb: Memory allocated during capture
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not replay_times_ms:
        return True, None
    
    avg_replay = sum(replay_times_ms) / len(replay_times_ms)
    
    # Check for suspicious capture/replay ratio
    if avg_replay > 0:
        ratio = capture_time_ms / avg_replay
        if ratio > 10.0:
            return False, (
                f"Graph capture time ({capture_time_ms:.2f}ms) is {ratio:.1f}x "
                f"replay time ({avg_replay:.2f}ms). Suspected work during capture."
            )
    
    # Check for near-zero replays
    if min(replay_times_ms) < 0.001:
        return False, "Replay times are near-zero, graph may be empty."
    
    return True, None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # GPU State
    "GPUState",
    "capture_gpu_state", 
    "check_gpu_state_consistency",
    
    # Memory Checks
    "get_tensor_addresses",
    "check_input_output_aliasing",
    "reset_cuda_memory_pool",
    
    # Memory Allocation Tracking
    "MemoryAllocationSnapshot",
    "MemoryAllocationTracker",
    "track_memory_allocations",
    
    # Setup Checks
    "hash_tensors",
    "check_setup_precomputation",
    
    # GC Control
    "gc_disabled",
    
    # Compile Checks
    "clear_compile_cache",
    "get_compile_state",
    
    # Lazy Evaluation
    "force_tensor_evaluation",
    
    # Environment
    "validate_environment",
    
    # Stream Auditing
    "StreamUsageInfo",
    "StreamAuditor",
    "audit_streams",
    "get_active_streams",
    "check_stream_sync_completeness",
    
    # Distributed Verification
    "DistributedVerifyResult",
    "gather_rank_outputs",
    "verify_distributed_outputs",
    "check_rank_execution",
    
    # Graph Capture Cheat Detection
    "GraphCaptureState",
    "GraphCaptureCheatDetector",
    "detect_graph_capture_cheat",
    "check_graph_capture_integrity",
]
