"""
Memory Profiler - Track GPU memory usage over time.

Features:
- Memory timeline tracking
- Allocation/deallocation events
- Peak memory analysis
- Memory leak detection
- Snapshot export for visualization
"""

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


@dataclass
class AllocationEvent:
    """A single memory allocation or deallocation event."""
    timestamp_ms: float
    size_bytes: int
    is_allocation: bool
    address: int = 0
    device_index: int = 0
    stream: int = 0
    stack_trace: str = ""


@dataclass
class MemorySnapshot:
    """Complete memory state at a point in time."""
    timestamp_ms: float
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    
    # Breakdown by category
    activations_bytes: int = 0
    weights_bytes: int = 0
    gradients_bytes: int = 0
    optimizer_bytes: int = 0
    other_bytes: int = 0
    
    # Active allocations
    active_allocations: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "allocated_mb": self.allocated_bytes / 1024 / 1024,
            "reserved_mb": self.reserved_bytes / 1024 / 1024,
            "peak_allocated_mb": self.peak_allocated_bytes / 1024 / 1024,
            "breakdown": {
                "activations_mb": self.activations_bytes / 1024 / 1024,
                "weights_mb": self.weights_bytes / 1024 / 1024,
                "gradients_mb": self.gradients_bytes / 1024 / 1024,
                "optimizer_mb": self.optimizer_bytes / 1024 / 1024,
                "other_mb": self.other_bytes / 1024 / 1024,
            }
        }


class MemoryProfiler:
    """
    Track and analyze GPU memory usage.
    
    Usage:
        profiler = MemoryProfiler()
        
        with profiler.track("forward_pass"):
            output = model(input)
        
        timeline = profiler.get_timeline()
        profiler.export_snapshot("memory.json")
    """
    
    def __init__(
        self,
        sample_interval_ms: float = 1.0,
        record_history: bool = True,
        track_allocations: bool = True,
    ):
        self.sample_interval_ms = sample_interval_ms
        self.record_history = record_history
        self.track_allocations = track_allocations
        
        self._timeline: List[MemorySnapshot] = []
        self._events: List[AllocationEvent] = []
        self._start_time: float = 0
        self._tracking: bool = False
        self._markers: Dict[str, Tuple[float, float]] = {}
    
    def track(self, name: str = "default"):
        """Context manager for tracking a code section."""
        return _MemoryTrackContext(self, name)
    
    def start_tracking(self, name: str = "default"):
        """Start tracking memory."""
        if not torch.cuda.is_available():
            return
        
        self._tracking = True
        self._start_time = time.perf_counter() * 1000
        
        # Reset stats
        torch.cuda.reset_peak_memory_stats()
        
        # Enable memory history recording
        if self.record_history:
            try:
                torch.cuda.memory._record_memory_history(
                    max_entries=100000,
                    context="all",
                )
            except Exception:
                pass
        
        # Record initial snapshot
        self._take_snapshot(name=f"{name}_start")
        self._markers[name] = (self._start_time, 0)
    
    def stop_tracking(self, name: str = "default"):
        """Stop tracking memory."""
        if not torch.cuda.is_available() or not self._tracking:
            return
        
        self._tracking = False
        end_time = time.perf_counter() * 1000
        
        # Record final snapshot
        self._take_snapshot(name=f"{name}_end")
        
        # Update marker
        if name in self._markers:
            self._markers[name] = (self._markers[name][0], end_time)
        
        # Stop memory history
        if self.record_history:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass
    
    def _take_snapshot(self, name: str = ""):
        """Take a memory snapshot."""
        if not torch.cuda.is_available():
            return
        
        torch.cuda.synchronize()
        
        current_time = time.perf_counter() * 1000 - self._start_time
        
        snapshot = MemorySnapshot(
            timestamp_ms=current_time,
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        
        self._timeline.append(snapshot)
    
    def sample(self):
        """Take a memory sample (for timeline building)."""
        if self._tracking:
            self._take_snapshot()
    
    def get_timeline(self) -> List[Dict[str, Any]]:
        """Get memory timeline as list of dicts for visualization."""
        return [
            {
                "time_ms": s.timestamp_ms,
                "allocated_mb": s.allocated_bytes / 1024 / 1024,
                "reserved_mb": s.reserved_bytes / 1024 / 1024,
                "peak_mb": s.peak_allocated_bytes / 1024 / 1024,
            }
            for s in self._timeline
        ]
    
    def get_peak_analysis(self) -> Dict[str, Any]:
        """Analyze peak memory usage."""
        if not self._timeline:
            return {}
        
        peak_snapshot = max(self._timeline, key=lambda s: s.allocated_bytes)
        
        return {
            "peak_allocated_mb": peak_snapshot.allocated_bytes / 1024 / 1024,
            "peak_reserved_mb": peak_snapshot.reserved_bytes / 1024 / 1024,
            "peak_time_ms": peak_snapshot.timestamp_ms,
            "current_allocated_mb": self._timeline[-1].allocated_bytes / 1024 / 1024 if self._timeline else 0,
            "potential_leak": (
                self._timeline[-1].allocated_bytes > self._timeline[0].allocated_bytes * 1.1
                if len(self._timeline) > 1 else False
            ),
        }
    
    def get_memory_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get detailed memory snapshot from PyTorch."""
        if not torch.cuda.is_available():
            return None
        
        try:
            snapshot = torch.cuda.memory._snapshot()
            if snapshot:
                return self._parse_memory_snapshot(snapshot)
        except Exception:
            pass
        
        return None
    
    def _parse_memory_snapshot(self, snapshot: Dict) -> Dict[str, Any]:
        """Parse PyTorch memory snapshot into visualization-friendly format."""
        
        result = {
            "segments": [],
            "total_allocated_mb": 0,
            "total_reserved_mb": 0,
            "allocations_by_size": {},
        }
        
        try:
            for segment in snapshot.get('segments', []):
                seg_info = {
                    "address": segment.get('address', 0),
                    "size_mb": segment.get('total_size', 0) / 1024 / 1024,
                    "allocated_size_mb": segment.get('allocated_size', 0) / 1024 / 1024,
                    "device": segment.get('device', 0),
                }
                result["segments"].append(seg_info)
                result["total_reserved_mb"] += seg_info["size_mb"]
                result["total_allocated_mb"] += seg_info["allocated_size_mb"]
            
            # Categorize allocations by size
            for trace in snapshot.get('traces', []):
                size = trace.get('size', 0)
                if size < 1024:
                    bucket = "< 1KB"
                elif size < 1024 * 1024:
                    bucket = "1KB - 1MB"
                elif size < 100 * 1024 * 1024:
                    bucket = "1MB - 100MB"
                else:
                    bucket = "> 100MB"
                
                result["allocations_by_size"][bucket] = result["allocations_by_size"].get(bucket, 0) + 1
                
        except Exception:
            pass
        
        return result
    
    def export(self, path: Path, format: str = "json"):
        """Export memory data to file."""
        path = Path(path)
        
        data = {
            "timeline": self.get_timeline(),
            "peak_analysis": self.get_peak_analysis(),
            "markers": {
                name: {"start_ms": start, "end_ms": end}
                for name, (start, end) in self._markers.items()
            },
        }
        
        if format == "json":
            path.write_text(json.dumps(data, indent=2))
        elif format == "pickle":
            with open(path, 'wb') as f:
                pickle.dump(data, f)
    
    def reset(self):
        """Reset the profiler state."""
        self._timeline = []
        self._events = []
        self._markers = {}
        self._tracking = False


class _MemoryTrackContext:
    """Context manager for memory tracking."""
    
    def __init__(self, profiler: MemoryProfiler, name: str):
        self.profiler = profiler
        self.name = name
    
    def __enter__(self):
        self.profiler.start_tracking(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.stop_tracking(self.name)
        return False


def profile_memory(
    fn: Callable,
    *args,
    warmup: int = 3,
    iterations: int = 10,
    **kwargs,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Profile memory usage of a function.
    
    Args:
        fn: Function to profile
        *args: Arguments to pass to function
        warmup: Warmup iterations
        iterations: Profile iterations
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Tuple of (function result, memory stats)
    """
    profiler = MemoryProfiler()
    
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Profile
    with profiler.track("profile"):
        for i in range(iterations):
            result = fn(*args, **kwargs)
            profiler.sample()
    
    return result, {
        "timeline": profiler.get_timeline(),
        "peak": profiler.get_peak_analysis(),
    }

