"""Quarantine management for non-compliant benchmarks.

This module manages the quarantine state for benchmarks that fail verification
or lack required verification methods. Quarantined benchmarks are excluded
from performance reports and comparisons.

Key Features:
- Persistent storage of quarantine records
- Skip flag detection
- Automatic quarantine clearing on successful verify pass
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from core.benchmark.verification import QuarantineReason, QuarantineRecord


# Default quarantine cache location
DEFAULT_QUARANTINE_PATH = Path("artifacts/verify_cache/quarantine.json")

# Skip flags that trigger automatic quarantine
SKIP_FLAGS = frozenset({
    "skip_output_check",
    "skip_input_check", 
    "skip_verification",
})


class QuarantineManager:
    """Manages quarantine state for benchmarks.
    
    Quarantined benchmarks are excluded from performance reports. The manager
    persists quarantine records to disk and provides methods for querying
    and modifying quarantine state.
    
    Usage:
        manager = QuarantineManager()
        
        # Quarantine a benchmark
        manager.quarantine("ch01/baseline_gemm.py", QuarantineReason.MISSING_VALIDATE_RESULT)
        
        # Check if quarantined
        if manager.is_quarantined("ch01/baseline_gemm.py"):
            print("Benchmark is quarantined")
            
        # Clear quarantine on successful verify
        manager.clear_quarantine("ch01/baseline_gemm.py")
    """
    
    def __init__(
        self,
        quarantine_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the QuarantineManager.
        
        Args:
            quarantine_path: Path to the quarantine JSON file (takes precedence).
            cache_dir: Directory containing quarantine.json. Used if quarantine_path
                      is not specified.
        """
        if quarantine_path:
            self.quarantine_path = quarantine_path
        elif cache_dir:
            self.quarantine_path = cache_dir / "quarantine.json"
        else:
            self.quarantine_path = DEFAULT_QUARANTINE_PATH
        self._records: Dict[str, QuarantineRecord] = {}
        self._load()
    
    def _load(self) -> None:
        """Load quarantine records from disk."""
        if not self.quarantine_path.exists():
            return
            
        try:
            data = json.loads(self.quarantine_path.read_text())
            records = data.get("records", {})
            for path, record_dict in records.items():
                self._records[path] = QuarantineRecord.from_dict(record_dict)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If file is corrupted, start fresh
            import warnings
            warnings.warn(f"Failed to load quarantine file: {e}. Starting fresh.")
            self._records = {}
    
    def _persist(self) -> None:
        """Persist quarantine records to disk."""
        # Ensure directory exists
        self.quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "schema_version": 1,
            "updated_at": datetime.now().isoformat(),
            "records": {
                path: record.to_dict()
                for path, record in self._records.items()
            },
        }
        
        self.quarantine_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    
    def quarantine(
        self,
        benchmark_path: str,
        reason: QuarantineReason,
        details: Optional[Dict[str, Any]] = None,
    ) -> QuarantineRecord:
        """Mark a benchmark as quarantined.
        
        Args:
            benchmark_path: Path to the benchmark file (relative to repo root)
            reason: The reason for quarantine
            details: Optional additional details about the quarantine
            
        Returns:
            The created QuarantineRecord
        """
        record = QuarantineRecord(
            benchmark_path=benchmark_path,
            quarantine_reason=reason,
            quarantine_timestamp=datetime.now(),
            details=details,
        )
        self._records[benchmark_path] = record
        self._persist()
        return record
    
    def is_quarantined(self, benchmark_path: str) -> bool:
        """Check if a benchmark is quarantined.
        
        Args:
            benchmark_path: Path to the benchmark file
            
        Returns:
            True if the benchmark is quarantined
        """
        return benchmark_path in self._records
    
    def get_quarantine_record(self, benchmark_path: str) -> Optional[QuarantineRecord]:
        """Get the quarantine record for a benchmark.
        
        Args:
            benchmark_path: Path to the benchmark file
            
        Returns:
            QuarantineRecord if quarantined, None otherwise
        """
        return self._records.get(benchmark_path)
    
    def clear_quarantine(self, benchmark_path: str) -> bool:
        """Remove quarantine status from a benchmark.
        
        Called automatically when a benchmark passes verification.
        
        Args:
            benchmark_path: Path to the benchmark file
            
        Returns:
            True if the benchmark was quarantined and is now cleared
        """
        if benchmark_path in self._records:
            del self._records[benchmark_path]
            self._persist()
            return True
        return False
    
    def get_all_quarantined(self) -> List[QuarantineRecord]:
        """Get all quarantined benchmarks.
        
        Returns:
            List of all QuarantineRecords
        """
        return list(self._records.values())
    
    def get_all_records(self) -> Dict[str, QuarantineRecord]:
        """Get all quarantine records as a dictionary.
        
        Returns:
            Dict mapping benchmark paths to their QuarantineRecords
        """
        return dict(self._records)
    
    def get_quarantined_by_reason(self, reason: QuarantineReason) -> List[QuarantineRecord]:
        """Get all benchmarks quarantined for a specific reason.
        
        Args:
            reason: The quarantine reason to filter by
            
        Returns:
            List of QuarantineRecords matching the reason
        """
        return [
            record for record in self._records.values()
            if record.quarantine_reason == reason
        ]
    
    def get_quarantine_summary(self) -> Dict[str, int]:
        """Get a summary of quarantine counts by reason.
        
        Returns:
            Dict mapping reason names to counts
        """
        summary: Dict[str, int] = {}
        for record in self._records.values():
            reason_name = record.quarantine_reason.value
            summary[reason_name] = summary.get(reason_name, 0) + 1
        return summary
    
    def clear_all(self) -> int:
        """Clear all quarantine records.
        
        Returns:
            Number of records cleared
        """
        count = len(self._records)
        self._records = {}
        self._persist()
        return count


def detect_skip_flags(benchmark: Any) -> Optional[QuarantineReason]:
    """Detect if a benchmark has skip flags that trigger quarantine.
    
    Checks for skip_output_check, skip_input_check, and skip_verification
    attributes. These are legacy flags that bypass verification and are
    treated as non-compliant.
    
    Args:
        benchmark: The benchmark instance to check
        
    Returns:
        QuarantineReason.SKIP_FLAG_PRESENT if skip flags found, None otherwise
    """
    for flag in SKIP_FLAGS:
        # Check both attribute and method
        if hasattr(benchmark, flag):
            value = getattr(benchmark, flag)
            # Handle both boolean attributes and methods
            if callable(value):
                try:
                    if value():
                        return QuarantineReason.SKIP_FLAG_PRESENT
                except Exception:
                    pass
            elif value:
                return QuarantineReason.SKIP_FLAG_PRESENT
    
    # Also check the methods in BaseBenchmark
    if hasattr(benchmark, "skip_input_verification"):
        try:
            if benchmark.skip_input_verification():
                return QuarantineReason.SKIP_FLAG_PRESENT
        except Exception:
            pass
            
    if hasattr(benchmark, "skip_output_verification"):
        try:
            if benchmark.skip_output_verification():
                return QuarantineReason.SKIP_FLAG_PRESENT
        except Exception:
            pass
    
    return None


def check_benchmark_compliance(benchmark: Any) -> List[QuarantineReason]:
    """Check a benchmark for all compliance issues.
    
    Performs comprehensive compliance checking including:
    - Skip flag detection
    - Required method presence (get_input_signature, validate_result)
    - Workload metadata availability
    
    Args:
        benchmark: The benchmark instance to check
        
    Returns:
        List of QuarantineReasons for any compliance issues found (empty if compliant)
    """
    issues: List[QuarantineReason] = []
    
    # Check for skip flags
    skip_flag = detect_skip_flags(benchmark)
    if skip_flag:
        issues.append(skip_flag)
    
    # Check for required methods
    if not hasattr(benchmark, "get_input_signature") or not callable(getattr(benchmark, "get_input_signature")):
        issues.append(QuarantineReason.MISSING_INPUT_SIGNATURE)
    else:
        # Try calling it to ensure it returns a valid signature
        try:
            sig = benchmark.get_input_signature()
            if sig is None or (isinstance(sig, dict) and not sig):
                issues.append(QuarantineReason.MISSING_INPUT_SIGNATURE)
        except Exception:
            issues.append(QuarantineReason.MISSING_INPUT_SIGNATURE)
    
    if not hasattr(benchmark, "get_verify_output") or not callable(getattr(benchmark, "get_verify_output")):
        issues.append(QuarantineReason.MISSING_VERIFY_OUTPUT)
    else:
        try:
            output = benchmark.get_verify_output()
            if output is None:
                issues.append(QuarantineReason.MISSING_VERIFY_OUTPUT)
        except Exception:
            issues.append(QuarantineReason.MISSING_VERIFY_OUTPUT)
    
    if not hasattr(benchmark, "get_output_tolerance") or not callable(getattr(benchmark, "get_output_tolerance")):
        issues.append(QuarantineReason.MISSING_OUTPUT_TOLERANCE)
    else:
        try:
            tol = benchmark.get_output_tolerance()
            if tol is None:
                issues.append(QuarantineReason.MISSING_OUTPUT_TOLERANCE)
        except Exception:
            issues.append(QuarantineReason.MISSING_OUTPUT_TOLERANCE)
    
    if not hasattr(benchmark, "get_verify_inputs") or not callable(getattr(benchmark, "get_verify_inputs")):
        issues.append(QuarantineReason.MISSING_VERIFY_INPUTS)
    else:
        try:
            inputs = benchmark.get_verify_inputs()
            if inputs is None:
                issues.append(QuarantineReason.MISSING_VERIFY_INPUTS)
        except Exception:
            issues.append(QuarantineReason.MISSING_VERIFY_INPUTS)
    
    if not hasattr(benchmark, "validate_result") or not callable(getattr(benchmark, "validate_result")):
        issues.append(QuarantineReason.MISSING_VALIDATE_RESULT)
    
    # Check for workload metadata
    if hasattr(benchmark, "get_workload_metadata"):
        try:
            metadata = benchmark.get_workload_metadata()
            if metadata is None:
                issues.append(QuarantineReason.MISSING_WORKLOAD_METADATA)
        except Exception:
            issues.append(QuarantineReason.MISSING_WORKLOAD_METADATA)
    else:
        issues.append(QuarantineReason.MISSING_WORKLOAD_METADATA)
    
    return issues
