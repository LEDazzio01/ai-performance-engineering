"""Property-based tests for benchmark expectations integrity.

These tests verify the correctness properties defined in the design document
for the benchmark expectations system.

Requires: hypothesis (pip install hypothesis)
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

# Try to import hypothesis; skip tests if not available
try:
    from hypothesis import HealthCheck, given, settings, strategies as st, assume

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators so module can load
    def given(*args, **kwargs):
        def decorator(fn):
            return pytest.mark.skip(reason="hypothesis not installed")(fn)
        return decorator

    def settings(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    class st:  # type: ignore
        @staticmethod
        def floats(*args, **kwargs):
            return None

        @staticmethod
        def text(*args, **kwargs):
            return None

        @staticmethod
        def integers(*args, **kwargs):
            return None

        @staticmethod
        def booleans():
            return None

    def assume(x):
        pass

    class HealthCheck:  # type: ignore
        too_slow = object()


from core.benchmark.expectations import (
    SPEEDUP_TOLERANCE,
    ExpectationEntry,
    ExpectationsStore,
    RunProvenance,
    UpdateResult,
    ValidationIssue,
    ValidationReport,
)


# =============================================================================
# Test Strategies for Property-Based Testing
# =============================================================================


def valid_time_ms():
    """Generate valid timing values (positive, finite floats)."""
    return st.floats(min_value=0.001, max_value=100000.0, allow_nan=False, allow_infinity=False)


def valid_provenance():
    """Generate valid RunProvenance instances."""
    return st.builds(
        RunProvenance,
        git_commit=st.text(min_size=7, max_size=40, alphabet="0123456789abcdef"),
        hardware_key=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_"),
        profile_name=st.sampled_from(["minimal", "deep_dive", "roofline", "none"]),
        timestamp=st.just(datetime.now().isoformat()),
        iterations=st.integers(min_value=1, max_value=1000),
        warmup_iterations=st.integers(min_value=0, max_value=100),
    )


# =============================================================================
# Property 1: Speedup-Timing Consistency
# Validates: Requirements 1.1, 1.2
# =============================================================================


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestSpeedupTimingConsistency:
    """Property 1: For any expectation entry with valid timing values,
    the stored best_speedup SHALL equal baseline_time_ms / best_optimized_time_ms
    within floating-point tolerance (1e-6).
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        baseline_time=valid_time_ms(),
        optimized_time=valid_time_ms(),
        provenance=valid_provenance(),
    )
    def test_speedup_equals_timing_ratio(
        self, baseline_time: float, optimized_time: float, provenance: RunProvenance
    ):
        """Speedup must always equal the ratio of timing values."""
        # Create entry with the given timing values
        entry = ExpectationEntry(
            example="test_example",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=baseline_time,
            best_optimized_time_ms=optimized_time,
            provenance=provenance,
        )

        # Compute expected speedup
        expected_speedup = baseline_time / optimized_time

        # Verify property: speedup equals computed ratio within tolerance
        assert abs(entry.best_speedup - expected_speedup) < SPEEDUP_TOLERANCE, (
            f"Speedup {entry.best_speedup} does not match expected ratio "
            f"{expected_speedup} (baseline={baseline_time}, optimized={optimized_time})"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        baseline_time=valid_time_ms(),
        optimized_time=valid_time_ms(),
        provenance=valid_provenance(),
    )
    def test_serialized_speedup_matches_computed(
        self, baseline_time: float, optimized_time: float, provenance: RunProvenance
    ):
        """Serialized speedup in to_dict() must match computed property."""
        entry = ExpectationEntry(
            example="test_example",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=baseline_time,
            best_optimized_time_ms=optimized_time,
            provenance=provenance,
        )

        serialized = entry.to_dict()
        stored_speedup = serialized["metrics"]["best_speedup"]

        # Verify: serialized speedup equals computed property
        assert abs(stored_speedup - entry.best_speedup) < SPEEDUP_TOLERANCE, (
            f"Serialized speedup {stored_speedup} differs from computed {entry.best_speedup}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        baseline_time=valid_time_ms(),
        optimized_time=valid_time_ms(),
        provenance=valid_provenance(),
    )
    def test_roundtrip_preserves_speedup_consistency(
        self, baseline_time: float, optimized_time: float, provenance: RunProvenance
    ):
        """Serialization roundtrip must preserve speedup-timing consistency."""
        original = ExpectationEntry(
            example="test_example",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=baseline_time,
            best_optimized_time_ms=optimized_time,
            provenance=provenance,
        )

        # Serialize and deserialize
        serialized = original.to_dict()
        restored = ExpectationEntry.from_dict(serialized)

        # Verify: restored entry has same timing values
        assert abs(restored.baseline_time_ms - original.baseline_time_ms) < SPEEDUP_TOLERANCE
        assert abs(restored.best_optimized_time_ms - original.best_optimized_time_ms) < SPEEDUP_TOLERANCE

        # Verify: restored speedup still matches timing ratio
        expected_speedup = restored.baseline_time_ms / restored.best_optimized_time_ms
        assert abs(restored.best_speedup - expected_speedup) < SPEEDUP_TOLERANCE


# =============================================================================
# Property 2: Regression Visibility
# Validates: Requirements 2.1, 2.2, 2.4
# =============================================================================


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestRegressionVisibility:
    """Property 2: For any expectation entry where baseline_time_ms / best_optimized_time_ms < 1.0,
    the stored best_speedup SHALL be less than 1.0 (not clamped) and is_regression SHALL be true.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        baseline_time=valid_time_ms(),
        optimized_time=valid_time_ms(),
        provenance=valid_provenance(),
    )
    def test_regression_when_optimized_slower(
        self, baseline_time: float, optimized_time: float, provenance: RunProvenance
    ):
        """When optimized is slower than baseline, is_regression must be True."""
        # Only test regression cases: optimized time > baseline time
        assume(optimized_time > baseline_time)

        entry = ExpectationEntry(
            example="test_example",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=baseline_time,
            best_optimized_time_ms=optimized_time,
            provenance=provenance,
        )

        # Verify: speedup < 1.0 for regression
        assert entry.best_speedup < 1.0, (
            f"Speedup {entry.best_speedup} should be < 1.0 when optimized "
            f"({optimized_time}ms) is slower than baseline ({baseline_time}ms)"
        )

        # Verify: is_regression flag is True
        assert entry.is_regression is True, (
            f"is_regression should be True when speedup is {entry.best_speedup}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        baseline_time=valid_time_ms(),
        optimized_time=valid_time_ms(),
        provenance=valid_provenance(),
    )
    def test_no_regression_when_optimized_faster(
        self, baseline_time: float, optimized_time: float, provenance: RunProvenance
    ):
        """When optimized is faster than baseline, is_regression must be False."""
        # Only test improvement cases: optimized time < baseline time
        assume(optimized_time < baseline_time)

        entry = ExpectationEntry(
            example="test_example",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=baseline_time,
            best_optimized_time_ms=optimized_time,
            provenance=provenance,
        )

        # Verify: speedup > 1.0 for improvement
        assert entry.best_speedup > 1.0, (
            f"Speedup {entry.best_speedup} should be > 1.0 when optimized "
            f"({optimized_time}ms) is faster than baseline ({baseline_time}ms)"
        )

        # Verify: is_regression flag is False
        assert entry.is_regression is False, (
            f"is_regression should be False when speedup is {entry.best_speedup}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        baseline_time=valid_time_ms(),
        provenance=valid_provenance(),
    )
    def test_speedup_not_clamped_to_one(
        self, baseline_time: float, provenance: RunProvenance
    ):
        """Speedup must not be artificially clamped to 1.0 for regressions."""
        # Create a clear regression case: optimized is 2x slower
        optimized_time = baseline_time * 2.0

        entry = ExpectationEntry(
            example="test_example",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=baseline_time,
            best_optimized_time_ms=optimized_time,
            provenance=provenance,
        )

        # Expected speedup is 0.5x
        expected_speedup = baseline_time / optimized_time

        # Verify: speedup is NOT clamped to 1.0
        assert entry.best_speedup != 1.0, (
            "Speedup should not be clamped to 1.0 for regressions"
        )

        # Verify: speedup matches actual ratio
        assert abs(entry.best_speedup - expected_speedup) < SPEEDUP_TOLERANCE, (
            f"Speedup {entry.best_speedup} should be {expected_speedup}, not clamped"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        baseline_time=valid_time_ms(),
        optimized_time=valid_time_ms(),
        provenance=valid_provenance(),
    )
    def test_serialized_regression_flag_matches_computed(
        self, baseline_time: float, optimized_time: float, provenance: RunProvenance
    ):
        """Serialized is_regression flag must match computed property."""
        entry = ExpectationEntry(
            example="test_example",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=baseline_time,
            best_optimized_time_ms=optimized_time,
            provenance=provenance,
        )

        serialized = entry.to_dict()
        stored_is_regression = serialized["metrics"]["is_regression"]

        # Verify: serialized flag matches computed property
        assert stored_is_regression == entry.is_regression, (
            f"Serialized is_regression {stored_is_regression} differs from "
            f"computed {entry.is_regression}"
        )


# =============================================================================
# Unit Tests for Data Classes
# =============================================================================


class TestRunProvenance:
    """Unit tests for RunProvenance dataclass."""

    def test_matches_same_config(self):
        """Two provenances with same config should match."""
        prov1 = RunProvenance(
            git_commit="abc123",
            hardware_key="b200",
            profile_name="minimal",
            timestamp="2025-01-01T00:00:00",
            iterations=100,
            warmup_iterations=10,
        )
        prov2 = RunProvenance(
            git_commit="abc123",
            hardware_key="b200",
            profile_name="minimal",
            timestamp="2025-01-02T00:00:00",  # Different timestamp
            iterations=50,  # Different iterations
            warmup_iterations=5,  # Different warmup
        )

        assert prov1.matches(prov2) is True

    def test_matches_different_commit(self):
        """Different git commit should not match."""
        prov1 = RunProvenance(
            git_commit="abc123",
            hardware_key="b200",
            profile_name="minimal",
            timestamp="2025-01-01T00:00:00",
            iterations=100,
            warmup_iterations=10,
        )
        prov2 = RunProvenance(
            git_commit="def456",  # Different commit
            hardware_key="b200",
            profile_name="minimal",
            timestamp="2025-01-01T00:00:00",
            iterations=100,
            warmup_iterations=10,
        )

        assert prov1.matches(prov2) is False

    def test_to_dict_roundtrip(self):
        """to_dict and from_dict should roundtrip correctly."""
        original = RunProvenance(
            git_commit="abc123def456",
            hardware_key="b200",
            profile_name="deep_dive",
            timestamp="2025-12-07T10:00:00",
            iterations=100,
            warmup_iterations=10,
        )

        serialized = original.to_dict()
        restored = RunProvenance.from_dict(serialized)

        assert restored.git_commit == original.git_commit
        assert restored.hardware_key == original.hardware_key
        assert restored.profile_name == original.profile_name
        assert restored.timestamp == original.timestamp
        assert restored.iterations == original.iterations
        assert restored.warmup_iterations == original.warmup_iterations


class TestExpectationEntry:
    """Unit tests for ExpectationEntry dataclass."""

    def test_speedup_computed_correctly(self):
        """Speedup should be baseline / optimized."""
        entry = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=50.0,
            provenance=RunProvenance(
                git_commit="abc123",
                hardware_key="b200",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )

        assert abs(entry.best_speedup - 2.0) < SPEEDUP_TOLERANCE

    def test_regression_detected(self):
        """is_regression should be True when optimized is slower."""
        entry = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=50.0,
            best_optimized_time_ms=100.0,  # Slower than baseline
            provenance=RunProvenance(
                git_commit="abc123",
                hardware_key="b200",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )

        assert entry.best_speedup < 1.0
        assert entry.is_regression is True

    def test_zero_optimized_time_returns_one(self):
        """Zero or negative optimized time should return speedup of 1.0."""
        entry = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=0.0,
            provenance=RunProvenance(
                git_commit="abc123",
                hardware_key="b200",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )

        assert entry.best_speedup == 1.0

    def test_to_dict_includes_computed_speedup(self):
        """Serialized dict should include computed speedup."""
        entry = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=25.0,
            provenance=RunProvenance(
                git_commit="abc123",
                hardware_key="b200",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )

        serialized = entry.to_dict()

        assert "best_speedup" in serialized["metrics"]
        assert abs(serialized["metrics"]["best_speedup"] - 4.0) < SPEEDUP_TOLERANCE
        assert "is_regression" in serialized["metrics"]
        assert serialized["metrics"]["is_regression"] is False

    def test_to_dict_from_dict_roundtrip(self):
        """Full roundtrip should preserve all data."""
        original = ExpectationEntry(
            example="test_example",
            type="cuda",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=50.0,
            provenance=RunProvenance(
                git_commit="abc123",
                hardware_key="b200",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
            baseline_p75_ms=95.0,
            baseline_p90_ms=98.0,
            best_optimized_p75_ms=48.0,
            best_optimized_p90_ms=49.0,
            baseline_throughput={"requests_per_s": 1000.0, "latency_ms": 1.0},
            best_optimized_throughput={"requests_per_s": 2000.0, "latency_ms": 0.5},
            custom_metrics={"custom_speedup": 2.5},
            best_optimization_name="optimized_test",
            best_optimization_file="optimized_test.py",
        )

        serialized = original.to_dict()
        restored = ExpectationEntry.from_dict(serialized)

        assert restored.example == original.example
        assert restored.type == original.type
        assert abs(restored.baseline_time_ms - original.baseline_time_ms) < SPEEDUP_TOLERANCE
        assert abs(restored.best_optimized_time_ms - original.best_optimized_time_ms) < SPEEDUP_TOLERANCE
        assert restored.baseline_p75_ms == original.baseline_p75_ms
        assert restored.baseline_p90_ms == original.baseline_p90_ms
        assert restored.best_optimized_p75_ms == original.best_optimized_p75_ms
        assert restored.best_optimized_p90_ms == original.best_optimized_p90_ms
        assert restored.best_optimization_name == original.best_optimization_name
        assert restored.best_optimization_file == original.best_optimization_file


class TestValidationIssue:
    """Unit tests for ValidationIssue dataclass."""

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        issue = ValidationIssue(
            example_key="test_cuda",
            issue_type="speedup_mismatch",
            message="Speedup does not match timing ratio",
            stored_value=1.5,
            expected_value=1.6,
            delta_pct=6.25,
        )

        serialized = issue.to_dict()

        assert serialized["example_key"] == "test_cuda"
        assert serialized["issue_type"] == "speedup_mismatch"
        assert serialized["stored_value"] == 1.5
        assert serialized["expected_value"] == 1.6
        assert serialized["delta_pct"] == 6.25


class TestValidationReport:
    """Unit tests for ValidationReport dataclass."""

    def test_has_issues_empty(self):
        """Empty report should have no issues."""
        report = ValidationReport(issues=[], total_entries=10, valid_entries=10)

        assert report.has_issues is False
        assert report.invalid_entries == 0

    def test_has_issues_with_issues(self):
        """Report with issues should flag has_issues."""
        issue = ValidationIssue(
            example_key="test",
            issue_type="speedup_mismatch",
            message="Test issue",
            stored_value=1.0,
            expected_value=0.9,
        )
        report = ValidationReport(issues=[issue], total_entries=10, valid_entries=9)

        assert report.has_issues is True
        assert report.invalid_entries == 1


# =============================================================================
# Property 3: Atomic Provenance Consistency
# Validates: Requirements 3.1, 3.2, 4.1
# =============================================================================


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestAtomicProvenanceConsistency:
    """Property 3: For any expectation entry, all metrics SHALL have the
    same provenance (git_commit, hardware_key, profile_name).
    """

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        baseline_time=valid_time_ms(),
        optimized_time=valid_time_ms(),
        provenance=valid_provenance(),
    )
    def test_entry_has_single_provenance(
        self, baseline_time: float, optimized_time: float, provenance: RunProvenance
    ):
        """All metrics in an entry must share the same provenance."""
        entry = ExpectationEntry(
            example="test_example",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=baseline_time,
            best_optimized_time_ms=optimized_time,
            provenance=provenance,
        )

        # Serialize and check provenance is present
        serialized = entry.to_dict()

        assert "provenance" in serialized
        stored_prov = serialized["provenance"]

        # Verify all provenance fields are from the same source
        assert stored_prov["git_commit"] == provenance.git_commit
        assert stored_prov["hardware_key"] == provenance.hardware_key
        assert stored_prov["profile_name"] == provenance.profile_name
        assert stored_prov["iterations"] == provenance.iterations
        assert stored_prov["warmup_iterations"] == provenance.warmup_iterations

    def test_update_entry_preserves_provenance(self, tmp_path):
        """update_entry should preserve the entry's provenance."""
        store = ExpectationsStore(tmp_path, "test_hw", accept_regressions=True)

        provenance = RunProvenance(
            git_commit="abc123def456",
            hardware_key="test_hw",
            profile_name="minimal",
            timestamp="2025-01-01T00:00:00",
            iterations=100,
            warmup_iterations=10,
        )

        entry = ExpectationEntry(
            example="test_example",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=50.0,
            provenance=provenance,
        )

        result = store.update_entry("test_python", entry)

        # Should succeed
        assert result.status in ("updated", "improved", "regressed", "unchanged")

        # Retrieved entry should have same provenance
        retrieved = store.get_entry("test_python")
        assert retrieved is not None
        assert retrieved.provenance.git_commit == provenance.git_commit
        assert retrieved.provenance.hardware_key == provenance.hardware_key
        assert retrieved.provenance.profile_name == provenance.profile_name


# =============================================================================
# Property 4: Mixed Provenance Rejection
# Validates: Requirements 3.3, 4.2
# =============================================================================


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestMixedProvenanceRejection:
    """Property 4: For any update attempt where the new metrics have different
    provenance than existing metrics, the update SHALL be rejected unless
    force_mixed_provenance=True.
    """

    def test_rejects_different_git_commit(self, tmp_path):
        """Updates with different git commit should be rejected."""
        store = ExpectationsStore(tmp_path, "test_hw", accept_regressions=True)

        # First entry
        entry1 = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=50.0,
            provenance=RunProvenance(
                git_commit="commit_aaa",
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )
        result1 = store.update_entry("test_python", entry1)
        assert result1.status in ("updated", "improved")

        # Second entry with different commit
        entry2 = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=40.0,  # Better
            provenance=RunProvenance(
                git_commit="commit_bbb",  # Different!
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-02T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )
        result2 = store.update_entry("test_python", entry2)

        # Should be rejected
        assert result2.status == "rejected"
        assert "provenance" in result2.message.lower() or "mismatch" in result2.message.lower()
        assert len(result2.validation_issues) > 0
        assert result2.validation_issues[0].issue_type == "provenance_mismatch"

    def test_allows_same_provenance(self, tmp_path):
        """Updates with same provenance should be allowed."""
        store = ExpectationsStore(tmp_path, "test_hw", accept_regressions=True)

        provenance = RunProvenance(
            git_commit="commit_same",
            hardware_key="test_hw",
            profile_name="minimal",
            timestamp="2025-01-01T00:00:00",
            iterations=100,
            warmup_iterations=10,
        )

        # First entry
        entry1 = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=50.0,
            provenance=provenance,
        )
        result1 = store.update_entry("test_python", entry1)
        assert result1.status in ("updated", "improved")

        # Second entry with same commit (different timestamp is OK)
        entry2 = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=40.0,
            provenance=RunProvenance(
                git_commit="commit_same",  # Same commit
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-02T00:00:00",  # Different timestamp OK
                iterations=100,
                warmup_iterations=10,
            ),
        )
        result2 = store.update_entry("test_python", entry2)

        # Should be allowed (improved or updated)
        assert result2.status in ("improved", "updated")

    def test_force_mixed_provenance_allows_update(self, tmp_path):
        """force_mixed_provenance=True should allow mismatched commits."""
        store = ExpectationsStore(
            tmp_path, "test_hw", accept_regressions=True, force_mixed_provenance=True
        )

        # First entry
        entry1 = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=50.0,
            provenance=RunProvenance(
                git_commit="commit_aaa",
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )
        result1 = store.update_entry("test_python", entry1)
        assert result1.status in ("updated", "improved")

        # Second entry with different commit
        entry2 = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=40.0,
            provenance=RunProvenance(
                git_commit="commit_bbb",  # Different commit
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-02T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )
        result2 = store.update_entry("test_python", entry2)

        # Should be allowed due to force flag (improved or updated)
        assert result2.status in ("improved", "updated")


# =============================================================================
# Tests for ExpectationsStore
# =============================================================================


class TestExpectationsStore:
    """Unit tests for ExpectationsStore class."""

    def test_creates_file_with_schema_version(self, tmp_path):
        """Saved file should include schema_version."""
        store = ExpectationsStore(tmp_path, "test_hw")

        entry = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=50.0,
            provenance=RunProvenance(
                git_commit="abc123",
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )
        store.update_entry("test_python", entry)
        store.save()

        # Read back and check schema version
        import json
        saved_data = json.loads((tmp_path / "expectations_test_hw.json").read_text())
        assert saved_data.get("schema_version") == 2

    def test_update_entry_computes_speedup(self, tmp_path):
        """update_entry should compute speedup from timing values."""
        store = ExpectationsStore(tmp_path, "test_hw")

        entry = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=25.0,  # 4x speedup
            provenance=RunProvenance(
                git_commit="abc123",
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )
        result = store.update_entry("test_python", entry)

        assert result.status in ("updated", "improved")
        assert result.entry is not None
        assert abs(result.entry.best_speedup - 4.0) < SPEEDUP_TOLERANCE

    def test_rejects_regression_by_default(self, tmp_path):
        """Should reject regressions by default."""
        store = ExpectationsStore(tmp_path, "test_hw")

        # First entry with good speedup
        entry1 = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=25.0,  # 4x speedup
            provenance=RunProvenance(
                git_commit="abc123",
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )
        result1 = store.update_entry("test_python", entry1)
        assert result1.status in ("updated", "improved")

        # Second entry with worse speedup (regression)
        entry2 = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=100.0,  # 1x speedup - regression!
            provenance=RunProvenance(
                git_commit="abc123",
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-02T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )
        result2 = store.update_entry("test_python", entry2)

        assert result2.status == "rejected"
        assert "regression" in result2.message.lower()

    def test_accepts_regression_when_flag_set(self, tmp_path):
        """Should accept regressions when accept_regressions=True."""
        store = ExpectationsStore(tmp_path, "test_hw", accept_regressions=True)

        # First entry
        entry1 = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=25.0,
            provenance=RunProvenance(
                git_commit="abc123",
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )
        store.update_entry("test_python", entry1)

        # Regression
        entry2 = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=100.0,
            provenance=RunProvenance(
                git_commit="abc123",
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-02T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )
        result2 = store.update_entry("test_python", entry2)

        assert result2.status == "regressed"

    def test_get_entry_returns_none_for_missing(self, tmp_path):
        """get_entry should return None for non-existent entries."""
        store = ExpectationsStore(tmp_path, "test_hw")

        result = store.get_entry("nonexistent")
        assert result is None

    def test_get_entry_returns_entry(self, tmp_path):
        """get_entry should return the stored entry."""
        store = ExpectationsStore(tmp_path, "test_hw")

        entry = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=50.0,
            provenance=RunProvenance(
                git_commit="abc123",
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )
        store.update_entry("test_python", entry)

        retrieved = store.get_entry("test_python")
        assert retrieved is not None
        assert retrieved.example == "test"
        assert abs(retrieved.baseline_time_ms - 100.0) < SPEEDUP_TOLERANCE


# =============================================================================
# Property 5: Validation Detects Inconsistencies
# Validates: Requirements 1.3, 5.1, 5.2
# =============================================================================


class TestValidationDetectsInconsistencies:
    """Property 5: For any expectation file containing entries where stored
    best_speedup differs from baseline_time_ms / best_optimized_time_ms by
    more than tolerance, the validation tool SHALL report each as a
    speedup_mismatch issue.
    """

    def test_detects_speedup_mismatch(self, tmp_path):
        """Validation should detect when stored speedup differs from computed."""
        store = ExpectationsStore(tmp_path, "test_hw")

        # Manually inject an inconsistent entry (simulating legacy data)
        store._data["examples"]["test_python"] = {
            "example": "test",
            "type": "python",
            "metrics": {
                "baseline_time_ms": 100.0,
                "best_optimized_time_ms": 50.0,
                "best_speedup": 3.0,  # Wrong! Should be 2.0
                "best_optimized_speedup": 3.0,
            },
            "provenance": {
                "git_commit": "abc123",
                "hardware_key": "test_hw",
                "profile_name": "minimal",
                "timestamp": "2025-01-01T00:00:00",
                "iterations": 100,
                "warmup_iterations": 10,
            },
            "metadata": {},
        }

        issues = store.validate_entry("test_python", store._data["examples"]["test_python"])

        assert len(issues) >= 1
        speedup_issues = [i for i in issues if i.issue_type == "speedup_mismatch"]
        assert len(speedup_issues) == 1
        assert speedup_issues[0].stored_value == 3.0
        assert abs(speedup_issues[0].expected_value - 2.0) < SPEEDUP_TOLERANCE

    def test_no_issues_for_consistent_entry(self, tmp_path):
        """Validation should report no issues for consistent entries."""
        store = ExpectationsStore(tmp_path, "test_hw")

        # Create proper entry using update_entry (which computes speedup correctly)
        entry = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=50.0,
            provenance=RunProvenance(
                git_commit="abc123",
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )
        store.update_entry("test_python", entry)

        issues = store.validate_entry("test_python", store._data["examples"]["test_python"])

        # Should have no issues (metadata speedup mismatch is acceptable since it's not set)
        speedup_issues = [i for i in issues if i.issue_type == "speedup_mismatch"]
        assert len(speedup_issues) == 0

    def test_validate_all_aggregates_issues(self, tmp_path):
        """validate_all should collect issues from all entries."""
        store = ExpectationsStore(tmp_path, "test_hw")

        # Add two inconsistent entries
        store._data["examples"]["entry1"] = {
            "example": "entry1",
            "type": "python",
            "metrics": {
                "baseline_time_ms": 100.0,
                "best_optimized_time_ms": 50.0,
                "best_speedup": 5.0,  # Wrong, should be 2.0
            },
            "provenance": {
                "git_commit": "abc",
                "hardware_key": "test_hw",
                "profile_name": "minimal",
                "timestamp": "2025-01-01T00:00:00",
                "iterations": 100,
                "warmup_iterations": 10,
            },
            "metadata": {},
        }
        store._data["examples"]["entry2"] = {
            "example": "entry2",
            "type": "python",
            "metrics": {
                "baseline_time_ms": 200.0,
                "best_optimized_time_ms": 100.0,
                "best_speedup": 10.0,  # Wrong, should be 2.0
            },
            "provenance": {
                "git_commit": "abc",
                "hardware_key": "test_hw",
                "profile_name": "minimal",
                "timestamp": "2025-01-01T00:00:00",
                "iterations": 100,
                "warmup_iterations": 10,
            },
            "metadata": {},
        }

        report = store.validate_all()

        assert report.has_issues
        assert report.total_entries == 2
        assert report.valid_entries == 0
        assert len(report.issues) >= 2


# =============================================================================
# Property 6: Masked Regression Detection
# Validates: Requirements 5.4
# =============================================================================


class TestMaskedRegressionDetection:
    """Property 6: For any expectation entry where best_speedup == 1.0 but
    baseline_time_ms / best_optimized_time_ms < 1.0, the validation tool
    SHALL report it as a masked_regression issue.
    """

    def test_detects_masked_regression(self, tmp_path):
        """Should detect when speedup=1.0 hides an actual regression."""
        store = ExpectationsStore(tmp_path, "test_hw")

        # Inject entry with masked regression
        store._data["examples"]["test_python"] = {
            "example": "test",
            "type": "python",
            "metrics": {
                "baseline_time_ms": 50.0,
                "best_optimized_time_ms": 100.0,  # Slower! ratio = 0.5
                "best_speedup": 1.0,  # Masked to 1.0!
                "best_optimized_speedup": 1.0,
            },
            "provenance": {
                "git_commit": "abc123",
                "hardware_key": "test_hw",
                "profile_name": "minimal",
                "timestamp": "2025-01-01T00:00:00",
                "iterations": 100,
                "warmup_iterations": 10,
            },
            "metadata": {},
        }

        issues = store.validate_entry("test_python", store._data["examples"]["test_python"])

        masked_issues = [i for i in issues if i.issue_type == "masked_regression"]
        assert len(masked_issues) == 1
        assert masked_issues[0].stored_value == 1.0
        assert masked_issues[0].expected_value < 1.0

    def test_no_masked_regression_for_actual_speedup_one(self, tmp_path):
        """Should not report masked regression when ratio actually is ~1.0."""
        store = ExpectationsStore(tmp_path, "test_hw")

        # Entry where speedup is legitimately ~1.0
        store._data["examples"]["test_python"] = {
            "example": "test",
            "type": "python",
            "metrics": {
                "baseline_time_ms": 100.0,
                "best_optimized_time_ms": 100.0,  # Equal times, speedup = 1.0
                "best_speedup": 1.0,
                "best_optimized_speedup": 1.0,
            },
            "provenance": {
                "git_commit": "abc123",
                "hardware_key": "test_hw",
                "profile_name": "minimal",
                "timestamp": "2025-01-01T00:00:00",
                "iterations": 100,
                "warmup_iterations": 10,
            },
            "metadata": {},
        }

        issues = store.validate_entry("test_python", store._data["examples"]["test_python"])

        masked_issues = [i for i in issues if i.issue_type == "masked_regression"]
        assert len(masked_issues) == 0


# =============================================================================
# Property 8: Single Best Selection Consistency
# Validates: Requirements 7.1, 7.2, 7.4
# =============================================================================


class TestSingleBestSelectionConsistency:
    """Property 8: For any expectation entry, the best_speedup in metrics and
    best_optimization_speedup in metadata SHALL refer to the same optimization
    (same speedup value within tolerance).
    """

    def test_detects_metadata_drift(self, tmp_path):
        """Should detect when metadata speedup differs from metrics speedup."""
        store = ExpectationsStore(tmp_path, "test_hw")

        # Inject entry with drifted metadata
        store._data["examples"]["test_python"] = {
            "example": "test",
            "type": "python",
            "metrics": {
                "baseline_time_ms": 100.0,
                "best_optimized_time_ms": 50.0,
                "best_speedup": 2.0,
                "best_optimized_speedup": 2.0,
            },
            "provenance": {
                "git_commit": "abc123",
                "hardware_key": "test_hw",
                "profile_name": "minimal",
                "timestamp": "2025-01-01T00:00:00",
                "iterations": 100,
                "warmup_iterations": 10,
            },
            "metadata": {
                "best_optimization_speedup": 3.5,  # Drifted from metrics!
            },
        }

        issues = store.validate_entry("test_python", store._data["examples"]["test_python"])

        drift_issues = [i for i in issues if i.issue_type == "metadata_drift"]
        assert len(drift_issues) == 1

    def test_select_best_optimization_returns_highest_speedup(self):
        """select_best_optimization should return the optimization with highest speedup."""
        from core.benchmark.expectations import select_best_optimization

        optimizations = [
            {"status": "succeeded", "speedup": 1.5, "file": "opt1.py"},
            {"status": "succeeded", "speedup": 2.5, "file": "opt2.py"},
            {"status": "failed", "speedup": 3.5, "file": "opt3.py"},  # Failed, should ignore
            {"status": "succeeded", "speedup": 2.0, "file": "opt4.py"},
        ]

        best = select_best_optimization(optimizations)

        assert best is not None
        assert best["file"] == "opt2.py"
        assert best["speedup"] == 2.5

    def test_select_best_optimization_ignores_failed(self):
        """select_best_optimization should ignore failed optimizations."""
        from core.benchmark.expectations import select_best_optimization

        optimizations = [
            {"status": "failed", "speedup": 10.0, "file": "opt1.py"},
            {"status": "succeeded", "speedup": 1.5, "file": "opt2.py"},
        ]

        best = select_best_optimization(optimizations)

        assert best is not None
        assert best["file"] == "opt2.py"

    def test_select_best_optimization_returns_none_for_empty(self):
        """select_best_optimization should return None for empty list."""
        from core.benchmark.expectations import select_best_optimization

        assert select_best_optimization([]) is None
        assert select_best_optimization(None) is None

    def test_select_best_optimization_handles_all_failed(self):
        """select_best_optimization should return None if all failed."""
        from core.benchmark.expectations import select_best_optimization

        optimizations = [
            {"status": "failed", "speedup": 2.0, "file": "opt1.py"},
            {"status": "error", "speedup": 3.0, "file": "opt2.py"},
        ]

        assert select_best_optimization(optimizations) is None


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestCustomMetricsSeparation:
    """Property 7: Custom Metrics Separation.

    Validates: Requirements 6.1, 6.2, 6.4
    For any benchmark run with custom metrics (e.g., scenario_total_phase_ms),
    the best_speedup SHALL equal the timing ratio, and any custom speedup
    SHALL be stored separately in custom_metrics.custom_speedup.
    """

    def test_custom_speedup_does_not_replace_timing_speedup(self, tmp_path):
        """Custom speedup should be stored separately, not replace timing speedup."""
        store = ExpectationsStore(tmp_path, "test_hw")

        # Simulate a benchmark with custom scenario metrics
        # Timing says 2x speedup, but scenario says 3x
        metrics = {
            "baseline_time_ms": 100.0,
            "best_optimized_time_ms": 50.0,  # 2x timing speedup
            "best_speedup": 2.0,
            "best_optimized_speedup": 2.0,
        }
        metadata = {
            "example": "test",
            "type": "python",
            "best_optimization": "opt1",
        }

        store.evaluate("test_entry", metrics, metadata)
        store.save(force=True)

        # The timing-based speedup should be preserved
        stored_metrics = store._data["examples"]["test_entry"]["metrics"]
        assert abs(stored_metrics["best_speedup"] - 2.0) < SPEEDUP_TOLERANCE

    def test_entry_with_custom_metrics_preserves_timing_speedup(self, tmp_path):
        """ExpectationEntry should compute speedup from timing, ignoring custom_metrics."""
        # Even if custom_metrics has a different speedup value,
        # best_speedup property should use timing
        provenance = RunProvenance(
            git_commit="abc123",
            hardware_key="test_hw",
            profile_name="test_profile",
            timestamp="2025-01-01T00:00:00",
            iterations=100,
            warmup_iterations=10,
        )

        entry = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=50.0,  # 2x speedup
            provenance=provenance,
            custom_metrics={"custom_speedup": 5.0},  # Different value
        )

        # best_speedup property should return timing-based value, not custom
        assert abs(entry.best_speedup - 2.0) < SPEEDUP_TOLERANCE

    def test_custom_metrics_roundtrip(self, tmp_path):
        """Custom metrics should survive serialization without affecting timing speedup."""
        store = ExpectationsStore(tmp_path, "test_hw")

        provenance = RunProvenance(
            git_commit="abc123",
            hardware_key="test_hw",
            profile_name="test_profile",
            timestamp="2025-01-01T00:00:00",
            iterations=100,
            warmup_iterations=10,
        )

        entry = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=25.0,  # 4x timing speedup
            provenance=provenance,
            custom_metrics={
                "scenario_total_phase_ms": 20.0,
                "custom_speedup": 5.0,  # Scenario says 5x
            },
        )

        result = store.update_entry("test::python", entry)
        store.save(force=True)

        # Reload and verify
        loaded_entry = store.get_entry("test::python")

        # Timing speedup preserved
        assert abs(loaded_entry.best_speedup - 4.0) < SPEEDUP_TOLERANCE

        # Custom metrics preserved
        assert loaded_entry.custom_metrics is not None
        assert abs(loaded_entry.custom_metrics.get("custom_speedup", 0) - 5.0) < SPEEDUP_TOLERANCE


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_validate_on_load_warns_on_issues(self, tmp_path):
        """validate_on_load should warn when file has issues."""
        import warnings

        # First create a file with inconsistent data
        store = ExpectationsStore(tmp_path, "test_hw")
        store._data["examples"]["bad_entry"] = {
            "example": "bad",
            "type": "python",
            "metrics": {
                "baseline_time_ms": 100.0,
                "best_optimized_time_ms": 50.0,
                "best_speedup": 5.0,  # Wrong! Should be 2.0
            },
            "provenance": {
                "git_commit": "abc",
                "hardware_key": "test_hw",
                "profile_name": "minimal",
                "timestamp": "2025-01-01T00:00:00",
                "iterations": 100,
                "warmup_iterations": 10,
            },
            "metadata": {},
        }
        store.save(force=True)

        # Now load with validation - should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            store2 = ExpectationsStore(tmp_path, "test_hw", validate_on_load=True)
            # Check that a warning was issued
            assert len(w) >= 1
            assert "validation issues" in str(w[0].message).lower()

    def test_validate_on_load_no_warn_when_valid(self, tmp_path):
        """validate_on_load should not warn when file is valid."""
        import warnings

        # Create a valid file
        store = ExpectationsStore(tmp_path, "test_hw")
        entry = ExpectationEntry(
            example="good",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=100.0,
            best_optimized_time_ms=50.0,
            provenance=RunProvenance(
                git_commit="abc",
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )
        store.update_entry("good_entry", entry)
        store.save()

        # Load with validation - should not warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            store2 = ExpectationsStore(tmp_path, "test_hw", validate_on_load=True)
            validation_warnings = [x for x in w if "validation issues" in str(x.message).lower()]
            assert len(validation_warnings) == 0

    def test_compute_speedup_handles_zero_optimized_time(self):
        """compute_speedup should return 1.0 for zero optimized time."""
        from core.benchmark.expectations import compute_speedup

        assert compute_speedup(100.0, 0.0) == 1.0
        assert compute_speedup(100.0, -1.0) == 1.0

    def test_compute_speedup_handles_normal_values(self):
        """compute_speedup should calculate correctly for normal values."""
        from core.benchmark.expectations import compute_speedup

        assert abs(compute_speedup(100.0, 50.0) - 2.0) < SPEEDUP_TOLERANCE
        assert abs(compute_speedup(100.0, 100.0) - 1.0) < SPEEDUP_TOLERANCE
        assert abs(compute_speedup(50.0, 100.0) - 0.5) < SPEEDUP_TOLERANCE

    def test_select_best_optimization_with_none_speedup(self):
        """select_best_optimization should handle None speedup values."""
        from core.benchmark.expectations import select_best_optimization

        optimizations = [
            {"status": "succeeded", "speedup": None, "file": "opt1.py"},
            {"status": "succeeded", "speedup": 2.0, "file": "opt2.py"},
        ]

        best = select_best_optimization(optimizations)
        assert best is not None
        assert best["file"] == "opt2.py"

    def test_select_best_optimization_all_none_speedup(self):
        """select_best_optimization with all None speedups should return first succeeded."""
        from core.benchmark.expectations import select_best_optimization

        optimizations = [
            {"status": "succeeded", "speedup": None, "file": "opt1.py"},
            {"status": "succeeded", "speedup": None, "file": "opt2.py"},
        ]

        best = select_best_optimization(optimizations)
        # Should return opt2 since 0.0 > -inf (initial best_speedup)
        # Actually both have speedup 0.0 (None -> 0.0), so first one wins
        assert best is not None

    def test_entry_with_very_large_speedup(self):
        """Entry should handle very large speedup values correctly."""
        entry = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=1000000.0,  # 1 second
            best_optimized_time_ms=0.001,  # 1 microsecond
            provenance=RunProvenance(
                git_commit="abc",
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )

        # Speedup is 1,000,000,000x
        assert entry.best_speedup == 1000000000.0
        assert entry.is_regression is False

    def test_entry_with_very_small_speedup(self):
        """Entry should handle very small (regression) speedup values correctly."""
        entry = ExpectationEntry(
            example="test",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=0.001,  # 1 microsecond
            best_optimized_time_ms=1000000.0,  # 1 second
            provenance=RunProvenance(
                git_commit="abc",
                hardware_key="test_hw",
                profile_name="minimal",
                timestamp="2025-01-01T00:00:00",
                iterations=100,
                warmup_iterations=10,
            ),
        )

        # Speedup is 0.000000001x (severe regression)
        assert entry.best_speedup < 1.0
        assert entry.is_regression is True

    def test_validation_with_missing_metrics(self, tmp_path):
        """Validation should handle entries with missing metrics gracefully."""
        store = ExpectationsStore(tmp_path, "test_hw")

        # Entry with no timing metrics
        store._data["examples"]["incomplete"] = {
            "example": "incomplete",
            "type": "python",
            "metrics": {},
            "provenance": {
                "git_commit": "abc",
                "hardware_key": "test_hw",
                "profile_name": "minimal",
                "timestamp": "2025-01-01T00:00:00",
                "iterations": 100,
                "warmup_iterations": 10,
            },
            "metadata": {},
        }

        # Should not crash
        issues = store.validate_entry("incomplete", store._data["examples"]["incomplete"])
        # May or may not have issues, but should not crash
        assert isinstance(issues, list)

    def test_evaluate_safeguard_corrects_wrong_speedup(self, tmp_path):
        """evaluate() should re-derive speedup even if caller passes wrong value.

        This is the critical safeguard - even if something upstream computes
        speedup incorrectly, the store will fix it before writing.
        """
        store = ExpectationsStore(tmp_path, "test_hw")

        # Pass in WRONG speedup (5.0 instead of correct 2.0)
        metrics = {
            "baseline_time_ms": 100.0,
            "best_optimized_time_ms": 50.0,
            "best_speedup": 5.0,  # WRONG! Should be 2.0
            "best_optimized_speedup": 5.0,  # WRONG!
        }

        store.evaluate("test_entry", metrics)
        store.save(force=True)

        # Read back and verify speedup was corrected
        stored = store._data["examples"]["test_entry"]["metrics"]
        assert abs(stored["best_speedup"] - 2.0) < SPEEDUP_TOLERANCE
        assert abs(stored["best_optimized_speedup"] - 2.0) < SPEEDUP_TOLERANCE

    def test_evaluate_safeguard_does_not_mutate_input(self, tmp_path):
        """evaluate() should not mutate the caller's metrics dict."""
        store = ExpectationsStore(tmp_path, "test_hw")

        metrics = {
            "baseline_time_ms": 100.0,
            "best_optimized_time_ms": 50.0,
            "best_speedup": 5.0,  # Wrong value
        }

        store.evaluate("test_entry", metrics)

        # Original dict should be unchanged
        assert metrics["best_speedup"] == 5.0

    def test_metadata_speedup_derived_from_metrics(self, tmp_path):
        """Metadata speedup should be derived from metrics timing, not passed value."""
        store = ExpectationsStore(tmp_path, "test_hw")

        metrics = {
            "baseline_time_ms": 100.0,
            "best_optimized_time_ms": 25.0,  # 4x speedup
            "best_speedup": 4.0,
            "best_optimized_speedup": 4.0,
        }
        metadata = {
            "example": "test",
            "type": "python",
            "best_optimization": "opt1",
            "best_optimization_speedup": 999.0,  # WRONG! Should be derived
        }

        store.evaluate("test_entry", metrics, metadata)
        store.save(force=True)

        # Metadata speedup should be derived from timing, not the passed value
        stored_meta = store._data["examples"]["test_entry"]["metadata"]
        assert abs(stored_meta["best_optimization_speedup"] - 4.0) < SPEEDUP_TOLERANCE
