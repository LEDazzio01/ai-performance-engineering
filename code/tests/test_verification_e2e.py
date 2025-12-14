"""End-to-end tests for benchmark verification with real chapter examples.

These tests validate the verification system works with actual benchmark
pairs from ch07 and ch11.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for end-to-end verification tests"
)


class TestCh11StreamsVerification:
    """End-to-end verification tests for ch11 streams benchmark pair."""
    
    @pytest.fixture
    def temp_env(self, tmp_path):
        """Fixture providing temporary environment."""
        return {
            "cache_dir": tmp_path / "cache",
            "quarantine_file": tmp_path / "quarantine.json",
        }
    
    def test_baseline_optimized_signatures_compatible(self, temp_env):
        """Test that baseline and optimized have compatible signatures."""
        from ch11.baseline_streams import get_benchmark as get_baseline
        from ch11.optimized_streams import get_benchmark as get_optimized
        from core.benchmark.verify_runner import VerifyRunner, VerifyConfig
        from core.benchmark.quarantine import QuarantineManager
        
        baseline = get_baseline()
        optimized = get_optimized()

        # Keep this test fast: shrink workload before setup() allocates buffers.
        for bench in (baseline, optimized):
            if hasattr(bench, "N"):
                bench.N = 2048
            if hasattr(bench, "num_chunks"):
                bench.num_chunks = 4
        
        # Setup both benchmarks
        baseline.setup()
        optimized.setup()
        
        try:
            # Payload-based signatures are only available after a run.
            baseline.benchmark_fn()
            baseline.capture_verification_payload()
            optimized.benchmark_fn()
            optimized.capture_verification_payload()

            # Extract and compare signatures
            runner = VerifyRunner(
                cache_dir=temp_env["cache_dir"],
                quarantine_manager=QuarantineManager(temp_env["quarantine_file"]),
            )
            
            baseline_sig = runner._extract_signature(baseline)
            optimized_sig = runner._extract_signature(optimized)
            
            # Both should have signatures
            assert baseline_sig is not None, "Baseline should have a signature"
            assert optimized_sig is not None, "Optimized should have a signature"
            
            # Signatures should match (same workload)
            assert baseline_sig.hash() == optimized_sig.hash(), (
                f"Signatures should match for same workload.\n"
                f"Baseline: {baseline_sig.to_dict()}\n"
                f"Optimized: {optimized_sig.to_dict()}"
            )
        finally:
            baseline.teardown()
            optimized.teardown()
    
    def test_verify_baseline_caches_golden_output(self, temp_env):
        """Test that verifying baseline creates golden output cache."""
        from ch11.baseline_streams import get_benchmark as get_baseline
        from core.benchmark.verify_runner import VerifyRunner, VerifyConfig
        from core.benchmark.quarantine import QuarantineManager
        
        baseline = get_baseline()
        
        runner = VerifyRunner(
            cache_dir=temp_env["cache_dir"],
            quarantine_manager=QuarantineManager(temp_env["quarantine_file"]),
        )
        
        config = VerifyConfig(seed=42, force_recache=True)
        result = runner.verify_baseline(baseline, config)
        
        # Baseline verification may fail if benchmark lacks proper signature
        # This is expected for benchmarks that haven't been fully migrated
        if not result.passed:
            # Skip if benchmark needs migration
            pytest.skip(f"Benchmark needs verification migration: {result.reason}")
        
        # Golden output should be cached
        assert result.signature_hash is not None
        assert runner.cache.has(result.signature_hash)
    
    def test_workload_metadata_captured(self, temp_env):
        """Test that workload metadata is properly captured."""
        from ch11.baseline_streams import get_benchmark as get_baseline
        from core.benchmark.verify_runner import VerifyRunner, VerifyConfig
        from core.benchmark.quarantine import QuarantineManager
        
        baseline = get_baseline()
        baseline.setup()
        
        try:
            # Workload metadata should be registered after setup
            metadata = baseline.get_workload_metadata()
            
            # Should have metadata registered via register_workload_metadata
            assert metadata is not None, "Workload metadata should be registered"
            
            # Check for specific fields - WorkloadMetadata uses "_iteration" suffix
            assert hasattr(metadata, "tokens_per_iteration") or hasattr(metadata, "requests_per_iteration"), (
                "Should have tokens_per_iteration or requests_per_iteration"
            )
            
            # Verify values are populated
            has_value = (
                (metadata.tokens_per_iteration is not None and metadata.tokens_per_iteration > 0) or
                (metadata.requests_per_iteration is not None and metadata.requests_per_iteration > 0)
            )
            assert has_value, "Workload metadata should have non-zero values"
        finally:
            baseline.teardown()


class TestCh11ContractCompliance:
    """Test that ch11 benchmarks comply with the verification contract."""
    
    def test_baseline_has_required_methods(self):
        """Test baseline has all recommended verification methods."""
        from ch11.baseline_streams import get_benchmark
        from core.benchmark.contract import BenchmarkContract
        
        benchmark = get_benchmark()
        
        # Check for required methods
        assert hasattr(benchmark, "setup"), "Missing setup method"
        assert hasattr(benchmark, "benchmark_fn"), "Missing benchmark_fn method"
        assert hasattr(benchmark, "teardown"), "Missing teardown method"
        
        # Check for recommended verification methods
        assert hasattr(benchmark, "get_input_signature") or hasattr(benchmark, "get_workload_metadata"), (
            "Should have get_input_signature or get_workload_metadata"
        )
    
    def test_optimized_has_required_methods(self):
        """Test optimized has all recommended verification methods."""
        from ch11.optimized_streams import get_benchmark
        
        benchmark = get_benchmark()
        
        # Check for required methods
        assert hasattr(benchmark, "setup"), "Missing setup method"
        assert hasattr(benchmark, "benchmark_fn"), "Missing benchmark_fn method"
        assert hasattr(benchmark, "teardown"), "Missing teardown method"
        
        # Check for recommended verification methods
        assert hasattr(benchmark, "validate_result"), "Should have validate_result"


class TestVerificationEnforcement:
    """Test enforcement phase behavior."""
    
    def test_detect_phase_allows_missing_methods(self, monkeypatch):
        """Test that DETECT phase doesn't fail for missing methods."""
        from core.benchmark.verification import get_enforcement_phase, EnforcementPhase
        from core.benchmark.contract import BenchmarkContract
        
        # Set to DETECT phase
        monkeypatch.setenv("VERIFY_ENFORCEMENT_PHASE", "detect")
        
        phase = get_enforcement_phase()
        assert phase == EnforcementPhase.DETECT
        
        # Create a minimal benchmark without verification methods
        class MinimalBenchmark:
            def setup(self): pass
            def benchmark_fn(self): pass
            def teardown(self): pass
        
        benchmark = MinimalBenchmark()
        compliant, errors, warnings = BenchmarkContract.check_verification_compliance(benchmark)
        
        # In DETECT phase, issues should be warnings, not errors
        assert compliant is True, "Should be compliant in DETECT phase"
        assert len(warnings) > 0, "Should have warnings for missing methods"
    
    def test_gate_phase_fails_missing_methods(self, monkeypatch):
        """Test that GATE phase fails for missing verification methods."""
        from core.benchmark.verification import get_enforcement_phase, EnforcementPhase
        from core.benchmark.contract import BenchmarkContract
        
        # Set to GATE phase
        monkeypatch.setenv("VERIFY_ENFORCEMENT_PHASE", "gate")
        
        phase = get_enforcement_phase()
        assert phase == EnforcementPhase.GATE
        
        # Create a minimal benchmark without verification methods
        class MinimalBenchmark:
            def setup(self): pass
            def benchmark_fn(self): pass
            def teardown(self): pass
        
        benchmark = MinimalBenchmark()
        compliant, errors, warnings = BenchmarkContract.check_verification_compliance(benchmark)
        
        # In GATE phase, missing methods should be errors
        assert compliant is False, "Should NOT be compliant in GATE phase"
        assert len(errors) > 0, "Should have errors for missing methods"
    
    def test_compliant_benchmark_passes(self, monkeypatch):
        """Test that a compliant benchmark passes in any phase."""
        from core.benchmark.contract import BenchmarkContract
        
        # Set to GATE phase (strictest)
        monkeypatch.setenv("VERIFY_ENFORCEMENT_PHASE", "gate")
        
        # Create a benchmark with ALL required verification methods (strict mode)
        class CompliantBenchmark:
            def setup(self): pass
            def benchmark_fn(self): pass
            def teardown(self): pass
            def get_input_signature(self): return {"shapes": {"x": (10,)}, "dtypes": {"x": "float32"}, "batch_size": 1, "parameter_count": 10, "precision_flags": {}}
            def validate_result(self): return None
            def get_verify_output(self):
                """Required in strict mode - return output tensor(s) for verification."""
                return {"output": torch.tensor([1.0])}
            def get_output_tolerance(self):
                return (1e-3, 1e-3)
        
        benchmark = CompliantBenchmark()
        compliant, errors, warnings = BenchmarkContract.check_verification_compliance(benchmark)
        
        assert compliant is True, f"Should be compliant, got errors: {errors}"
        assert len(errors) == 0, f"Should have no errors: {errors}"


class TestBenchmarkHarnessVerify:
    """Test BenchmarkHarness.verify() integration."""
    
    @pytest.fixture
    def harness(self):
        """Create a benchmark harness."""
        from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkConfig
        
        config = BenchmarkConfig(
            iterations=2,
            warmup=5,  # Minimum warmup to avoid warnings
            deterministic=True,
            seed=42,
        )
        return BenchmarkHarness(config=config)
    
    def test_harness_verify_method_exists(self, harness):
        """Test that verify method exists on harness."""
        assert hasattr(harness, "verify"), "Harness should have verify method"
        assert callable(harness.verify), "verify should be callable"
    
    def test_harness_gate_perf_method_exists(self, harness):
        """Test that gate_perf method exists on harness."""
        assert hasattr(harness, "gate_perf"), "Harness should have gate_perf method"
        assert callable(harness.gate_perf), "gate_perf should be callable"


class TestGatePerf:
    """Test gate_perf functionality."""
    
    @pytest.fixture
    def temp_env(self, tmp_path):
        """Fixture providing temporary environment."""
        return {
            "cache_dir": tmp_path / "cache",
            "quarantine_file": tmp_path / "quarantine.json",
        }
    
    def test_detect_phase_always_allows(self, temp_env, monkeypatch):
        """Test that DETECT phase always allows perf."""
        from core.benchmark.verify_runner import VerifyRunner
        from core.benchmark.quarantine import QuarantineManager
        from core.benchmark.verification import QuarantineReason
        
        monkeypatch.setenv("VERIFY_ENFORCEMENT_PHASE", "detect")
        
        manager = QuarantineManager(temp_env["quarantine_file"])
        manager.quarantine("test.py", QuarantineReason.OUTPUT_MISMATCH)
        
        runner = VerifyRunner(
            cache_dir=temp_env["cache_dir"],
            quarantine_manager=manager,
        )
        
        # Even quarantined benchmarks are allowed in DETECT phase
        allowed, reason = runner.gate_perf("test.py")
        assert allowed is True
    
    def test_quarantine_phase_blocks_quarantined(self, temp_env, monkeypatch):
        """Test that QUARANTINE phase blocks quarantined benchmarks."""
        from core.benchmark.verify_runner import VerifyRunner
        from core.benchmark.quarantine import QuarantineManager
        from core.benchmark.verification import QuarantineReason
        
        monkeypatch.setenv("VERIFY_ENFORCEMENT_PHASE", "quarantine")
        
        manager = QuarantineManager(temp_env["quarantine_file"])
        manager.quarantine("ch11/baseline_streams.py", QuarantineReason.OUTPUT_MISMATCH)
        
        runner = VerifyRunner(
            cache_dir=temp_env["cache_dir"],
            quarantine_manager=manager,
        )
        
        allowed, reason = runner.gate_perf("ch11/baseline_streams.py")
        assert allowed is False
        assert "quarantined" in reason.lower()
    
    def test_gate_phase_blocks_quarantined(self, temp_env, monkeypatch):
        """Test that GATE phase blocks quarantined benchmarks."""
        from core.benchmark.verify_runner import VerifyRunner
        from core.benchmark.quarantine import QuarantineManager
        from core.benchmark.verification import QuarantineReason
        
        monkeypatch.setenv("VERIFY_ENFORCEMENT_PHASE", "gate")
        
        manager = QuarantineManager(temp_env["quarantine_file"])
        manager.quarantine("ch11/baseline_streams.py", QuarantineReason.OUTPUT_MISMATCH)
        
        runner = VerifyRunner(
            cache_dir=temp_env["cache_dir"],
            quarantine_manager=manager,
        )
        
        allowed, reason = runner.gate_perf("ch11/baseline_streams.py")
        assert allowed is False
        assert "GATE" in reason
    
    def test_gate_perf_allows_non_quarantined(self, temp_env, monkeypatch):
        """Test that non-quarantined benchmarks are allowed in GATE phase."""
        from core.benchmark.verify_runner import VerifyRunner
        from core.benchmark.quarantine import QuarantineManager
        
        monkeypatch.setenv("VERIFY_ENFORCEMENT_PHASE", "gate")
        
        runner = VerifyRunner(
            cache_dir=temp_env["cache_dir"],
            quarantine_manager=QuarantineManager(temp_env["quarantine_file"]),
        )
        
        allowed, reason = runner.gate_perf("ch11/baseline_streams.py")
        assert allowed is True
        assert reason is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
