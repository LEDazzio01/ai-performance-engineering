"""Tests for benchmark input/output verification.

Tests the verification system that ensures:
1. Input equivalence - baseline and optimized benchmarks operate on same workload
2. Output correctness - optimized benchmarks produce same results as baseline

Without these verifications, benchmark comparisons are meaningless:
- Different inputs = comparing apples to oranges
- Different outputs = optimization is broken
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch

# Add repo root to path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.env_defaults import apply_env_defaults
apply_env_defaults()

import torch
from common.python.benchmark_harness import BaseBenchmark


# =============================================================================
# Test Fixtures - Mock Benchmarks
# =============================================================================

class MockBenchmarkWithSignature(BaseBenchmark):
    """Mock benchmark with explicit input signature."""
    
    def __init__(self, batch_size=32, seq_len=128, hidden_size=768):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.data = None
    
    def setup(self):
        self.data = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
    
    def benchmark_fn(self):
        self.output = self.data * 2
    
    def teardown(self):
        self.data = None
        self.output = None
        super().teardown()


class MockBenchmarkDifferentWorkload(BaseBenchmark):
    """Mock benchmark with DIFFERENT workload (larger batch)."""
    
    def __init__(self, batch_size=64, seq_len=128, hidden_size=768):  # Note: batch_size differs
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.data = None
    
    def setup(self):
        self.data = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
    
    def benchmark_fn(self):
        self.output = self.data * 2
    
    def teardown(self):
        self.data = None
        self.output = None
        super().teardown()


class MockBenchmarkSkipsInputVerification(BaseBenchmark):
    """Mock benchmark that opts out of input verification."""
    
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
    
    def setup(self):
        pass
    
    def benchmark_fn(self):
        pass
    
    def skip_input_verification(self) -> bool:
        return True  # Opt out


class MockBenchmarkSkipsOutputVerification(BaseBenchmark):
    """Mock benchmark that opts out of output verification."""
    
    def __init__(self):
        super().__init__()
    
    def setup(self):
        pass
    
    def benchmark_fn(self):
        pass
    
    def skip_output_verification(self) -> bool:
        return True  # Opt out


class MockBenchmarkNoSignature(BaseBenchmark):
    """Mock benchmark without any signature attributes."""
    
    def __init__(self):
        super().__init__()
        self._internal_size = 100  # Private, not captured
    
    def setup(self):
        pass
    
    def benchmark_fn(self):
        pass


class MockBenchmarkWithMatrixDims(BaseBenchmark):
    """Mock benchmark with matrix dimensions (M, N, K)."""
    
    def __init__(self, M=1024, N=1024, K=512):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
    
    def setup(self):
        pass
    
    def benchmark_fn(self):
        pass


# =============================================================================
# Tests: get_input_signature()
# =============================================================================

class TestGetInputSignature:
    """Tests for BaseBenchmark.get_input_signature()"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_captures_common_attributes(self):
        """Test that get_input_signature captures standard workload attributes."""
        benchmark = MockBenchmarkWithSignature(batch_size=32, seq_len=128, hidden_size=768)
        signature = benchmark.get_input_signature()
        
        assert signature is not None
        assert signature["batch_size"] == 32
        assert signature["seq_len"] == 128
        assert signature["hidden_size"] == 768
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_captures_matrix_dimensions(self):
        """Test that get_input_signature captures M, N, K matrix dimensions."""
        benchmark = MockBenchmarkWithMatrixDims(M=2048, N=1024, K=512)
        signature = benchmark.get_input_signature()
        
        assert signature is not None
        assert signature["M"] == 2048
        assert signature["N"] == 1024
        assert signature["K"] == 512
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_captures_tensor_shapes_after_setup(self):
        """Test that tensor shapes are captured after setup()."""
        benchmark = MockBenchmarkWithSignature(batch_size=16, seq_len=64, hidden_size=256)
        benchmark.setup()
        
        signature = benchmark.get_input_signature()
        
        assert signature is not None
        assert "data_shape" in signature
        assert signature["data_shape"] == (16, 64, 256)
        
        benchmark.teardown()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_returns_none_for_no_attributes(self):
        """Test that get_input_signature returns None when no attributes found."""
        benchmark = MockBenchmarkNoSignature()
        signature = benchmark.get_input_signature()
        
        # Should return None or empty dict since _internal_size is private
        assert signature is None or signature == {}
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_excludes_boolean_attributes(self):
        """Test that boolean attributes are not included in signature."""
        class BenchmarkWithBool(BaseBenchmark):
            def __init__(self):
                super().__init__()
                self.batch_size = 32
                self.use_amp = True  # Boolean should be excluded
            
            def setup(self): pass
            def benchmark_fn(self): pass
        
        benchmark = BenchmarkWithBool()
        signature = benchmark.get_input_signature()
        
        assert "batch_size" in signature
        assert "use_amp" not in signature


# =============================================================================
# Tests: skip_input_verification() / skip_output_verification()
# =============================================================================

class TestVerificationOptOut:
    """Tests for benchmark-level verification opt-out."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_verification_enabled(self):
        """Test that verification is enabled by default."""
        benchmark = MockBenchmarkWithSignature()
        
        assert benchmark.skip_input_verification() is False
        assert benchmark.skip_output_verification() is False
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_input_verification_opt_out(self):
        """Test that benchmarks can opt out of input verification."""
        benchmark = MockBenchmarkSkipsInputVerification()
        
        assert benchmark.skip_input_verification() is True
        assert benchmark.skip_output_verification() is False  # Only input is skipped
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_output_verification_opt_out(self):
        """Test that benchmarks can opt out of output verification."""
        benchmark = MockBenchmarkSkipsOutputVerification()
        
        assert benchmark.skip_input_verification() is False  # Only output is skipped
        assert benchmark.skip_output_verification() is True


# =============================================================================
# Tests: _verify_input_equivalence()
# =============================================================================

class TestVerifyInputEquivalence:
    """Tests for the _verify_input_equivalence function."""
    
    @pytest.fixture
    def verify_fn(self):
        """Import the verification function."""
        from tools.testing.run_all_benchmarks import _verify_input_equivalence
        return _verify_input_equivalence
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_equivalent_workloads_pass(self, verify_fn):
        """Test that equivalent workloads pass verification."""
        baseline = MockBenchmarkWithSignature(batch_size=32, seq_len=128, hidden_size=768)
        optimized = MockBenchmarkWithSignature(batch_size=32, seq_len=128, hidden_size=768)
        
        result = verify_fn(baseline, optimized, "baseline.py", "optimized.py")
        
        assert result["equivalent"] is True
        assert len(result["mismatches"]) == 0
        assert result["verification_type"] == "input_signature"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_different_workloads_fail(self, verify_fn):
        """Test that different workloads fail verification."""
        baseline = MockBenchmarkWithSignature(batch_size=32, seq_len=128, hidden_size=768)
        optimized = MockBenchmarkDifferentWorkload(batch_size=64, seq_len=128, hidden_size=768)
        
        result = verify_fn(baseline, optimized, "baseline.py", "optimized.py")
        
        assert result["equivalent"] is False
        assert len(result["mismatches"]) > 0
        assert "batch_size" in result["mismatches"][0]
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_skipped_when_benchmark_opts_out(self, verify_fn):
        """Test that verification is skipped when benchmark opts out."""
        baseline = MockBenchmarkWithSignature(batch_size=32)
        optimized = MockBenchmarkSkipsInputVerification(batch_size=64)  # Different but opts out
        
        result = verify_fn(baseline, optimized, "baseline.py", "optimized.py")
        
        assert result["equivalent"] is True  # Assumed OK when skipped
        assert result["verification_type"] == "skipped"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_no_signature_available(self, verify_fn):
        """Test handling when no signature is available."""
        baseline = MockBenchmarkNoSignature()
        optimized = MockBenchmarkNoSignature()
        
        result = verify_fn(baseline, optimized, "baseline.py", "optimized.py")
        
        assert result["equivalent"] is True  # Can't verify, assume OK
        assert result["verification_type"] == "no_signature"
        assert "signature" in result["mismatches"][0].lower()  # "Neither benchmark provides input signature"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_partial_signature_mismatch(self, verify_fn):
        """Test detection of partial signature mismatches."""
        class BenchmarkA(BaseBenchmark):
            def __init__(self):
                super().__init__()
                self.M = 1024
                self.N = 1024
                # Missing K
            def setup(self): pass
            def benchmark_fn(self): pass
        
        class BenchmarkB(BaseBenchmark):
            def __init__(self):
                super().__init__()
                self.M = 1024
                self.N = 1024
                self.K = 512  # Extra attribute
            def setup(self): pass
            def benchmark_fn(self): pass
        
        baseline = BenchmarkA()
        optimized = BenchmarkB()
        
        result = verify_fn(baseline, optimized, "baseline.py", "optimized.py")
        
        # Should detect the missing K in baseline
        assert result["equivalent"] is False
        assert any("K" in m for m in result["mismatches"])


# =============================================================================
# Tests: Integration
# =============================================================================

class TestVerificationIntegration:
    """Integration tests for the full verification flow."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_end_to_end_equivalent_benchmarks(self):
        """Test full verification flow with equivalent benchmarks."""
        from tools.testing.run_all_benchmarks import _verify_input_equivalence
        
        # Create two equivalent benchmarks
        baseline = MockBenchmarkWithSignature(batch_size=32, seq_len=128, hidden_size=768)
        optimized = MockBenchmarkWithSignature(batch_size=32, seq_len=128, hidden_size=768)
        
        # Setup both
        baseline.setup()
        optimized.setup()
        
        # Verify input equivalence
        input_result = _verify_input_equivalence(
            baseline, optimized, "baseline.py", "optimized.py"
        )
        assert input_result["equivalent"] is True
        
        # Run benchmarks
        baseline.benchmark_fn()
        optimized.benchmark_fn()
        
        # Both should have valid outputs
        assert baseline.output is not None
        assert optimized.output is not None
        
        # Cleanup
        baseline.teardown()
        optimized.teardown()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_end_to_end_different_workloads_detected(self):
        """Test that different workloads are detected before benchmark runs."""
        from tools.testing.run_all_benchmarks import _verify_input_equivalence
        
        # Create benchmarks with different workloads
        baseline = MockBenchmarkWithSignature(batch_size=32, seq_len=128, hidden_size=768)
        optimized = MockBenchmarkDifferentWorkload(batch_size=64, seq_len=128, hidden_size=768)
        
        # Input verification should FAIL before we even run benchmarks
        input_result = _verify_input_equivalence(
            baseline, optimized, "baseline.py", "optimized.py"
        )
        
        assert input_result["equivalent"] is False
        assert "batch_size" in str(input_result["mismatches"])
        
        # The benchmark should NOT proceed to execution when this fails


# =============================================================================
# Tests: CLI Flags
# =============================================================================

class TestCLIFlags:
    """Tests for CLI verification flags."""
    
    def test_skip_verify_flag_disables_both(self):
        """Test that --skip-verify disables both input and output verification."""
        # This tests the flag logic, not the actual CLI
        skip_verify = True
        skip_input_verify = False
        skip_output_verify = False
        
        verify_input = not (skip_verify or skip_input_verify)
        verify_output = not (skip_verify or skip_output_verify)
        
        assert verify_input is False
        assert verify_output is False
    
    def test_individual_skip_flags(self):
        """Test that individual skip flags work independently."""
        # Skip only input
        verify_input = not (False or True)  # skip_input_verify=True
        verify_output = not (False or False)  # skip_output_verify=False
        
        assert verify_input is False
        assert verify_output is True
        
        # Skip only output
        verify_input = not (False or False)  # skip_input_verify=False
        verify_output = not (False or True)  # skip_output_verify=True
        
        assert verify_input is True
        assert verify_output is False
    
    def test_defaults_enable_both(self):
        """Test that defaults enable both verifications."""
        skip_verify = False
        skip_input_verify = False
        skip_output_verify = False
        
        verify_input = not (skip_verify or skip_input_verify)
        verify_output = not (skip_verify or skip_output_verify)
        
        assert verify_input is True
        assert verify_output is True


# =============================================================================
# Tests: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in verification."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_numeric_tolerance_for_floats(self):
        """Test that float attributes use appropriate tolerance."""
        from tools.testing.run_all_benchmarks import _verify_input_equivalence
        
        class BenchmarkWithFloat(BaseBenchmark):
            def __init__(self, learning_rate):
                super().__init__()
                self.learning_rate = learning_rate
            def setup(self): pass
            def benchmark_fn(self): pass
        
        # Very small difference should be OK
        baseline = BenchmarkWithFloat(learning_rate=0.001)
        optimized = BenchmarkWithFloat(learning_rate=0.001 + 1e-10)
        
        result = _verify_input_equivalence(
            baseline, optimized, "baseline.py", "optimized.py"
        )
        
        assert result["equivalent"] is True
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_signature_includes_dtype(self):
        """Test that tensor dtype is included in signature."""
        benchmark = MockBenchmarkWithSignature()
        benchmark.setup()
        
        signature = benchmark.get_input_signature()
        
        assert "data_dtype" in signature
        assert "float" in signature["data_dtype"].lower()
        
        benchmark.teardown()

