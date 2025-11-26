"""
Tests for the auto-optimizer module.

Run with: pytest tests/test_optimize.py -v
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestInputAdapters:
    """Test input adapter functionality."""
    
    def test_file_adapter_single_file(self, tmp_path):
        """Test FileAdapter with a single file."""
        from tools.optimize.input_adapters import FileAdapter
        
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")
        
        adapter = FileAdapter(paths=[test_file])
        sources = list(adapter.get_sources())
        
        assert len(sources) == 1
        assert sources[0].name == "test"
        assert sources[0].content == "print('hello')"
    
    def test_file_adapter_multiple_files(self, tmp_path):
        """Test FileAdapter with multiple files."""
        from tools.optimize.input_adapters import FileAdapter
        
        # Create test files
        for i in range(3):
            (tmp_path / f"test_{i}.py").write_text(f"# File {i}")
        
        adapter = FileAdapter(paths=list(tmp_path.glob("*.py")))
        sources = list(adapter.get_sources())
        
        assert len(sources) == 3
    
    def test_file_adapter_output(self, tmp_path):
        """Test FileAdapter output writing."""
        from tools.optimize.input_adapters import FileAdapter, CodeSource
        
        adapter = FileAdapter(
            paths=[tmp_path / "test.py"],
            output_dir=tmp_path / "output",
            suffix="_optimized"
        )
        
        source = CodeSource(
            path=tmp_path / "test.py",
            content="original",
            name="test"
        )
        
        output_path = adapter.write_output(source, "optimized code")
        
        assert output_path.exists()
        assert output_path.read_text() == "optimized code"
        assert "_optimized" in output_path.name
    
    def test_benchmark_adapter(self, tmp_path):
        """Test BenchmarkAdapter for benchmark directories."""
        from tools.optimize.input_adapters import BenchmarkAdapter
        
        # Create benchmark structure
        (tmp_path / "baseline_test.py").write_text("# baseline")
        (tmp_path / "optimized_test.py").write_text("# optimized")
        
        adapter = BenchmarkAdapter(
            directory=tmp_path,
            threshold=1.1,
            pattern="optimized_*.py"
        )
        
        sources = list(adapter.get_sources())
        assert len(sources) == 1
        assert "test" in sources[0].name
    
    def test_detect_input_type_file(self, tmp_path):
        """Test input type detection for files."""
        from tools.optimize.input_adapters import detect_input_type, FileAdapter
        
        test_file = tmp_path / "test.py"
        test_file.write_text("# test")
        
        input_type, adapter = detect_input_type(str(test_file))
        
        assert input_type == "file"
        assert isinstance(adapter, FileAdapter)
    
    def test_detect_input_type_repo(self):
        """Test input type detection for repo URLs."""
        from tools.optimize.input_adapters import detect_input_type, RepoAdapter
        
        input_type, adapter = detect_input_type("https://github.com/user/repo")
        
        assert input_type == "repo"
        assert isinstance(adapter, RepoAdapter)
    
    def test_detect_input_type_benchmark_dir(self, tmp_path):
        """Test input type detection for benchmark directories."""
        from tools.optimize.input_adapters import detect_input_type, BenchmarkAdapter
        
        # Create benchmark structure
        (tmp_path / "baseline_test.py").write_text("# baseline")
        (tmp_path / "optimized_test.py").write_text("# optimized")
        
        input_type, adapter = detect_input_type(str(tmp_path))
        
        assert input_type == "benchmark"
        assert isinstance(adapter, BenchmarkAdapter)


class TestAutoOptimizer:
    """Test AutoOptimizer core functionality."""
    
    def test_optimizer_initialization(self):
        """Test optimizer can be initialized."""
        from tools.optimize import AutoOptimizer
        
        optimizer = AutoOptimizer(
            llm_provider="anthropic",
            max_iterations=2,
            target_speedup=1.5,
            verbose=False,
        )
        
        assert optimizer.llm_provider == "anthropic"
        assert optimizer.max_iterations == 2
        assert optimizer.target_speedup == 1.5
    
    def test_bottleneck_analysis(self):
        """Test static bottleneck analysis."""
        from tools.optimize.optimizer import AutoOptimizer
        
        optimizer = AutoOptimizer(verbose=False)
        
        code_with_issues = """
import torch
for i in range(100):
    x = tensor.item()  # GPU→CPU sync
    y = tensor.cpu()   # Transfer
"""
        
        bottlenecks = optimizer._analyze_bottlenecks(code_with_issues)
        
        assert any("loop" in b.lower() for b in bottlenecks)
        assert any(".item()" in b for b in bottlenecks)
        assert any("cpu" in b.lower() for b in bottlenecks)
    
    def test_bottleneck_analysis_missing_optimizations(self):
        """Test detection of missing optimizations."""
        from tools.optimize.optimizer import AutoOptimizer
        
        optimizer = AutoOptimizer(verbose=False)
        
        code_without_optimizations = """
import torch
def forward(x):
    return x + 1
"""
        
        bottlenecks = optimizer._analyze_bottlenecks(code_without_optimizations)
        
        assert any("torch.compile" in b for b in bottlenecks)
        assert any("precision" in b.lower() for b in bottlenecks)
    
    def test_detect_benchmark_fn(self):
        """Test benchmark function detection."""
        from tools.optimize.optimizer import AutoOptimizer
        
        optimizer = AutoOptimizer(verbose=False)
        
        code = """
class MyBenchmark:
    def benchmark_fn(self):
        pass
    
    def forward(self):
        pass
"""
        
        fn_name = optimizer._detect_benchmark_fn(code)
        assert "benchmark" in fn_name.lower() or "forward" in fn_name.lower()


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating an optimization result."""
        from tools.optimize.optimizer import OptimizationResult
        
        result = OptimizationResult(
            success=True,
            original_code="# original",
            optimized_code="# optimized",
            original_time_ms=100.0,
            optimized_time_ms=50.0,
            speedup=2.0,
            techniques_applied=["torch.compile"],
            explanation="Applied torch.compile for 2x speedup",
            profile_data={},
        )
        
        assert result.success
        assert result.speedup == 2.0
        assert "torch.compile" in result.techniques_applied


class TestProfileResult:
    """Test ProfileResult dataclass."""
    
    def test_profile_result_creation(self):
        """Test creating a profile result."""
        from tools.optimize.optimizer import ProfileResult
        
        result = ProfileResult(
            total_time_ms=100.0,
            gpu_time_ms=90.0,
            cpu_time_ms=10.0,
            memory_peak_mb=1024.0,
            memory_allocated_mb=800.0,
            kernel_times={"matmul": 50.0},
            bottlenecks=["High kernel launch rate"],
        )
        
        assert result.total_time_ms == 100.0
        assert result.memory_peak_mb == 1024.0
        assert len(result.bottlenecks) == 1


# Fixtures

@pytest.fixture
def sample_benchmark_code():
    """Sample benchmark code for testing."""
    return '''
import torch
from tools.testing.benchmark_harness import BaseBenchmark

class TestBenchmark(BaseBenchmark):
    def setup(self):
        self.x = torch.randn(1024, 1024, device='cuda')
    
    def benchmark_fn(self):
        return torch.matmul(self.x, self.x)

def get_benchmark():
    return TestBenchmark()
'''


@pytest.fixture
def tmp_benchmark_file(tmp_path, sample_benchmark_code):
    """Create a temporary benchmark file."""
    file_path = tmp_path / "benchmark.py"
    file_path.write_text(sample_benchmark_code)
    return file_path

