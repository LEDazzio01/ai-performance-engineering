"""
Tests for the profiling module.

Run with: pytest tests/test_profiling.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestProfileSession:
    """Test ProfileSession dataclass."""
    
    def test_session_creation(self):
        """Test creating a profile session."""
        from tools.profiling.profiler import ProfileSession, KernelInfo
        
        session = ProfileSession(
            total_time_ms=100.0,
            cuda_time_ms=90.0,
            cpu_time_ms=10.0,
            peak_memory_mb=1024.0,
            allocated_memory_mb=800.0,
            reserved_memory_mb=1200.0,
        )
        
        assert session.total_time_ms == 100.0
        assert session.cuda_time_ms == 90.0
        assert session.peak_memory_mb == 1024.0
    
    def test_session_to_dict(self):
        """Test converting session to dictionary."""
        from tools.profiling.profiler import ProfileSession
        
        session = ProfileSession(
            total_time_ms=100.0,
            cuda_time_ms=90.0,
            cpu_time_ms=10.0,
            peak_memory_mb=1024.0,
            allocated_memory_mb=800.0,
            reserved_memory_mb=1200.0,
            device_name="Test GPU",
            timestamp="2024-01-01T00:00:00",
        )
        
        data = session.to_dict()
        
        assert data["timing"]["total_ms"] == 100.0
        assert data["memory"]["peak_mb"] == 1024.0
        assert data["device"] == "Test GPU"


class TestKernelInfo:
    """Test KernelInfo dataclass."""
    
    def test_kernel_info_creation(self):
        """Test creating kernel info."""
        from tools.profiling.profiler import KernelInfo
        
        kernel = KernelInfo(
            name="volta_h884gemm",
            duration_us=1000.0,
            cuda_time_us=1000.0,
            cpu_time_us=50.0,
            call_count=100,
        )
        
        assert kernel.name == "volta_h884gemm"
        assert kernel.duration_us == 1000.0
        assert kernel.call_count == 100


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass."""
    
    def test_snapshot_creation(self):
        """Test creating memory snapshot."""
        from tools.profiling.memory import MemorySnapshot
        
        snapshot = MemorySnapshot(
            timestamp_ms=100.0,
            allocated_bytes=1024 * 1024 * 100,
            reserved_bytes=1024 * 1024 * 200,
            peak_allocated_bytes=1024 * 1024 * 150,
            peak_reserved_bytes=1024 * 1024 * 250,
        )
        
        assert snapshot.timestamp_ms == 100.0
        assert snapshot.allocated_bytes == 1024 * 1024 * 100
    
    def test_snapshot_to_dict(self):
        """Test converting snapshot to dictionary."""
        from tools.profiling.memory import MemorySnapshot
        
        snapshot = MemorySnapshot(
            timestamp_ms=100.0,
            allocated_bytes=1024 * 1024 * 100,  # 100MB
            reserved_bytes=1024 * 1024 * 200,
            peak_allocated_bytes=1024 * 1024 * 150,
            peak_reserved_bytes=1024 * 1024 * 250,
        )
        
        data = snapshot.to_dict()
        
        assert data["timestamp_ms"] == 100.0
        assert data["allocated_mb"] == 100.0


class TestMemoryProfiler:
    """Test MemoryProfiler functionality."""
    
    def test_profiler_initialization(self):
        """Test memory profiler initialization."""
        from tools.profiling.memory import MemoryProfiler
        
        profiler = MemoryProfiler(
            sample_interval_ms=1.0,
            record_history=True,
        )
        
        assert profiler.sample_interval_ms == 1.0
        assert profiler.record_history
    
    def test_profiler_reset(self):
        """Test profiler reset."""
        from tools.profiling.memory import MemoryProfiler
        
        profiler = MemoryProfiler()
        profiler._timeline = [Mock()]  # Add dummy data
        profiler._markers = {"test": (0, 100)}
        
        profiler.reset()
        
        assert len(profiler._timeline) == 0
        assert len(profiler._markers) == 0
    
    def test_get_peak_analysis_empty(self):
        """Test peak analysis with no data."""
        from tools.profiling.memory import MemoryProfiler
        
        profiler = MemoryProfiler()
        peak = profiler.get_peak_analysis()
        
        assert peak == {}


class TestFlameGraphGenerator:
    """Test FlameGraphGenerator functionality."""
    
    def test_generator_initialization(self):
        """Test flame graph generator initialization."""
        from tools.profiling.flame_graph import FlameGraphGenerator
        
        generator = FlameGraphGenerator(
            min_duration_us=10.0,
            max_depth=50,
            group_small_kernels=True,
        )
        
        assert generator.min_duration_us == 10.0
        assert generator.max_depth == 50
    
    def test_clean_kernel_name(self):
        """Test kernel name cleaning."""
        from tools.profiling.flame_graph import FlameGraphGenerator
        
        generator = FlameGraphGenerator()
        
        # Test namespace removal
        assert generator._clean_kernel_name("at::native::matmul") == "matmul"
        
        # Test empty name
        assert generator._clean_kernel_name("") == "unknown"
        assert generator._clean_kernel_name(None) == "unknown"
        
        # Test truncation
        long_name = "a" * 100
        cleaned = generator._clean_kernel_name(long_name)
        assert len(cleaned) <= 60
    
    def test_extract_kernel_type(self):
        """Test kernel type extraction."""
        from tools.profiling.flame_graph import FlameGraphGenerator
        
        generator = FlameGraphGenerator()
        
        assert generator._extract_kernel_type("volta_h884gemm_128x128") == "Matrix Multiply"
        assert generator._extract_kernel_type("cudnn_conv_fwd") == "Convolution"
        assert generator._extract_kernel_type("softmax_kernel") == "Softmax"
        assert generator._extract_kernel_type("relu_forward") == "Activation"
        assert generator._extract_kernel_type("layernorm_kernel") == "Normalization"
        assert generator._extract_kernel_type("unknown_op") == "Other"
    
    def test_from_kernel_list(self):
        """Test flame graph from kernel list."""
        from tools.profiling.flame_graph import FlameGraphGenerator
        
        generator = FlameGraphGenerator()
        
        kernels = [
            {"name": "matmul_kernel", "time_us": 1000},
            {"name": "softmax_kernel", "time_us": 500},
            {"name": "relu_kernel", "time_us": 200},
        ]
        
        data = generator.from_kernel_list(kernels)
        
        assert data["name"] == "GPU Execution"
        assert data["value"] > 0
        assert len(data["children"]) > 0
    
    def test_from_chrome_trace(self, tmp_path):
        """Test flame graph from Chrome trace."""
        from tools.profiling.flame_graph import FlameGraphGenerator
        
        # Create sample trace
        trace = {
            "traceEvents": [
                {"name": "matmul", "cat": "kernel", "ph": "X", "ts": 0, "dur": 1000},
                {"name": "softmax", "cat": "kernel", "ph": "X", "ts": 1000, "dur": 500},
            ]
        }
        
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps(trace))
        
        generator = FlameGraphGenerator(min_duration_us=0)
        data = generator.from_chrome_trace(trace_path)
        
        assert data["name"] == "GPU Execution"
        assert data["value"] > 0
    
    def test_export_json(self, tmp_path):
        """Test JSON export."""
        from tools.profiling.flame_graph import FlameGraphGenerator
        
        generator = FlameGraphGenerator()
        data = {"name": "root", "value": 100, "children": []}
        
        output_path = tmp_path / "flame.json"
        generator.export(data, output_path, format="json")
        
        assert output_path.exists()
        loaded = json.loads(output_path.read_text())
        assert loaded["name"] == "root"
    
    def test_export_html(self, tmp_path):
        """Test HTML export."""
        from tools.profiling.flame_graph import FlameGraphGenerator
        
        generator = FlameGraphGenerator()
        data = {"name": "root", "value": 100, "children": []}
        
        output_path = tmp_path / "flame.html"
        generator.export(data, output_path, format="html")
        
        assert output_path.exists()
        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "d3-flame-graph" in content


class TestTimelineGenerator:
    """Test TimelineGenerator functionality."""
    
    def test_generator_initialization(self):
        """Test timeline generator initialization."""
        from tools.profiling.timeline import TimelineGenerator
        
        generator = TimelineGenerator(
            min_duration_us=1.0,
            include_python_events=True,
        )
        
        assert generator.min_duration_us == 1.0
        assert generator.include_python_events
    
    def test_classify_category(self):
        """Test event category classification."""
        from tools.profiling.timeline import TimelineGenerator, EventType
        
        generator = TimelineGenerator()
        
        assert generator._classify_category("kernel") == EventType.CUDA_KERNEL
        assert generator._classify_category("cuda") == EventType.CUDA_KERNEL
        assert generator._classify_category("memcpy") == EventType.CUDA_MEMCPY
        assert generator._classify_category("python") == EventType.PYTHON
        assert generator._classify_category("cpu_op") == EventType.CPU_OP
    
    def test_from_chrome_trace(self, tmp_path):
        """Test timeline from Chrome trace."""
        from tools.profiling.timeline import TimelineGenerator
        
        trace = {
            "traceEvents": [
                {"name": "cpu_op", "cat": "cpu", "ph": "X", "ts": 0, "dur": 100},
                {"name": "kernel", "cat": "cuda", "ph": "X", "ts": 50, "dur": 200},
            ]
        }
        
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps(trace))
        
        generator = TimelineGenerator(min_duration_us=0)
        timeline = generator.from_chrome_trace(trace_path)
        
        assert timeline.total_time_us > 0
        assert len(timeline.events) >= 2
    
    def test_calculate_active_time(self):
        """Test active time calculation with overlaps."""
        from tools.profiling.timeline import TimelineGenerator, TimelineEvent, EventType
        
        generator = TimelineGenerator()
        
        # Overlapping events
        events = [
            TimelineEvent("a", EventType.CUDA_KERNEL, 0, 100),
            TimelineEvent("b", EventType.CUDA_KERNEL, 50, 100),  # Overlaps with a
            TimelineEvent("c", EventType.CUDA_KERNEL, 200, 50),  # No overlap
        ]
        
        active_time = generator._calculate_active_time(events)
        
        # Should be 150 + 50 = 200 (merged first two, plus third)
        assert active_time == 200


class TestHTAAnalyzer:
    """Test HTAAnalyzer functionality."""
    
    def test_analyzer_initialization(self):
        """Test HTA analyzer initialization."""
        from tools.profiling.hta_integration import HTAAnalyzer
        
        analyzer = HTAAnalyzer(output_dir=Path("/tmp/hta"))
        
        assert analyzer.output_dir == Path("/tmp/hta")
    
    def test_manual_trace_analysis(self, tmp_path):
        """Test manual trace analysis fallback."""
        from tools.profiling.hta_integration import HTAAnalyzer
        
        trace = {
            "traceEvents": [
                {"name": "matmul", "cat": "kernel", "ph": "X", "ts": 0, "dur": 1000},
                {"name": "nccl_allreduce", "cat": "comm", "ph": "X", "ts": 1000, "dur": 500},
            ]
        }
        
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps(trace))
        
        analyzer = HTAAnalyzer()
        report = analyzer._manual_trace_analysis(trace_path)
        
        assert report.compute_time_pct > 0
        assert len(report.top_kernels) > 0
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        from tools.profiling.hta_integration import HTAAnalyzer, HTAReport
        
        analyzer = HTAAnalyzer()
        
        # High idle time
        report = HTAReport(gpu_idle_time_pct=50)
        recs = analyzer._generate_recommendations(report)
        assert any("idle" in r.lower() for r in recs)
        
        # High communication
        report = HTAReport(communication_time_pct=30)
        recs = analyzer._generate_recommendations(report)
        assert any("communication" in r.lower() for r in recs)


class TestTorchCompileAnalyzer:
    """Test TorchCompileAnalyzer functionality."""
    
    def test_analyzer_initialization(self):
        """Test compile analyzer initialization."""
        from tools.profiling.torch_compile import TorchCompileAnalyzer
        
        analyzer = TorchCompileAnalyzer(
            backend="inductor",
            mode="default",
        )
        
        assert analyzer.backend == "inductor"
        assert analyzer.mode == "default"
    
    def test_get_break_suggestion(self):
        """Test graph break suggestions."""
        from tools.profiling.torch_compile import TorchCompileAnalyzer
        
        analyzer = TorchCompileAnalyzer()
        
        assert "static" in analyzer._get_break_suggestion("data-dependent control flow").lower()
        assert "print" in analyzer._get_break_suggestion("print statement").lower()
        # numpy operations get a generic suggestion about supported alternatives
        assert "supported" in analyzer._get_break_suggestion("unsupported numpy operation").lower()
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        from tools.profiling.torch_compile import TorchCompileAnalyzer, CompileReport
        
        analyzer = TorchCompileAnalyzer()
        
        # Low speedup
        report = CompileReport(speedup=0.8, total_graph_breaks=0)
        recs = analyzer._generate_recommendations(report)
        assert any("slower" in r.lower() for r in recs)
        
        # Graph breaks
        report = CompileReport(speedup=1.5, total_graph_breaks=5)
        recs = analyzer._generate_recommendations(report)
        assert any("graph break" in r.lower() for r in recs)
        
        # Good speedup
        report = CompileReport(speedup=2.5, total_graph_breaks=0)
        recs = analyzer._generate_recommendations(report)
        assert any("excellent" in r.lower() or "working well" in r.lower() for r in recs)


class TestCompileReport:
    """Test CompileReport dataclass."""
    
    def test_report_creation(self):
        """Test creating compile report."""
        from tools.profiling.torch_compile import CompileReport
        
        report = CompileReport(
            speedup=1.5,
            compile_time_ms=5000,
            total_graph_breaks=2,
            fusion_ratio=3.0,
            ops_before_fusion=100,
            ops_after_fusion=33,
        )
        
        assert report.speedup == 1.5
        assert report.fusion_ratio == 3.0
    
    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        from tools.profiling.torch_compile import CompileReport
        
        report = CompileReport(
            speedup=1.5,
            compile_time_ms=5000,
            backend="inductor",
            mode="default",
        )
        
        data = report.to_dict()
        
        assert data["performance"]["speedup"] == 1.5
        assert data["compilation"]["backend"] == "inductor"


# Fixtures

@pytest.fixture
def sample_trace_data():
    """Sample Chrome trace data."""
    return {
        "traceEvents": [
            {"name": "matmul", "cat": "kernel", "ph": "X", "ts": 0, "dur": 1000, "pid": 0, "tid": 0},
            {"name": "softmax", "cat": "kernel", "ph": "X", "ts": 1000, "dur": 500, "pid": 0, "tid": 0},
            {"name": "cpu_op", "cat": "cpu", "ph": "X", "ts": 0, "dur": 200, "pid": 1, "tid": 0},
        ]
    }


@pytest.fixture
def trace_file(tmp_path, sample_trace_data):
    """Create a temporary trace file."""
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(sample_trace_data))
    return trace_path

