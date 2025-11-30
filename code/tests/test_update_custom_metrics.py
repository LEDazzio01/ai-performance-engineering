#!/usr/bin/env python3
"""Unit tests for core/scripts/update_custom_metrics.py.

Run with: pytest tests/test_update_custom_metrics.py -v
"""

import pytest
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from core.scripts.update_custom_metrics import (
    get_chapter_from_path,
    has_conditional_none_return,
    analyze_get_custom_metrics,
    generate_helper_code,
    CHAPTER_METRIC_HELPERS,
    HELPER_SIGNATURES,
)


class TestGetChapterFromPath:
    """Test chapter number extraction from paths."""
    
    def test_standard_chapter_path(self):
        """Should extract chapter from standard paths."""
        assert get_chapter_from_path(Path("ch07/baseline_memory.py")) == 7
        assert get_chapter_from_path(Path("ch10/optimized_pipeline.py")) == 10
        assert get_chapter_from_path(Path("ch01/baseline_perf.py")) == 1
    
    def test_nested_path(self):
        """Should extract chapter from nested paths."""
        assert get_chapter_from_path(Path("/code/ch07/baseline.py")) == 7
        assert get_chapter_from_path(Path("some/dir/ch15/file.py")) == 15
    
    def test_no_chapter(self):
        """Should return None for paths without chapter."""
        assert get_chapter_from_path(Path("core/benchmark/utils.py")) is None
        assert get_chapter_from_path(Path("labs/test.py")) is None


class TestHasConditionalNoneReturn:
    """Test detection of conditional None returns."""
    
    def test_conditional_none_if_not(self):
        """Should detect 'if not x: return None' pattern."""
        content = '''
def get_custom_metrics(self) -> Optional[dict]:
    if not hasattr(self, '_data'):
        return None
    return {"metric": 1.0}
'''
        assert has_conditional_none_return(content) is True
    
    def test_conditional_none_if_is_none(self):
        """Should detect 'if x is None: return None' pattern."""
        content = '''
def get_custom_metrics(self) -> Optional[dict]:
    if self._result is None:
        return None
    return {"metric": 1.0}
'''
        assert has_conditional_none_return(content) is True
    
    def test_unconditional_none(self):
        """Should not flag unconditional None returns."""
        content = '''
def get_custom_metrics(self) -> Optional[dict]:
    return None
'''
        assert has_conditional_none_return(content) is False
    
    def test_no_none_return(self):
        """Should not flag methods without None return."""
        content = '''
def get_custom_metrics(self) -> Optional[dict]:
    return {"metric": 1.0}
'''
        assert has_conditional_none_return(content) is False


class TestAnalyzeGetCustomMetrics:
    """Test analysis of get_custom_metrics implementations."""
    
    def test_uses_helper_function(self):
        """Should detect use of helper functions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
from core.benchmark.metrics import compute_memory_transfer_metrics

class MyBenchmark:
    def get_custom_metrics(self):
        return compute_memory_transfer_metrics(1000, 1.0)
''')
            f.flush()
            result = analyze_get_custom_metrics(Path(f.name), Path(f.name).parent)
            assert result["uses_helper"] is True
    
    def test_returns_none(self):
        """Should detect None return."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
class MyBenchmark:
    def get_custom_metrics(self):
        return None
''')
            f.flush()
            result = analyze_get_custom_metrics(Path(f.name), Path(f.name).parent)
            assert result["returns_none"] is True
    
    def test_returns_empty_dict(self):
        """Should detect empty dict return."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
class MyBenchmark:
    def get_custom_metrics(self):
        return {}
''')
            f.flush()
            result = analyze_get_custom_metrics(Path(f.name), Path(f.name).parent)
            assert result["returns_empty"] is True
    
    def test_has_method(self):
        """Should detect presence of method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
class MyBenchmark:
    def get_custom_metrics(self):
        return {"a": 1}
''')
            f.flush()
            result = analyze_get_custom_metrics(Path(f.name), Path(f.name).parent)
            assert result["has_method"] is True
    
    def test_no_method(self):
        """Should detect missing method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
class MyBenchmark:
    def setup(self):
        pass
''')
            f.flush()
            result = analyze_get_custom_metrics(Path(f.name), Path(f.name).parent)
            assert result["has_method"] is False


class TestGenerateHelperCode:
    """Test code generation for helper functions."""
    
    def test_generates_valid_code(self):
        """Should generate syntactically valid code."""
        for helper_name in HELPER_SIGNATURES.keys():
            code = generate_helper_code(helper_name, "        ")
            assert "return" in code
            assert helper_name in code
    
    def test_includes_parameters(self):
        """Should include all parameters."""
        code = generate_helper_code("compute_memory_transfer_metrics", "")
        assert "bytes_transferred" in code
        assert "elapsed_ms" in code
    
    def test_proper_indentation(self):
        """Should use specified indentation."""
        code = generate_helper_code("compute_memory_transfer_metrics", "    ")
        assert code is not None, "generate_helper_code returned None"
        lines = code.split("\n")
        for line in lines:
            if line.strip():
                assert line.startswith("    ") or line.startswith("return")


class TestChapterMetricHelpers:
    """Test chapter to helper mapping."""
    
    def test_all_chapters_have_mapping(self):
        """All chapters 1-20 should have a mapping."""
        for ch in range(1, 21):
            assert ch in CHAPTER_METRIC_HELPERS, f"Missing mapping for ch{ch}"
    
    def test_helpers_exist_in_signatures(self):
        """All mapped helpers should have signatures defined."""
        for ch, helper in CHAPTER_METRIC_HELPERS.items():
            if helper is not None:
                assert helper in HELPER_SIGNATURES, f"Missing signature for {helper}"
    
    def test_no_none_helpers(self):
        """With new helpers, no chapter should have None."""
        for ch, helper in CHAPTER_METRIC_HELPERS.items():
            assert helper is not None, f"Ch{ch} still has None helper"


class TestHelperSignatures:
    """Test helper function signatures."""
    
    def test_all_have_import(self):
        """All signatures should have import statement."""
        for name, sig in HELPER_SIGNATURES.items():
            assert "import" in sig, f"Missing import for {name}"
            assert "from core.benchmark.metrics" in sig["import"]
    
    def test_all_have_params(self):
        """All signatures should have params list."""
        for name, sig in HELPER_SIGNATURES.items():
            assert "params" in sig, f"Missing params for {name}"
            assert isinstance(sig["params"], list)
    
    def test_all_have_defaults(self):
        """All signatures should have defaults dict."""
        for name, sig in HELPER_SIGNATURES.items():
            assert "defaults" in sig, f"Missing defaults for {name}"
            assert isinstance(sig["defaults"], dict)
    
    def test_defaults_cover_params(self):
        """All params should have default values."""
        for name, sig in HELPER_SIGNATURES.items():
            for param in sig["params"]:
                assert param in sig["defaults"], f"Missing default for {param} in {name}"


class TestIntegration:
    """Integration tests with real codebase patterns."""
    
    def test_analyze_real_pattern_baseline(self):
        """Should correctly analyze typical baseline pattern."""
        content = '''
#!/usr/bin/env python3
"""Baseline benchmark."""

from core.harness.benchmark_harness import BaseBenchmark

class BaselineBenchmark(BaseBenchmark):
    def setup(self):
        pass
    
    def benchmark_fn(self):
        pass
    
    def get_custom_metrics(self):
        if not hasattr(self, '_last_result'):
            return None
        return {
            "metric.value": self._last_result,
        }
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f.flush()
            result = analyze_get_custom_metrics(Path(f.name), Path(f.name).parent)
            
            assert result["has_method"] is True
            assert result["has_conditional_none"] is True
            assert result["returns_none"] is False  # Conditional doesn't count
    
    def test_analyze_real_pattern_optimized(self):
        """Should correctly analyze typical optimized pattern with helper."""
        content = '''
#!/usr/bin/env python3
"""Optimized benchmark."""

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.metrics import compute_memory_transfer_metrics

class OptimizedBenchmark(BaseBenchmark):
    def setup(self):
        self.N = 1024
    
    def benchmark_fn(self):
        pass
    
    def get_custom_metrics(self):
        return compute_memory_transfer_metrics(
            bytes_transferred=self.N * 4,
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
        )
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f.flush()
            result = analyze_get_custom_metrics(Path(f.name), Path(f.name).parent)
            
            assert result["has_method"] is True
            assert result["uses_helper"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
