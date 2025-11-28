import json
import sys
from pathlib import Path

from tools.core import (
    profile_artifacts,
    compile_analysis,
    costs,
    optimization_reports,
    optimization_stack,
    whatif,
    ncu_analysis,
    profile_insights,
)


def test_profile_artifacts_empty(tmp_path: Path):
    # With no traces, loaders return default structures and messages
    empty_root = tmp_path
    assert profile_artifacts.load_flame_graph_data(empty_root).get("message")
    assert profile_artifacts.load_memory_timeline(empty_root).get("message")
    assert profile_artifacts.load_cpu_gpu_timeline(empty_root) is not None
    assert profile_artifacts.load_kernel_breakdown({"children": []})["kernels"] == []


def test_compile_analysis_empty():
    result = compile_analysis.load_compile_analysis(Path.cwd(), [])
    assert "compile_benchmarks" in result
    assert isinstance(result.get("recommendations"), list)


def test_costs_and_roi_empty():
    cost = costs.calculate_costs([], {"name": "H100"})
    assert cost["current_rate"] > 0
    roi_result = optimization_reports.compute_roi([], cost)
    assert "techniques" in roi_result


def test_optimization_stack_fallbacks():
    # These should not raise even if advanced_analysis is missing
    assert optimization_stack.get_all_optimizations()
    assert optimization_stack.get_optimization_playbooks()
    assert optimization_stack.calculate_compound_optimization([], {}) is not None
    assert optimization_stack.get_optimal_optimization_stack(2.0, "medium", {}) is not None


def test_whatif_and_ncu_empty(tmp_path: Path):
    scenarios = whatif.get_scenarios()
    assert scenarios.get("scenarios")
    ncu = ncu_analysis.load_ncu_deepdive(tmp_path)
    assert "available" in ncu


def test_profile_insights_bottlenecks_and_score():
    flame_data = {
        "value": 100.0,
        "children": [
            {"name": "gpu_memcpy", "value": 30},
            {"name": "python_function", "value": 20},
            {"name": "overhead", "value": 6},
        ],
    }
    kernel_data = {
        "summary": {"total_time_us": 80},
        "kernels": [
            {"name": "gemm_kernel", "time_us": 20},
            {"name": "copy_kernel", "time_us": 15},
        ],
    }
    hw_caps = {
        "features": [
            {"name": "TMA Copy", "supported": True, "optimization": "Use async copies"},
            {"name": "FP8 Tensor Cores", "supported": True, "optimization": "Enable FP8"},
        ],
        "architecture": "blackwell",
        "gpu": {"name": "B200"},
    }

    result = profile_insights.detect_bottlenecks(flame_data, kernel_data, hw_caps)
    assert result["bottlenecks"], "Expected bottlenecks from synthetic data"

    score = profile_insights.calculate_optimization_score(hw_caps, result, kernel_data)
    assert 0 <= score["score"] <= 100
    assert score["quick_wins"], "Feature-based quick wins should be suggested"


def test_profile_insights_ncu_comparison_and_recommendations(tmp_path: Path):
    baseline_csv = tmp_path / "demo_baseline_ncu.csv"
    optimized_csv = tmp_path / "demo_optimized_ncu.csv"
    baseline_csv.write_text("Metric Name,Metric Value\nsm__throughput,50\noccupancy,40\n")
    optimized_csv.write_text("Metric Name,Metric Value\nsm__throughput,70\noccupancy,45\n")

    comparison = profile_insights.compare_ncu_files(tmp_path)
    assert comparison is not None
    assert comparison.get("metrics"), "CSV-based NCU comparison should return metrics"

    recs = profile_insights.generate_recommendations_from_profiles(
        {
            "ncu_comparison": comparison,
            "nsys_comparison": {"metrics": [{"name": "dram_util", "delta": -20}]},
        }
    )
    assert recs, "Recommendations should be produced from comparison data"


def test_profile_insights_nsys_comparison_monkeypatch(tmp_path: Path, monkeypatch):
    dummy_module_name = "tools.profiling.extract_nsys_summary"
    import types
    dummy_mod = types.ModuleType(dummy_module_name)

    def harvest(path):  # noqa: ANN001
        return [{"metric": "sm__throughput", "value": "10"}, {"metric": "dram_util", "value": "5"}]

    dummy_mod.harvest = harvest  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, dummy_module_name, dummy_mod)

    (tmp_path / "a_baseline.nsys-rep").write_text("")
    (tmp_path / "a_optimized.nsys-rep").write_text("")

    comparison = profile_insights.compare_nsys_files(tmp_path)
    assert comparison is not None
    assert comparison.get("metrics"), "Mocked nsys comparison should yield metrics"
