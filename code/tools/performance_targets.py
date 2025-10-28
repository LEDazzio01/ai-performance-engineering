#!/usr/bin/env python3
"""
Performance targets and thresholds for all chapters.
Extracted from README.md Performance Targets Summary.
"""

from typing import Dict, Any

# Performance targets by chapter
# Format: {"min": minimum acceptable, "target": ideal goal, "unit": display unit}
TARGETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "overall": {
        "hbm3e_bandwidth_tbs": {"min": 3.5, "target": 7.8, "unit": "TB/s", "realistic_max": 4.0},
        "fp16_compute_tflops": {"min": 1000, "target": 2000, "unit": "TFLOPS", "realistic_max": 1300},
        "torch_compile_speedup_small": {"min": 1.1, "target": 1.2, "unit": "x"},
        "torch_compile_speedup_large": {"min": 1.0, "target": 1.3, "unit": "x"},
        "flex_attention_speedup": {"min": 1.5, "target": 2.0, "unit": "x"},
        "deepseek_l2_speedup": {"min": 1.1, "target": 1.3, "unit": "x"},
    },
    "ch1": {
        "description": "Performance Basics",
        "metrics": {}
    },
    "ch2": {
        "description": "Hardware Overview",
        "metrics": {
            "nvlink_bandwidth_gbs": {"min": 850, "target": 900, "unit": "GB/s"},
            "hbm3e_bandwidth_tbs": {"min": 6.5, "target": 7.8, "unit": "TB/s"},
        }
    },
    "ch3": {
        "description": "System Setup",
        "metrics": {}
    },
    "ch4": {
        "description": "Distributed Networking",
        "metrics": {
            "allreduce_bandwidth_gbs": {"min": 700, "target": 800, "unit": "GB/s"},
            "p2p_bandwidth_gbs": {"min": 800, "target": 900, "unit": "GB/s"},
            "small_message_latency_us": {"min": 0, "target": 2, "unit": "μs", "lower_is_better": True},
            "scaling_efficiency_percent": {"min": 85, "target": 95, "unit": "%"},
        }
    },
    "ch5": {
        "description": "Storage & I/O",
        "metrics": {}
    },
    "ch6": {
        "description": "CUDA Kernels",
        "metrics": {}
    },
    "ch7": {
        "description": "Memory Access Patterns",
        "metrics": {
            "hbm3e_bandwidth_percent": {"min": 85, "target": 90, "unit": "%"},
            "coalesced_efficiency_percent": {"min": 85, "target": 90, "unit": "%"},
        }
    },
    "ch8": {
        "description": "Occupancy & ILP",
        "metrics": {}
    },
    "ch9": {
        "description": "Kernel Fusion",
        "metrics": {}
    },
    "ch10": {
        "description": "Tensor Cores",
        "metrics": {
            "fp8_tflops": {"min": 1200, "target": 1500, "unit": "TFLOPS"},
            "fp16_tflops": {"min": 1000, "target": 1300, "unit": "TFLOPS"},
            "tensor_core_utilization_percent": {"min": 80, "target": 90, "unit": "%"},
        }
    },
    "ch11": {
        "description": "Streams & Concurrency",
        "metrics": {
            "stream_speedup": {"min": 1.5, "target": 2.0, "unit": "x"},
            "stream_overlap_percent": {"min": 60, "target": 80, "unit": "%"},
        }
    },
    "ch12": {
        "description": "CUDA Graphs",
        "metrics": {
            "graph_speedup": {"min": 1.15, "target": 1.3, "unit": "x"},
            "launch_overhead_reduction_percent": {"min": 20, "target": 40, "unit": "%"},
        }
    },
    "ch13": {
        "description": "PyTorch Profiling",
        "metrics": {
            "compiled_autograd_speedup": {"min": 1.1, "target": 1.3, "unit": "x"},
        }
    },
    "ch14": {
        "description": "Compiler & Triton",
        "metrics": {
            "torch_compile_speedup_large": {"min": 1.0, "target": 1.3, "unit": "x"},
            "triton_fp8_speedup": {"min": 1.5, "target": 2.0, "unit": "x"},
            "deepseek_l2_speedup": {"min": 1.1, "target": 1.3, "unit": "x"},
        }
    },
    "ch15": {
        "description": "Disaggregated Inference",
        "metrics": {
            "resource_utilization_improvement_percent": {"min": 15, "target": 30, "unit": "%"},
        }
    },
    "ch16": {
        "description": "Inference Optimization",
        "metrics": {
            "fp8_inference_speedup": {"min": 1.8, "target": 2.0, "unit": "x"},
            "kv_cache_efficiency_percent": {"min": 80, "target": 90, "unit": "%"},
        }
    },
    "ch17": {
        "description": "Dynamic Routing",
        "metrics": {
            "routing_overhead_ms": {"min": 0, "target": 1.0, "unit": "ms", "lower_is_better": True},
            "load_balance_variance": {"min": 0, "target": 0.1, "unit": "", "lower_is_better": True},
        }
    },
    "ch18": {
        "description": "Attention Mechanisms",
        "metrics": {
            "flex_attention_compiled_speedup": {"min": 1.5, "target": 3.0, "unit": "x"},
            "flex_attention_uncompiled_slowdown": {"min": 0.8, "target": 0.9, "unit": "x", "lower_is_better": True},
        }
    },
    "ch19": {
        "description": "Advanced Training",
        "metrics": {
            "fp8_training_speedup": {"min": 1.5, "target": 2.0, "unit": "x"},
            "memory_reduction_percent": {"min": 30, "target": 50, "unit": "%"},
        }
    },
    "ch20": {
        "description": "AI Kernel Generator",
        "metrics": {}
    },
}


def get_target(chapter: str, metric: str) -> Dict[str, Any]:
    """Get target definition for a specific chapter/metric."""
    chapter_lower = chapter.lower()
    if chapter_lower not in TARGETS:
        return {}
    
    chapter_data = TARGETS[chapter_lower]
    if "metrics" not in chapter_data:
        return {}
    
    return chapter_data["metrics"].get(metric, {})


def get_all_chapters() -> list:
    """Get list of all chapters with targets."""
    return [ch for ch in TARGETS.keys() if ch != "overall"]


def get_chapter_description(chapter: str) -> str:
    """Get chapter description."""
    chapter_lower = chapter.lower()
    if chapter_lower not in TARGETS:
        return ""
    if chapter_lower == "overall":
        return "Overall System Performance"
    return TARGETS[chapter_lower].get("description", "")


def get_chapter_metrics(chapter: str) -> Dict[str, Dict[str, Any]]:
    """Get all metrics for a chapter."""
    chapter_lower = chapter.lower()
    if chapter_lower not in TARGETS:
        return {}
    # Overall stores metrics directly, not in a "metrics" sub-dict
    if chapter_lower == "overall":
        return TARGETS[chapter_lower]
    return TARGETS[chapter_lower].get("metrics", {})


def compute_status(actual: float, target_def: Dict[str, Any]) -> str:
    """
    Compute status (PASS/WARN/FAIL) based on actual value and target.
    
    Args:
        actual: Measured value
        target_def: Target definition with 'min', 'target', 'lower_is_better' fields
    
    Returns:
        Status string: "PASS", "WARN", or "FAIL"
    """
    if not target_def or "target" not in target_def:
        return "UNKNOWN"
    
    target_value = target_def["target"]
    min_value = target_def.get("min", target_value * 0.85)
    lower_is_better = target_def.get("lower_is_better", False)
    
    if lower_is_better:
        # For metrics where lower is better (latency, overhead)
        if actual <= target_value:
            return "PASS"
        elif actual <= min_value or actual <= target_value * 1.15:
            return "WARN"
        else:
            return "FAIL"
    else:
        # For metrics where higher is better (throughput, speedup)
        if actual >= target_value or actual >= min_value:
            return "PASS"
        elif actual >= target_value * 0.85:
            return "WARN"
        else:
            return "FAIL"


def format_value(value: float, unit: str) -> str:
    """Format a value with its unit."""
    if unit == "x":
        return f"{value:.2f}x"
    elif unit == "%":
        return f"{value:.1f}%"
    elif "FLOPS" in unit:
        return f"{value:.0f} {unit}"
    elif unit in ["GB/s", "TB/s"]:
        return f"{value:.2f} {unit}"
    elif unit in ["ms", "μs"]:
        return f"{value:.2f} {unit}"
    else:
        return f"{value:.2f} {unit}" if unit else f"{value:.2f}"

