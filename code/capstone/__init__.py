from .capstone_extension import load_capstone_module
from .scenario_benchmark import (
    CapstoneScenarioBenchmark,
    ScenarioVariant,
    list_available_scenarios,
)

try:
    from .capstone_extension_tcgen05 import load_tcgen05_module
except Exception:  # pragma: no cover - build optional
    load_tcgen05_module = None


def baseline_matmul_non_tcgen05(a, b):
    """Reference kernel (no tcgen05)."""
    return load_capstone_module().baseline_matmul(a, b)


def optimized_matmul_non_tcgen05(a, b):
    """Clustered TMA kernel without tcgen05."""
    return load_capstone_module().optimized_matmul(a, b)


def optimized_matmul_tcgen05(a, b):
    """Inline tcgen05 path (requires SM100)."""
    if load_tcgen05_module is None:
        raise RuntimeError("tcgen05 inline extension unavailable.")
    module = load_tcgen05_module()
    return module.optimized_matmul_tcgen05(a, b)


def optimized_matmul_tcgen05_cta2(a, b):
    """CTA-group::2 tcgen05 path (requires SM100 with cluster multicast)."""
    if load_tcgen05_module is None:
        raise RuntimeError("tcgen05 inline extension unavailable.")
    module = load_tcgen05_module()
    return module.optimized_matmul_tcgen05_cta2(a, b)


# Backwards compatibility
baseline_matmul = baseline_matmul_non_tcgen05
optimized_matmul = optimized_matmul_non_tcgen05


__all__ = [
    "baseline_matmul_non_tcgen05",
    "optimized_matmul_non_tcgen05",
    "optimized_matmul_tcgen05",
    "optimized_matmul_tcgen05_cta2",
    "baseline_matmul",
    "optimized_matmul",
    "load_capstone_module",
    "CapstoneScenarioBenchmark",
    "ScenarioVariant",
    "list_available_scenarios",
]
