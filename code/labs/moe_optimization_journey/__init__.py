"""MoE Optimization Journey: From Naive to Production-Speed.

This lab takes you on a journey from a deliberately slow MoE implementation
to production-quality performance by applying optimization techniques from
the AI Performance Engineering book.

Each level builds on the previous, demonstrating compound speedups.

Levels:
    0. Naive - Sequential experts, Python loops (baseline)
    1. Parallel - Batched expert execution (~8x)
    2. Compiled - torch.compile kernel fusion (~2x)
    3. FP8 - 8-bit quantization (~1.5x)
    4. Triton - Custom fused kernels (~1.3x)
    5. Expert Parallel - Multi-stream execution (~1.2x)
    6. Full Stack - All optimizations combined (50-100x total!)

Usage with bench CLI:
    # Run all levels
    python -m cli.aisp bench run --targets labs/moe_optimization_journey
    
    # Run specific level
    python -m cli.aisp bench run --targets labs/moe_optimization_journey/level0_naive
    
    # Compare levels
    python -m cli.aisp bench compare labs/moe_optimization_journey/level0_naive labs/moe_optimization_journey/level6_full_stack
"""

from labs.moe_optimization_journey.moe_config import MoEConfig, get_config, CONFIGS

__all__ = [
    "MoEConfig",
    "get_config", 
    "CONFIGS",
]
