"""
🎯 Optimization Search Module

RL-based and LLM-guided optimization discovery for compound performance techniques.

Components:
- MCTSOptimizer: Monte Carlo Tree Search for compound optimization discovery
- LLMOracle: LLM-guided optimization suggestions with learned context
- UnifiedOptimizer: Combines MCTS + LLM + heuristics

Usage:
    from tools.optimization_search import search_optimal_config
    
    result = search_optimal_config(
        model_config={"parameters_billions": 70, ...},
        hardware_config={"num_gpus": 8, "gpu_arch": "hopper", ...},
        optimization_goal="throughput",
        budget=100
    )
"""

from .mcts_optimizer import (
    MCTSOptimizer,
    OptimizationAction,
    OptimizationState,
    OptimizationDomain,
    ActionLibrary,
    search_optimal_config,
)

from .llm_oracle import (
    LLMOracle,
    OptimizationSuggestion,
    OracleKnowledgeBase,
    ContextCollector,
    get_suggestions,
    ask_oracle,
)

__all__ = [
    # MCTS
    "MCTSOptimizer",
    "OptimizationAction",
    "OptimizationState",
    "OptimizationDomain",
    "ActionLibrary",
    "search_optimal_config",
    # LLM Oracle
    "LLMOracle",
    "OptimizationSuggestion",
    "OracleKnowledgeBase",
    "ContextCollector",
    "get_suggestions",
    "ask_oracle",
]

