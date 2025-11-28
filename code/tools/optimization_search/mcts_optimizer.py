#!/usr/bin/env python3
"""
🎯 Monte Carlo Tree Search (MCTS) Optimization Engine

RL-based search for discovering optimal compound optimization strategies.
Uses MCTS with UCB exploration to find the best combination of:
- Parallelism configurations (TP, PP, DP, CP, EP)
- Precision modes (FP32, BF16, FP8)
- Checkpointing strategies
- Communication optimizations
- Kernel fusion patterns
- Memory-efficient optimizers

The engine:
1. Models the optimization space as a tree
2. Uses UCB1 to balance exploration vs exploitation
3. Learns value functions from actual benchmark results
4. Can incorporate LLM-guided priors for novel combinations
5. Persists learned knowledge for future sessions

Usage:
    from tools.optimization_search import MCTSOptimizer
    
    optimizer = MCTSOptimizer(hardware_config, model_config)
    result = optimizer.search(
        budget=100,  # Number of rollouts
        optimization_goal="throughput"
    )
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import copy


# =============================================================================
# OPTIMIZATION ACTIONS (Moves in the search space)
# =============================================================================

class OptimizationDomain(Enum):
    """Domains of optimization."""
    PARALLELISM = "parallelism"
    PRECISION = "precision"
    CHECKPOINTING = "checkpointing"
    OPTIMIZER = "optimizer"
    COMMUNICATION = "communication"
    KERNELS = "kernels"
    MEMORY = "memory"
    SCHEDULING = "scheduling"


@dataclass
class OptimizationAction:
    """A single optimization action that can be applied."""
    name: str
    domain: OptimizationDomain
    params: Dict[str, Any]
    prerequisites: List[str] = field(default_factory=list)  # Required other actions
    conflicts: List[str] = field(default_factory=list)  # Incompatible actions
    estimated_memory_delta_gb: float = 0.0  # Memory impact
    estimated_throughput_delta_pct: float = 0.0  # Throughput impact
    hardware_requirements: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return self.name == other.name


# =============================================================================
# OPTIMIZATION STATE (Node in the search tree)
# =============================================================================

@dataclass
class OptimizationState:
    """
    State in the optimization search tree.
    Represents a configuration with applied optimizations.
    """
    applied_actions: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Estimated metrics (before actual evaluation)
    estimated_memory_gb: float = 0.0
    estimated_throughput_tps: float = 0.0
    
    # Actual metrics (after evaluation)
    actual_memory_gb: Optional[float] = None
    actual_throughput_tps: Optional[float] = None
    actual_speedup: Optional[float] = None
    
    # Validity
    is_valid: bool = True
    validity_reason: str = ""
    
    def get_hash(self) -> str:
        """Get unique hash for this state."""
        return hashlib.md5(json.dumps(sorted(self.applied_actions)).encode()).hexdigest()[:12]
    
    def clone(self) -> "OptimizationState":
        """Create a deep copy."""
        return OptimizationState(
            applied_actions=list(self.applied_actions),
            config=copy.deepcopy(self.config),
            estimated_memory_gb=self.estimated_memory_gb,
            estimated_throughput_tps=self.estimated_throughput_tps,
        )


# =============================================================================
# MCTS NODE
# =============================================================================

@dataclass
class MCTSNode:
    """Node in the MCTS search tree."""
    state: OptimizationState
    parent: Optional["MCTSNode"] = None
    children: Dict[str, "MCTSNode"] = field(default_factory=dict)  # action_name -> child
    
    # Statistics
    visits: int = 0
    total_value: float = 0.0
    
    # Unexplored actions from this node
    untried_actions: List[str] = field(default_factory=list)
    
    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    @property
    def is_terminal(self) -> bool:
        return len(self.untried_actions) == 0 and len(self.children) == 0
    
    @property
    def avg_value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0
    
    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCB1 score for selection."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.avg_value
        if self.parent is None:
            return exploitation
        
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration


# =============================================================================
# ACTION LIBRARY (All possible optimization actions)
# =============================================================================

class ActionLibrary:
    """
    Library of all available optimization actions.
    Dynamically generates actions based on hardware capabilities.
    """
    
    def __init__(
        self,
        hardware_config: Dict[str, Any],
        model_config: Dict[str, Any]
    ):
        self.hardware = hardware_config
        self.model = model_config
        self.actions: Dict[str, OptimizationAction] = {}
        self._build_action_library()
    
    def _build_action_library(self):
        """Build the complete action library."""
        self._add_parallelism_actions()
        self._add_precision_actions()
        self._add_checkpointing_actions()
        self._add_optimizer_actions()
        self._add_communication_actions()
        self._add_kernel_actions()
        self._add_memory_actions()
        self._add_scheduling_actions()
    
    def _add_parallelism_actions(self):
        """Add parallelism configuration actions."""
        num_gpus = self.hardware.get("num_gpus", 8)
        has_nvlink = self.hardware.get("has_nvlink", True)
        
        # Tensor Parallelism
        for tp in [1, 2, 4, 8]:
            if tp <= num_gpus:
                throughput_delta = -5 * (tp - 1) if has_nvlink else -15 * (tp - 1)
                self.actions[f"tp_{tp}"] = OptimizationAction(
                    name=f"tp_{tp}",
                    domain=OptimizationDomain.PARALLELISM,
                    params={"tensor_parallel": tp},
                    estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 2 * (1 - 1/tp),
                    estimated_throughput_delta_pct=throughput_delta,
                    conflicts=[f"tp_{x}" for x in [1, 2, 4, 8] if x != tp],
                )
        
        # Pipeline Parallelism
        for pp in [1, 2, 4, 8]:
            if pp <= num_gpus:
                bubble_overhead = (pp - 1) / (pp * 4) * 100
                self.actions[f"pp_{pp}"] = OptimizationAction(
                    name=f"pp_{pp}",
                    domain=OptimizationDomain.PARALLELISM,
                    params={"pipeline_parallel": pp},
                    estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 2 * (1 - 1/pp),
                    estimated_throughput_delta_pct=-bubble_overhead if pp > 1 else 0,
                    conflicts=[f"pp_{x}" for x in [1, 2, 4, 8] if x != pp],
                )
        
        # Data Parallelism
        for dp in [1, 2, 4, 8, 16, 32, 64]:
            if dp <= num_gpus:
                self.actions[f"dp_{dp}"] = OptimizationAction(
                    name=f"dp_{dp}",
                    domain=OptimizationDomain.PARALLELISM,
                    params={"data_parallel": dp},
                    estimated_memory_delta_gb=0,  # DP doesn't reduce model memory
                    estimated_throughput_delta_pct=(dp - 1) * 90 / dp,  # Near-linear scaling
                    conflicts=[f"dp_{x}" for x in [1, 2, 4, 8, 16, 32, 64] if x != dp],
                )
    
    def _add_precision_actions(self):
        """Add precision mode actions."""
        gpu_arch = self.hardware.get("gpu_arch", "ampere").lower()
        supports_fp8 = gpu_arch in ["hopper", "blackwell", "h100", "h200", "b100", "b200", "gb200"]
        supports_bf16 = gpu_arch not in ["volta", "turing", "pascal"]
        
        self.actions["precision_fp32"] = OptimizationAction(
            name="precision_fp32",
            domain=OptimizationDomain.PRECISION,
            params={"precision": "fp32"},
            estimated_memory_delta_gb=0,
            estimated_throughput_delta_pct=0,
            conflicts=["precision_bf16", "precision_fp16", "precision_fp8"],
        )
        
        if supports_bf16:
            self.actions["precision_bf16"] = OptimizationAction(
                name="precision_bf16",
                domain=OptimizationDomain.PRECISION,
                params={"precision": "bf16"},
                estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 2,
                estimated_throughput_delta_pct=40,
                conflicts=["precision_fp32", "precision_fp16", "precision_fp8"],
            )
        
        self.actions["precision_fp16"] = OptimizationAction(
            name="precision_fp16",
            domain=OptimizationDomain.PRECISION,
            params={"precision": "fp16"},
            estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 2,
            estimated_throughput_delta_pct=35,
            conflicts=["precision_fp32", "precision_bf16", "precision_fp8"],
        )
        
        if supports_fp8:
            self.actions["precision_fp8"] = OptimizationAction(
                name="precision_fp8",
                domain=OptimizationDomain.PRECISION,
                params={"precision": "fp8"},
                estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 3,
                estimated_throughput_delta_pct=80,
                conflicts=["precision_fp32", "precision_bf16", "precision_fp16"],
                hardware_requirements=["Hopper/Blackwell GPU"],
            )
    
    def _add_checkpointing_actions(self):
        """Add activation checkpointing actions."""
        self.actions["checkpoint_none"] = OptimizationAction(
            name="checkpoint_none",
            domain=OptimizationDomain.CHECKPOINTING,
            params={"gradient_checkpointing": False},
            estimated_memory_delta_gb=0,
            estimated_throughput_delta_pct=0,
            conflicts=["checkpoint_full", "checkpoint_selective", "checkpoint_block"],
        )
        
        self.actions["checkpoint_full"] = OptimizationAction(
            name="checkpoint_full",
            domain=OptimizationDomain.CHECKPOINTING,
            params={"gradient_checkpointing": True, "policy": "full"},
            estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 4,  # Significant savings
            estimated_throughput_delta_pct=-33,  # Recompute overhead
            conflicts=["checkpoint_none", "checkpoint_selective", "checkpoint_block"],
        )
        
        self.actions["checkpoint_selective"] = OptimizationAction(
            name="checkpoint_selective",
            domain=OptimizationDomain.CHECKPOINTING,
            params={"gradient_checkpointing": True, "policy": "selective"},
            estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 2,
            estimated_throughput_delta_pct=-15,
            conflicts=["checkpoint_none", "checkpoint_full", "checkpoint_block"],
        )
    
    def _add_optimizer_actions(self):
        """Add optimizer actions."""
        self.actions["optimizer_adamw"] = OptimizationAction(
            name="optimizer_adamw",
            domain=OptimizationDomain.OPTIMIZER,
            params={"optimizer": "adamw"},
            estimated_memory_delta_gb=0,  # Baseline
            estimated_throughput_delta_pct=0,
            conflicts=["optimizer_adamw_8bit", "optimizer_adafactor", "optimizer_lion"],
        )
        
        self.actions["optimizer_adamw_8bit"] = OptimizationAction(
            name="optimizer_adamw_8bit",
            domain=OptimizationDomain.OPTIMIZER,
            params={"optimizer": "adamw_8bit"},
            estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 6,
            estimated_throughput_delta_pct=-2,
            conflicts=["optimizer_adamw", "optimizer_adafactor", "optimizer_lion"],
        )
        
        self.actions["optimizer_adafactor"] = OptimizationAction(
            name="optimizer_adafactor",
            domain=OptimizationDomain.OPTIMIZER,
            params={"optimizer": "adafactor"},
            estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 7.5,
            estimated_throughput_delta_pct=-5,
            conflicts=["optimizer_adamw", "optimizer_adamw_8bit", "optimizer_lion"],
        )
    
    def _add_communication_actions(self):
        """Add communication optimization actions."""
        self.actions["comm_overlap"] = OptimizationAction(
            name="comm_overlap",
            domain=OptimizationDomain.COMMUNICATION,
            params={"overlap_communication": True},
            estimated_memory_delta_gb=0,
            estimated_throughput_delta_pct=10,
            prerequisites=["dp_2", "dp_4", "dp_8", "dp_16", "dp_32", "dp_64"],
        )
        
        self.actions["gradient_compression"] = OptimizationAction(
            name="gradient_compression",
            domain=OptimizationDomain.COMMUNICATION,
            params={"gradient_compression": "powersgd"},
            estimated_memory_delta_gb=0,
            estimated_throughput_delta_pct=5,
            prerequisites=["dp_2", "dp_4", "dp_8", "dp_16", "dp_32", "dp_64"],
        )
    
    def _add_kernel_actions(self):
        """Add kernel fusion actions."""
        gpu_arch = self.hardware.get("gpu_arch", "ampere").lower()
        supports_flash_attn = gpu_arch in ["ampere", "hopper", "blackwell", "a100", "h100", "h200", "b100", "b200", "gb200"]
        
        if supports_flash_attn:
            self.actions["flash_attention"] = OptimizationAction(
                name="flash_attention",
                domain=OptimizationDomain.KERNELS,
                params={"flash_attention": True},
                estimated_memory_delta_gb=-2,
                estimated_throughput_delta_pct=20,
            )
        
        self.actions["torch_compile"] = OptimizationAction(
            name="torch_compile",
            domain=OptimizationDomain.KERNELS,
            params={"torch_compile": True, "mode": "reduce-overhead"},
            estimated_memory_delta_gb=0,
            estimated_throughput_delta_pct=15,
        )
        
        self.actions["fused_kernels"] = OptimizationAction(
            name="fused_kernels",
            domain=OptimizationDomain.KERNELS,
            params={"fused_layer_norm": True, "fused_bias_gelu": True},
            estimated_memory_delta_gb=0,
            estimated_throughput_delta_pct=10,
        )
    
    def _add_memory_actions(self):
        """Add memory optimization actions."""
        self.actions["cpu_offload_optimizer"] = OptimizationAction(
            name="cpu_offload_optimizer",
            domain=OptimizationDomain.MEMORY,
            params={"offload_optimizer": True},
            estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 8,
            estimated_throughput_delta_pct=-20,
        )
        
        self.actions["cpu_offload_params"] = OptimizationAction(
            name="cpu_offload_params",
            domain=OptimizationDomain.MEMORY,
            params={"offload_params": True},
            estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 2,
            estimated_throughput_delta_pct=-40,
        )
    
    def _add_scheduling_actions(self):
        """Add pipeline scheduling actions."""
        self.actions["schedule_1f1b"] = OptimizationAction(
            name="schedule_1f1b",
            domain=OptimizationDomain.SCHEDULING,
            params={"pipeline_schedule": "1f1b"},
            estimated_throughput_delta_pct=0,
            prerequisites=["pp_2", "pp_4", "pp_8"],
            conflicts=["schedule_interleaved", "schedule_zero_bubble"],
        )
        
        self.actions["schedule_interleaved"] = OptimizationAction(
            name="schedule_interleaved",
            domain=OptimizationDomain.SCHEDULING,
            params={"pipeline_schedule": "interleaved", "virtual_stages": 2},
            estimated_throughput_delta_pct=10,
            prerequisites=["pp_2", "pp_4", "pp_8"],
            conflicts=["schedule_1f1b", "schedule_zero_bubble"],
        )
        
        self.actions["schedule_zero_bubble"] = OptimizationAction(
            name="schedule_zero_bubble",
            domain=OptimizationDomain.SCHEDULING,
            params={"pipeline_schedule": "zero_bubble"},
            estimated_memory_delta_gb=self.model.get("parameters_billions", 7) * 0.5,
            estimated_throughput_delta_pct=20,
            prerequisites=["pp_4", "pp_8"],
            conflicts=["schedule_1f1b", "schedule_interleaved"],
        )
    
    def get_valid_actions(self, state: OptimizationState) -> List[str]:
        """Get all valid actions from current state."""
        valid = []
        
        for name, action in self.actions.items():
            if name in state.applied_actions:
                continue
            
            # Check conflicts
            has_conflict = any(c in state.applied_actions for c in action.conflicts)
            if has_conflict:
                continue
            
            # Check prerequisites (at least one must be satisfied)
            if action.prerequisites:
                has_prereq = any(p in state.applied_actions for p in action.prerequisites)
                if not has_prereq:
                    continue
            
            # Check hardware requirements
            hardware_ok = True
            for req in action.hardware_requirements:
                if "hopper" in req.lower() or "blackwell" in req.lower():
                    gpu_arch = self.hardware.get("gpu_arch", "").lower()
                    if gpu_arch not in ["hopper", "blackwell", "h100", "h200", "b100", "b200", "gb200"]:
                        hardware_ok = False
                        break
            
            if hardware_ok:
                valid.append(name)
        
        return valid


# =============================================================================
# MCTS OPTIMIZER
# =============================================================================

class MCTSOptimizer:
    """
    Monte Carlo Tree Search optimizer for compound optimization discovery.
    
    Uses UCB1 for exploration/exploitation balance and learns from actual
    benchmark results to guide future searches.
    """
    
    def __init__(
        self,
        hardware_config: Dict[str, Any],
        model_config: Dict[str, Any],
        evaluator: Optional[Callable[[OptimizationState], float]] = None,
        exploration_constant: float = 1.414,
        knowledge_base_path: Optional[Path] = None,
    ):
        self.hardware = hardware_config
        self.model = model_config
        self.action_library = ActionLibrary(hardware_config, model_config)
        self.exploration_constant = exploration_constant
        
        # Evaluator function (can be actual benchmark or simulator)
        self.evaluator = evaluator or self._default_evaluator
        
        # Knowledge base for persistent learning
        self.knowledge_base_path = knowledge_base_path or Path.home() / ".cache" / "mcts_optimizer" / "knowledge.json"
        self.knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
        self.knowledge_base = self._load_knowledge_base()
        
        # Statistics
        self.total_evaluations = 0
        self.cache_hits = 0
    
    def search(
        self,
        budget: int = 100,
        optimization_goal: str = "throughput",  # "throughput", "memory", "balanced"
        max_depth: int = 10,
        early_stop_threshold: float = 0.95,  # Stop if we find something this good
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run MCTS search to find optimal compound optimization.
        
        Args:
            budget: Number of rollouts/simulations
            optimization_goal: What to optimize for
            max_depth: Maximum depth of search
            early_stop_threshold: Normalized score to trigger early stop
            verbose: Print progress
        
        Returns:
            Best configuration found with statistics
        """
        start_time = time.time()
        
        # Initialize root
        initial_state = self._create_initial_state()
        root = MCTSNode(
            state=initial_state,
            untried_actions=self.action_library.get_valid_actions(initial_state)
        )
        
        best_score = float('-inf')
        best_node = root
        scores_history = []
        
        for i in range(budget):
            # 1. Selection
            node = self._select(root)
            
            # 2. Expansion
            if not node.is_terminal and node.untried_actions:
                node = self._expand(node)
            
            # 3. Simulation/Evaluation
            score = self._evaluate(node.state, optimization_goal)
            scores_history.append(score)
            
            # 4. Backpropagation
            self._backpropagate(node, score)
            
            # Track best
            if score > best_score:
                best_score = score
                best_node = node
                if verbose:
                    print(f"  Iteration {i+1}/{budget}: New best score {score:.4f}")
                    print(f"    Actions: {node.state.applied_actions}")
            
            # Early stopping
            if score >= early_stop_threshold:
                if verbose:
                    print(f"  Early stop at iteration {i+1} - found excellent configuration")
                break
        
        # Save learned knowledge
        self._save_knowledge_base()
        
        search_time = time.time() - start_time
        
        return {
            "best_config": best_node.state.config,
            "best_actions": best_node.state.applied_actions,
            "best_score": best_score,
            "estimated_speedup": 1.0 + best_node.state.estimated_throughput_tps / 100,
            "estimated_memory_reduction_gb": abs(best_node.state.estimated_memory_gb),
            "search_statistics": {
                "total_iterations": len(scores_history),
                "total_evaluations": self.total_evaluations,
                "cache_hits": self.cache_hits,
                "search_time_seconds": search_time,
                "scores_history": scores_history[-20:],  # Last 20
            },
            "tree_statistics": {
                "root_visits": root.visits,
                "num_children": len(root.children),
                "avg_depth": self._get_avg_depth(root),
            },
            "recommendations": self._generate_recommendations(best_node),
        }
    
    def _create_initial_state(self) -> OptimizationState:
        """Create the initial state (no optimizations applied)."""
        model_params_b = self.model.get("parameters_billions", 7)
        return OptimizationState(
            applied_actions=[],
            config={},
            estimated_memory_gb=model_params_b * 20,  # Rough estimate: 20x params
            estimated_throughput_tps=1000,  # Baseline throughput
        )
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1 until we reach unexpanded node."""
        while node.children and node.is_fully_expanded:
            # Select child with highest UCB1 score
            node = max(
                node.children.values(),
                key=lambda n: n.ucb1_score(self.exploration_constant)
            )
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by trying an untried action."""
        if not node.untried_actions:
            return node
        
        # Pick action (can use LLM prior here for smarter selection)
        action_name = self._select_action_to_try(node)
        node.untried_actions.remove(action_name)
        
        # Create child state
        action = self.action_library.actions[action_name]
        child_state = node.state.clone()
        child_state.applied_actions.append(action_name)
        child_state.config.update(action.params)
        child_state.estimated_memory_gb += action.estimated_memory_delta_gb
        child_state.estimated_throughput_tps += action.estimated_throughput_delta_pct
        
        # Create child node
        valid_actions = self.action_library.get_valid_actions(child_state)
        child = MCTSNode(
            state=child_state,
            parent=node,
            untried_actions=valid_actions
        )
        
        node.children[action_name] = child
        return child
    
    def _select_action_to_try(self, node: MCTSNode) -> str:
        """
        Select which untried action to expand.
        Can incorporate LLM prior or learned heuristics here.
        """
        # Check knowledge base for prior scores
        priors = []
        for action_name in node.untried_actions:
            prior = self.knowledge_base.get("action_priors", {}).get(action_name, 0.5)
            priors.append((action_name, prior))
        
        # Use weighted random selection based on priors
        if priors:
            total = sum(p for _, p in priors)
            r = random.uniform(0, total)
            cumulative = 0
            for action_name, prior in priors:
                cumulative += prior
                if cumulative >= r:
                    return action_name
        
        # Fallback: random
        return random.choice(node.untried_actions)
    
    def _evaluate(
        self,
        state: OptimizationState,
        optimization_goal: str
    ) -> float:
        """Evaluate a state and return normalized score."""
        state_hash = state.get_hash()
        
        # Check cache
        if state_hash in self.knowledge_base.get("evaluations", {}):
            self.cache_hits += 1
            return self.knowledge_base["evaluations"][state_hash]
        
        self.total_evaluations += 1
        
        # Use evaluator (actual benchmark or simulator)
        raw_score = self.evaluator(state)
        
        # Normalize based on goal
        if optimization_goal == "throughput":
            score = (state.estimated_throughput_tps + 100) / 200  # Normalize to 0-1 ish
        elif optimization_goal == "memory":
            score = (100 - state.estimated_memory_gb) / 100
        else:  # balanced
            throughput_score = (state.estimated_throughput_tps + 100) / 200
            memory_score = (100 - state.estimated_memory_gb) / 100
            score = 0.5 * throughput_score + 0.5 * memory_score
        
        # Combine with evaluator score
        score = 0.7 * raw_score + 0.3 * score
        
        # Cache
        if "evaluations" not in self.knowledge_base:
            self.knowledge_base["evaluations"] = {}
        self.knowledge_base["evaluations"][state_hash] = score
        
        # Update action priors based on results
        self._update_action_priors(state, score)
        
        return score
    
    def _update_action_priors(self, state: OptimizationState, score: float):
        """Update action priors based on evaluation results."""
        if "action_priors" not in self.knowledge_base:
            self.knowledge_base["action_priors"] = {}
        
        for action in state.applied_actions:
            current = self.knowledge_base["action_priors"].get(action, 0.5)
            # Exponential moving average
            self.knowledge_base["action_priors"][action] = 0.9 * current + 0.1 * score
    
    def _backpropagate(self, node: MCTSNode, score: float):
        """Backpropagate score up the tree."""
        while node is not None:
            node.visits += 1
            node.total_value += score
            node = node.parent
    
    def _default_evaluator(self, state: OptimizationState) -> float:
        """Default evaluator using estimated metrics."""
        # Check validity
        if state.estimated_memory_gb > self.hardware.get("gpu_memory_gb", 80):
            return 0.0  # Invalid - doesn't fit
        
        # Score based on throughput improvement
        throughput_score = min(1.0, state.estimated_throughput_tps / 200)
        
        # Bonus for memory efficiency
        memory_score = 1.0 - (state.estimated_memory_gb / 100)
        
        return 0.7 * throughput_score + 0.3 * memory_score
    
    def _get_avg_depth(self, node: MCTSNode, depth: int = 0) -> float:
        """Calculate average depth of the tree."""
        if not node.children:
            return depth
        
        child_depths = [self._get_avg_depth(c, depth + 1) for c in node.children.values()]
        return sum(child_depths) / len(child_depths)
    
    def _generate_recommendations(self, node: MCTSNode) -> List[str]:
        """Generate human-readable recommendations from best configuration."""
        recommendations = []
        
        for action_name in node.state.applied_actions:
            action = self.action_library.actions.get(action_name)
            if action:
                domain = action.domain.value
                params = action.params
                recommendations.append(
                    f"[{domain.upper()}] Apply {action_name}: {params}"
                )
        
        return recommendations
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load persistent knowledge base."""
        if self.knowledge_base_path.exists():
            try:
                with open(self.knowledge_base_path) as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_knowledge_base(self):
        """Save knowledge base to disk."""
        try:
            with open(self.knowledge_base_path, "w") as f:
                json.dump(self.knowledge_base, f, indent=2)
        except:
            pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def search_optimal_config(
    model_config: Dict[str, Any],
    hardware_config: Dict[str, Any],
    optimization_goal: str = "throughput",
    budget: int = 100,
) -> Dict[str, Any]:
    """
    Convenience function to search for optimal configuration.
    
    Args:
        model_config: Model configuration
        hardware_config: Hardware configuration
        optimization_goal: "throughput", "memory", or "balanced"
        budget: Search budget (number of rollouts)
    
    Returns:
        Best configuration found
    """
    optimizer = MCTSOptimizer(hardware_config, model_config)
    return optimizer.search(
        budget=budget,
        optimization_goal=optimization_goal,
        verbose=False,
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCTS Optimization Search")
    parser.add_argument("--model-size", type=float, default=70, help="Model size in billions")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--gpu-memory", type=float, default=80, help="GPU memory in GB")
    parser.add_argument("--gpu-arch", default="hopper", help="GPU architecture")
    parser.add_argument("--goal", default="throughput", choices=["throughput", "memory", "balanced"])
    parser.add_argument("--budget", type=int, default=100, help="Search budget")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    model_config = {
        "parameters_billions": args.model_size,
        "num_layers": int(args.model_size * 1.2),  # Rough estimate
        "hidden_size": int((args.model_size * 1e9 / 100) ** 0.5 * 128),
    }
    
    hardware_config = {
        "num_gpus": args.num_gpus,
        "gpu_memory_gb": args.gpu_memory,
        "gpu_arch": args.gpu_arch,
        "has_nvlink": True,
    }
    
    print(f"\n🎯 MCTS Optimization Search")
    print(f"   Model: {args.model_size}B parameters")
    print(f"   Hardware: {args.num_gpus}x {args.gpu_arch} ({args.gpu_memory}GB each)")
    print(f"   Goal: {args.goal}")
    print(f"   Budget: {args.budget} rollouts")
    print()
    
    optimizer = MCTSOptimizer(hardware_config, model_config)
    result = optimizer.search(
        budget=args.budget,
        optimization_goal=args.goal,
        verbose=args.verbose,
    )
    
    print("\n" + "=" * 60)
    print("🏆 BEST CONFIGURATION FOUND")
    print("=" * 60)
    print(f"\nScore: {result['best_score']:.4f}")
    print(f"Estimated Speedup: {result['estimated_speedup']:.2f}x")
    print(f"Memory Reduction: {result['estimated_memory_reduction_gb']:.1f} GB")
    
    print("\n📋 Applied Optimizations:")
    for action in result['best_actions']:
        print(f"   • {action}")
    
    print("\n⚙️  Configuration:")
    print(json.dumps(result['best_config'], indent=4))
    
    print("\n📊 Search Statistics:")
    stats = result['search_statistics']
    print(f"   Iterations: {stats['total_iterations']}")
    print(f"   Evaluations: {stats['total_evaluations']}")
    print(f"   Cache Hits: {stats['cache_hits']}")
    print(f"   Time: {stats['search_time_seconds']:.2f}s")
    
    print("\n💡 Recommendations:")
    for rec in result['recommendations']:
        print(f"   {rec}")


if __name__ == "__main__":
    main()


