#!/usr/bin/env python3
"""
Model Architecture Analyzer

Analyzes model architectures to extract key parameters needed for
parallelism planning. Supports:
- HuggingFace model IDs (auto-fetch config)
- Local model configs
- Custom architecture specifications
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error


class ModelType(Enum):
    """Type of model architecture."""
    DENSE = "dense"
    MOE = "moe"  # Mixture of Experts
    HYBRID_MOE = "hybrid_moe"  # Mixed dense + MoE layers


class AttentionType(Enum):
    """Type of attention mechanism."""
    MHA = "multi_head"
    MQA = "multi_query"
    GQA = "grouped_query"


@dataclass
class ModelArchitecture:
    """Detailed model architecture specification."""
    
    # Basic info
    name: str
    model_type: ModelType
    
    # Core dimensions
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    vocab_size: int
    
    # Derived/optional dimensions
    intermediate_size: int = 0  # FFN hidden size
    num_key_value_heads: int = 0  # For GQA/MQA
    head_dim: int = 0
    
    # MoE parameters (if applicable)
    num_experts: int = 1
    num_experts_per_token: int = 1
    moe_layer_freq: int = 0  # Every N layers is MoE (0 = all dense)
    num_moe_layers: int = 0
    
    # Sequence parameters
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    
    # Precision
    dtype_bytes: int = 2  # FP16/BF16 default
    
    # Attention type
    attention_type: AttentionType = AttentionType.MHA
    
    # Computed values
    total_params_billion: float = 0.0
    active_params_billion: float = 0.0  # For MoE
    
    def __post_init__(self):
        # Fill in defaults
        if self.intermediate_size == 0:
            self.intermediate_size = self.hidden_size * 4
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Determine attention type
        if self.num_key_value_heads == 1:
            self.attention_type = AttentionType.MQA
        elif self.num_key_value_heads < self.num_attention_heads:
            self.attention_type = AttentionType.GQA
        
        # Count MoE layers
        if self.moe_layer_freq > 0:
            self.num_moe_layers = self.num_layers // self.moe_layer_freq
        elif self.num_experts > 1:
            self.num_moe_layers = self.num_layers
        
        # Set model type
        if self.num_experts > 1:
            if self.num_moe_layers < self.num_layers:
                self.model_type = ModelType.HYBRID_MOE
            else:
                self.model_type = ModelType.MOE
        
        # Calculate parameters
        self._calculate_params()
    
    def _calculate_params(self):
        """Calculate total and active parameters."""
        # Embedding: vocab_size * hidden_size (input embedding)
        # Output head often tied to input embedding, count once
        embedding_params = self.vocab_size * self.hidden_size
        
        # Attention per layer:
        # Q: hidden_size * hidden_size
        # K: hidden_size * (num_kv_heads * head_dim)
        # V: hidden_size * (num_kv_heads * head_dim)
        # O: hidden_size * hidden_size
        kv_size = self.num_key_value_heads * self.head_dim
        attention_params = (
            self.hidden_size * self.hidden_size +  # Q
            self.hidden_size * kv_size +            # K
            self.hidden_size * kv_size +            # V
            self.hidden_size * self.hidden_size     # O
        )
        
        # FFN per layer (SwiGLU style): 
        # Gate: hidden_size * intermediate_size
        # Up: hidden_size * intermediate_size
        # Down: intermediate_size * hidden_size
        ffn_params = 3 * self.hidden_size * self.intermediate_size
        
        # Count layers
        num_dense_layers = self.num_layers - self.num_moe_layers
        
        # Dense layer: attention + FFN
        dense_layer_params = attention_params + ffn_params
        
        # MoE layer: attention + (FFN * num_experts) + router
        router_params = self.hidden_size * self.num_experts  # Router is small
        moe_layer_params = attention_params + (ffn_params * self.num_experts) + router_params
        
        # Total params (embedding counted once, assuming tied)
        total = (
            embedding_params +
            (dense_layer_params * num_dense_layers) +
            (moe_layer_params * self.num_moe_layers)
        )
        self.total_params_billion = total / 1e9
        
        # Active params (for MoE, only top-k experts are active per forward pass)
        active_moe_layer_params = attention_params + (ffn_params * self.num_experts_per_token) + router_params
        active = (
            embedding_params +
            (dense_layer_params * num_dense_layers) +
            (active_moe_layer_params * self.num_moe_layers)
        )
        self.active_params_billion = active / 1e9
    
    def estimate_memory_gb(
        self,
        batch_size: int = 1,
        seq_length: int = 2048,
        include_kv_cache: bool = True,
        include_activations: bool = True,
        include_optimizer: bool = False,
        precision_bytes: Optional[int] = None,
    ) -> Dict[str, float]:
        """Estimate memory requirements in GB.
        
        Args:
            batch_size: Batch size for inference/training
            seq_length: Sequence length
            include_kv_cache: Include KV cache memory
            include_activations: Include activation memory
            include_optimizer: Include optimizer states (training)
            precision_bytes: Override default precision
            
        Returns:
            Dictionary with memory breakdown
        """
        dtype = precision_bytes or self.dtype_bytes
        
        # Model weights
        weights_bytes = self.total_params_billion * 1e9 * dtype
        weights_gb = weights_bytes / (1024 ** 3)
        
        # KV cache: 2 * num_layers * batch_size * seq_length * num_kv_heads * head_dim * dtype
        kv_cache_gb = 0.0
        if include_kv_cache:
            kv_cache_bytes = (
                2 * self.num_layers * batch_size * seq_length *
                self.num_key_value_heads * self.head_dim * dtype
            )
            kv_cache_gb = kv_cache_bytes / (1024 ** 3)
        
        # Activations (rough estimate)
        activation_gb = 0.0
        if include_activations:
            # Per layer: hidden states + attention scores
            per_layer = (
                batch_size * seq_length * self.hidden_size * dtype +  # Hidden states
                batch_size * self.num_attention_heads * seq_length * seq_length * dtype / 2  # Attention
            )
            activation_bytes = per_layer * self.num_layers
            activation_gb = activation_bytes / (1024 ** 3)
        
        # Optimizer states (Adam: 2x for momentum + variance)
        optimizer_gb = 0.0
        if include_optimizer:
            optimizer_gb = weights_gb * 2  # FP32 optimizer states
        
        # Gradient memory (training)
        gradient_gb = weights_gb if include_optimizer else 0.0
        
        total_gb = weights_gb + kv_cache_gb + activation_gb + optimizer_gb + gradient_gb
        
        return {
            "weights_gb": weights_gb,
            "kv_cache_gb": kv_cache_gb,
            "activations_gb": activation_gb,
            "optimizer_gb": optimizer_gb,
            "gradients_gb": gradient_gb,
            "total_gb": total_gb,
        }
    
    def get_communication_volumes(
        self,
        batch_size: int,
        seq_length: int,
        tp_size: int = 1,
        pp_size: int = 1,
        dp_size: int = 1,
    ) -> Dict[str, float]:
        """Estimate communication volumes for different parallelism types.
        
        Returns volumes in GB per step.
        """
        dtype = self.dtype_bytes
        
        # Tensor Parallel: All-reduce on attention output and FFN output
        # Volume: 2 * batch * seq * hidden * dtype per layer
        tp_volume_gb = 0.0
        if tp_size > 1:
            volume_per_layer = 2 * batch_size * seq_length * self.hidden_size * dtype
            tp_volume_bytes = volume_per_layer * self.num_layers * 2  # Forward + backward
            tp_volume_gb = tp_volume_bytes / (1024 ** 3)
        
        # Pipeline Parallel: Point-to-point activations
        pp_volume_gb = 0.0
        if pp_size > 1:
            # One activation tensor per micro-batch boundary
            volume_per_boundary = batch_size * seq_length * self.hidden_size * dtype
            pp_volume_bytes = volume_per_boundary * (pp_size - 1) * 2
            pp_volume_gb = pp_volume_bytes / (1024 ** 3)
        
        # Data Parallel: All-reduce gradients
        dp_volume_gb = 0.0
        if dp_size > 1:
            # All gradients (simplified - actual depends on sharding)
            dp_volume_bytes = self.total_params_billion * 1e9 * dtype
            dp_volume_gb = dp_volume_bytes / (1024 ** 3)
        
        # Context Parallel: Ring all-reduce for KV
        cp_volume_gb = 0.0
        # Volume depends on CP implementation
        
        return {
            "tp_volume_gb": tp_volume_gb,
            "pp_volume_gb": pp_volume_gb,
            "dp_volume_gb": dp_volume_gb,
            "cp_volume_gb": cp_volume_gb,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "num_experts": self.num_experts,
            "num_experts_per_token": self.num_experts_per_token,
            "moe_layer_freq": self.moe_layer_freq,
            "num_moe_layers": self.num_moe_layers,
            "attention_type": self.attention_type.value,
            "total_params_billion": self.total_params_billion,
            "active_params_billion": self.active_params_billion,
            "dtype_bytes": self.dtype_bytes,
        }


# Well-known model presets
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "llama-7b": {
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "intermediate_size": 11008,
        "vocab_size": 32000,
    },
    "llama-13b": {
        "hidden_size": 5120,
        "num_layers": 40,
        "num_attention_heads": 40,
        "intermediate_size": 13824,
        "vocab_size": 32000,
    },
    "llama-70b": {
        "hidden_size": 8192,
        "num_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 28672,
        "vocab_size": 32000,
    },
    "llama-3.1-8b": {
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
    },
    "llama-3.1-70b": {
        "hidden_size": 8192,
        "num_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 28672,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
    },
    "llama-3.1-405b": {
        "hidden_size": 16384,
        "num_layers": 126,
        "num_attention_heads": 128,
        "num_key_value_heads": 8,
        "intermediate_size": 53248,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
    },
    "mixtral-8x7b": {
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "vocab_size": 32000,
        "num_experts": 8,
        "num_experts_per_token": 2,
    },
    "mixtral-8x22b": {
        "hidden_size": 6144,
        "num_layers": 56,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "intermediate_size": 16384,
        "vocab_size": 32000,
        "num_experts": 8,
        "num_experts_per_token": 2,
    },
    "deepseek-v2": {
        # DeepSeek-V2: 236B total, ~21B active
        # Uses smaller expert FFN size (1536) with many experts
        "hidden_size": 5120,
        "num_layers": 60,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,  # MLA (Multi-head Latent Attention)
        "intermediate_size": 1536,   # Per-expert FFN is small
        "vocab_size": 102400,
        "num_experts": 160,
        "num_experts_per_token": 6,
        "moe_layer_freq": 2,  # ~30 MoE layers
    },
    "deepseek-v3": {
        # DeepSeek-V3: 671B total, ~37B active
        "hidden_size": 7168,
        "num_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "intermediate_size": 2048,   # Per-expert FFN
        "vocab_size": 129280,
        "num_experts": 256,
        "num_experts_per_token": 8,
        "moe_layer_freq": 1,
    },
    "deepseek-r1": {
        # DeepSeek-R1: Based on V3 architecture (~671B)
        "hidden_size": 7168,
        "num_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "intermediate_size": 2048,
        "vocab_size": 129280,
        "num_experts": 256,
        "num_experts_per_token": 8,
        "moe_layer_freq": 1,
    },
    "gpt-4-estimated": {
        "hidden_size": 12288,
        "num_layers": 120,
        "num_attention_heads": 96,
        "num_key_value_heads": 96,
        "intermediate_size": 49152,
        "vocab_size": 100000,
        "num_experts": 16,
        "num_experts_per_token": 2,
    },
    "qwen-72b": {
        "hidden_size": 8192,
        "num_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "intermediate_size": 24576,
        "vocab_size": 152064,
    },
}


class ModelAnalyzer:
    """Analyzes model architectures for parallelism planning."""
    
    HF_API_URL = "https://huggingface.co/api/models"
    
    def __init__(self):
        self._cache: Dict[str, ModelArchitecture] = {}
    
    def analyze(
        self,
        model_id: str,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> ModelArchitecture:
        """Analyze a model and return its architecture.
        
        Args:
            model_id: HuggingFace model ID or preset name
            custom_config: Override config values
            
        Returns:
            ModelArchitecture with full specs
        """
        cache_key = f"{model_id}:{json.dumps(custom_config or {}, sort_keys=True)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try preset first
        preset_key = self._normalize_preset_name(model_id)
        if preset_key in MODEL_PRESETS:
            config = MODEL_PRESETS[preset_key].copy()
            if custom_config:
                config.update(custom_config)
            arch = self._create_architecture(model_id, config)
            self._cache[cache_key] = arch
            return arch
        
        # Try to fetch from HuggingFace
        try:
            config = self._fetch_hf_config(model_id)
            if custom_config:
                config.update(custom_config)
            arch = self._create_architecture(model_id, config)
            self._cache[cache_key] = arch
            return arch
        except Exception as e:
            raise ValueError(
                f"Could not analyze model '{model_id}': {e}. "
                f"Available presets: {list(MODEL_PRESETS.keys())}"
            )
    
    def _normalize_preset_name(self, name: str) -> str:
        """Normalize model name to match presets."""
        name = name.lower()
        # Remove common prefixes
        for prefix in ["meta-llama/", "mistralai/", "deepseek-ai/", "qwen/"]:
            if name.startswith(prefix):
                name = name[len(prefix):]
        # Common normalizations
        name = name.replace("meta-llama-", "llama-")
        name = name.replace("llama-3-", "llama-3.1-")
        name = name.replace("-instruct", "")
        name = name.replace("-chat", "")
        return name
    
    def _fetch_hf_config(self, model_id: str) -> Dict[str, Any]:
        """Fetch model config from HuggingFace."""
        # Try to get config.json
        config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        
        try:
            req = urllib.request.Request(
                config_url,
                headers={"User-Agent": "parallelism-planner/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                config = json.loads(response.read().decode())
        except urllib.error.HTTPError:
            # Try API endpoint
            api_url = f"{self.HF_API_URL}/{model_id}"
            req = urllib.request.Request(
                api_url,
                headers={"User-Agent": "parallelism-planner/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                config = data.get("config", {})
        
        return self._normalize_hf_config(config)
    
    def _normalize_hf_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize HuggingFace config to our format."""
        normalized = {}
        
        # Map common field names
        field_mappings = {
            "hidden_size": ["hidden_size", "d_model", "n_embd"],
            "num_layers": ["num_hidden_layers", "n_layer", "num_layers"],
            "num_attention_heads": ["num_attention_heads", "n_head", "num_heads"],
            "num_key_value_heads": ["num_key_value_heads", "n_head_kv"],
            "intermediate_size": ["intermediate_size", "d_ff", "n_inner"],
            "vocab_size": ["vocab_size"],
            "max_position_embeddings": ["max_position_embeddings", "n_positions", "max_seq_len"],
            "num_experts": ["num_local_experts", "num_experts", "n_experts"],
            "num_experts_per_token": ["num_experts_per_tok", "top_k", "num_selected_experts"],
        }
        
        for our_field, hf_fields in field_mappings.items():
            for hf_field in hf_fields:
                if hf_field in config:
                    normalized[our_field] = config[hf_field]
                    break
        
        # Handle nested MoE config
        if "moe" in config:
            moe_config = config["moe"]
            if "num_experts" in moe_config:
                normalized["num_experts"] = moe_config["num_experts"]
            if "top_k" in moe_config:
                normalized["num_experts_per_token"] = moe_config["top_k"]
        
        return normalized
    
    def _create_architecture(
        self, name: str, config: Dict[str, Any]
    ) -> ModelArchitecture:
        """Create ModelArchitecture from config."""
        return ModelArchitecture(
            name=name,
            model_type=ModelType.DENSE,  # Will be updated in __post_init__
            hidden_size=config.get("hidden_size", 4096),
            num_layers=config.get("num_layers", 32),
            num_attention_heads=config.get("num_attention_heads", 32),
            vocab_size=config.get("vocab_size", 32000),
            intermediate_size=config.get("intermediate_size", 0),
            num_key_value_heads=config.get("num_key_value_heads", 0),
            num_experts=config.get("num_experts", 1),
            num_experts_per_token=config.get("num_experts_per_token", 1),
            moe_layer_freq=config.get("moe_layer_freq", 0),
            max_position_embeddings=config.get("max_position_embeddings", 4096),
            dtype_bytes=config.get("dtype_bytes", 2),
        )
    
    def from_custom_spec(
        self,
        name: str,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        vocab_size: int = 32000,
        **kwargs,
    ) -> ModelArchitecture:
        """Create architecture from custom specification."""
        config = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "vocab_size": vocab_size,
            **kwargs,
        }
        return self._create_architecture(name, config)
    
    def list_presets(self) -> List[str]:
        """List available model presets."""
        return list(MODEL_PRESETS.keys())
    
    def format_architecture_report(self, arch: ModelArchitecture) -> str:
        """Generate a human-readable architecture report."""
        lines = [
            "=" * 70,
            f"MODEL ARCHITECTURE: {arch.name}",
            "=" * 70,
            "",
            f"Type: {arch.model_type.value.upper()}",
            f"Total Parameters: {arch.total_params_billion:.2f}B",
        ]
        
        if arch.model_type != ModelType.DENSE:
            lines.append(f"Active Parameters: {arch.active_params_billion:.2f}B")
        
        lines.extend([
            "",
            "DIMENSIONS:",
            f"  Hidden Size: {arch.hidden_size:,}",
            f"  Layers: {arch.num_layers}",
            f"  Attention Heads: {arch.num_attention_heads}",
            f"  KV Heads: {arch.num_key_value_heads} ({arch.attention_type.value})",
            f"  Head Dim: {arch.head_dim}",
            f"  FFN Size: {arch.intermediate_size:,}",
            f"  Vocab Size: {arch.vocab_size:,}",
            f"  Max Sequence: {arch.max_position_embeddings:,}",
        ])
        
        if arch.num_experts > 1:
            lines.extend([
                "",
                "MOE CONFIGURATION:",
                f"  Total Experts: {arch.num_experts}",
                f"  Experts per Token: {arch.num_experts_per_token}",
                f"  MoE Layers: {arch.num_moe_layers}/{arch.num_layers}",
            ])
        
        # Memory estimates
        mem = arch.estimate_memory_gb(batch_size=1, seq_length=2048)
        lines.extend([
            "",
            "MEMORY ESTIMATES (BS=1, Seq=2K, BF16):",
            f"  Weights: {mem['weights_gb']:.2f} GB",
            f"  KV Cache: {mem['kv_cache_gb']:.2f} GB",
            f"  Total Inference: {mem['weights_gb'] + mem['kv_cache_gb']:.2f} GB",
        ])
        
        # Training estimates
        train_mem = arch.estimate_memory_gb(
            batch_size=1, seq_length=2048, include_optimizer=True
        )
        lines.extend([
            "",
            "TRAINING ESTIMATES (BS=1, Seq=2K, BF16+FP32 opt):",
            f"  Total: {train_mem['total_gb']:.2f} GB",
        ])
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    
    analyzer = ModelAnalyzer()
    
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        model_id = "llama-3.1-70b"
    
    try:
        arch = analyzer.analyze(model_id)
        print(analyzer.format_architecture_report(arch))
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nAvailable presets: {analyzer.list_presets()}")

