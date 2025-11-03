import arch_config  # noqa: F401 - Configure architecture optimizations
from arch_config import ArchitectureConfig
import torch.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx
import torch
import os

_ARCH_CFG = ArchitectureConfig()


def get_architecture():
    """Detect and return the current GPU architecture."""
    if not torch.cuda.is_available():
        return "cpu"
    return _ARCH_CFG.arch


def get_architecture_info():
    """Get detailed architecture information."""
    return {
        "name": _ARCH_CFG.get_architecture_name(),
        "compute_capability": _ARCH_CFG.config.get("compute_capability", "Unknown"),
        "sm_version": _ARCH_CFG.config.get("sm_version", "sm_unknown"),
        "memory_bandwidth": _ARCH_CFG.config.get("memory_bandwidth", "Unknown"),
        "tensor_cores": _ARCH_CFG.config.get("tensor_cores", "Unknown"),
        "features": _ARCH_CFG.config.get("features", []),
    }

"""disaggregated_inference.py
Chapter 15: Disaggregated Inference Architectures

Simulated disaggregated prefill-decode benchmarking on Blackwell clusters."""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import threading
import queue
from dataclasses import dataclass
from enum import Enum

class ParallelismStrategy(Enum):
    TENSOR = "tensor"
    PIPELINE = "pipeline" 
    EXPERT = "expert"
    DATA = "data"
    CONTEXT = "context"

@dataclass
class InferenceConfig:
    """Configuration for multi-node inference."""
    model_size: int = 7_000_000_000  # 7B parameters
    num_gpus: int = None  # Will be set to actual GPU count
    batch_size: int = 32
    sequence_length: int = 2048
    num_experts: int = 8
    top_k_experts: int = 2
    capacity_factor: float = 1.2
    use_speculative: bool = True
    use_disaggregated: bool = True
    
    def __post_init__(self):
        """Initialize num_gpus to actual available GPUs if not set."""
        if self.num_gpus is None:
            if torch.cuda.is_available():
                self.num_gpus = torch.cuda.device_count()
            else:
                self.num_gpus = 1  # Fallback to CPU simulation
        # Ensure we don't exceed available GPUs
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            self.num_gpus = min(self.num_gpus, available_gpus)

class DisaggregatedInferenceSystem:
    """Demonstrates disaggregated prefill-decode architecture."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.prefill_workers = []
        self.decode_workers = []
        self.kv_cache = {}
        
    def setup_prefill_workers(self):
        """Initialize dedicated prefill GPU workers."""
        print("Setting up prefill workers...")
        num_prefill_workers = max(1, self.config.num_gpus // 2)
        for i in range(num_prefill_workers):
            gpu_id = i % self.config.num_gpus  # Use modulo to avoid exceeding GPU count
            worker = PrefillWorker(
                worker_id=i,
                gpu_id=gpu_id,
                config=self.config
            )
            self.prefill_workers.append(worker)
            
    def setup_decode_workers(self):
        """Initialize dedicated decode GPU workers."""
        print("Setting up decode workers...")
        num_decode_workers = max(1, self.config.num_gpus // 2)
        for i in range(num_decode_workers):
            gpu_id = (i + self.config.num_gpus // 2) % self.config.num_gpus  # Use modulo to avoid exceeding GPU count
            worker = DecodeWorker(
                worker_id=i,
                gpu_id=gpu_id,
                config=self.config
            )
            self.decode_workers.append(worker)
            
    def process_request(self, prompt: str) -> str:
        """Process a request using disaggregated architecture."""
        # Phase 1: Prefill on dedicated workers
        kv_cache = self.prefill_phase(prompt)
        
        # Phase 2: Transfer KV cache to decode workers
        self.transfer_kv_cache(kv_cache)
        
        # Phase 3: Decode on dedicated workers
        response = self.decode_phase()
        
        return response
        
    def prefill_phase(self, prompt: str) -> Dict:
        """Process prompt and generate KV cache."""
        print(f"Prefill phase: Processing prompt of length {len(prompt)}")
        
        # Simulate prefill computation
        start_time = time.time()
        
        # Distribute across prefill workers
        kv_cache = {}
        for worker in self.prefill_workers:
            worker_kv = worker.process_prompt(prompt)
            kv_cache.update(worker_kv)
            
        prefill_time = time.time() - start_time
        print(f"Prefill completed in {prefill_time:.3f}s")
        
        return kv_cache
        
    def transfer_kv_cache(self, kv_cache: Dict):
        """Transfer KV cache from prefill to decode workers."""
        print("Transferring KV cache to decode workers...")
        
        # Simulate high-bandwidth transfer
        transfer_size = len(str(kv_cache))  # Simplified size calculation
        transfer_time = transfer_size / (100 * 1024 * 1024)  # 100 MB/s transfer rate
        
        time.sleep(transfer_time)
        
        # Distribute KV cache to decode workers
        for worker in self.decode_workers:
            worker.load_kv_cache(kv_cache)
            
    def decode_phase(self) -> str:
        """Generate response using decode workers."""
        print("Decode phase: Generating response tokens...")
        
        start_time = time.time()
        response_tokens = []
        
        # Generate tokens autoregressively
        for i in range(100):  # Generate up to 100 tokens
            token = self.decode_workers[0].generate_next_token()
            response_tokens.append(token)
            
            if token == "<EOS>":
                break
                
        decode_time = time.time() - start_time
        print(f"Decode completed in {decode_time:.3f}s")
        
        return " ".join(response_tokens)

class PrefillWorker:
    """Dedicated worker for prompt processing."""
    
    def __init__(self, worker_id: int, gpu_id: int, config: InferenceConfig):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.config = config
        
        # Handle device assignment
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        
        # Initialize model components
        self.attention_layers = self._create_attention_layers()
        self.ffn_layers = self._create_ffn_layers()
        
    def _create_attention_layers(self) -> nn.ModuleList:
        """Create attention layers for prefill."""
        layers = nn.ModuleList()
        for i in range(6):  # 6 attention layers
            layer = nn.MultiheadAttention(
                embed_dim=4096,
                num_heads=32,
                batch_first=True
            )
            layer = layer.to(self.device)
            layers.append(layer)
        return layers
        
    def _create_ffn_layers(self) -> nn.ModuleList:
        """Create feed-forward layers for prefill."""
        layers = nn.ModuleList()
        for i in range(6):  # 6 FFN layers
            layer = nn.Sequential(
                nn.Linear(4096, 16384),
                nn.GELU(),
                nn.Linear(16384, 4096)
            )
            layer = layer.to(self.device)
            layers.append(layer)
        return layers
        
    def process_prompt(self, prompt: str) -> Dict:
        """Process prompt and generate KV cache."""
        # Tokenize prompt
        tokens = self._tokenize(prompt)
        
        # Convert tokens to embeddings (simulate embedding layer)
        # In practice, this would use a proper embedding layer
        embed_dim = 4096
        x = torch.randn(1, len(tokens), embed_dim, device=self.device)  # batch_size=1, seq_len, embed_dim
        
        kv_cache = {}
        
        # Process through attention layers
        for i, layer in enumerate(self.attention_layers):
            # Generate KV cache for this layer
            with torch.no_grad():
                # Use the attention layer without cache first
                output = layer(x, x, x, need_weights=False)
                
                # Simulate KV cache generation
                # In practice, this would extract key and value tensors from the attention computation
                batch_size, seq_len, embed_dim = x.shape
                key_cache = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
                value_cache = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
                
            kv_cache[f"layer_{i}_k"] = key_cache
            kv_cache[f"layer_{i}_v"] = value_cache
            
            # Process through FFN
            x = self.ffn_layers[i](x)
            
        return kv_cache
        
    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization for demonstration."""
        return [ord(c) % 1000 for c in text[:self.config.sequence_length]]

class DecodeWorker:
    """Dedicated worker for token generation."""
    
    def __init__(self, worker_id: int, gpu_id: int, config: InferenceConfig):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.config = config
        
        # Handle device assignment
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
            
        self.kv_cache = {}
        
        # Initialize model components
        self.attention_layers = self._create_attention_layers()
        self.ffn_layers = self._create_ffn_layers()
        self.lm_head = nn.Linear(4096, 50000)  # Vocab size
        self.lm_head = self.lm_head.to(self.device)
        
    def _create_attention_layers(self) -> nn.ModuleList:
        """Create attention layers for decode."""
        layers = nn.ModuleList()
        for i in range(6):  # 6 attention layers
            layer = nn.MultiheadAttention(
                embed_dim=4096,
                num_heads=32,
                batch_first=True
            )
            layer = layer.to(self.device)
            layers.append(layer)
        return layers
        
    def _create_ffn_layers(self) -> nn.ModuleList:
        """Create feed-forward layers for decode."""
        layers = nn.ModuleList()
        for i in range(6):  # 6 FFN layers
            layer = nn.Sequential(
                nn.Linear(4096, 16384),
                nn.GELU(),
                nn.Linear(16384, 4096)
            )
            layer = layer.to(self.device)
            layers.append(layer)
        return layers
        
    def load_kv_cache(self, kv_cache: Dict):
        """Load KV cache from prefill phase."""
        self.kv_cache = kv_cache
        
    def generate_next_token(self) -> str:
        """Generate next token using cached KV values."""
        # Simulate token generation
        if not self.kv_cache:
            return "<EOS>"
            
        # Use cached KV values for efficient generation
        # In practice, this would use the cached key/value pairs
        # to avoid recomputing attention for previous tokens
        
        # Simulate autoregressive generation
        vocab = ["the", "a", "is", "was", "in", "on", "at", "to", "for", "<EOS>"]
        # Normalize probabilities to sum to 1.0
        probs = np.array([0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.0])
        probs = probs / probs.sum()  # Normalize to sum to 1.0
        token = np.random.choice(vocab, p=probs)
        
        return token

class MoERouter:
    """Demonstrates MoE routing and load balancing."""
    
    def __init__(self, num_experts: int, top_k: int, capacity_factor: float):
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.expert_loads = {i: 0 for i in range(num_experts)}
        self.expert_capacities = {i: int(32 * capacity_factor) for i in range(num_experts)}
        
    def route_tokens(self, tokens: List[int]) -> Dict[int, List[int]]:
        """Route tokens to experts using load balancing."""
        expert_assignments = {i: [] for i in range(self.num_experts)}
        
        for token in tokens:
            # Get expert preferences for this token
            expert_scores = self._get_expert_scores(token)
            
            # Select top-k experts
            top_experts = sorted(
                range(self.num_experts),
                key=lambda x: expert_scores[x],
                reverse=True
            )[:self.top_k]
            
            # Route to available expert with highest score
            routed = False
            for expert_id in top_experts:
                if self.expert_loads[expert_id] < self.expert_capacities[expert_id]:
                    expert_assignments[expert_id].append(token)
                    self.expert_loads[expert_id] += 1
                    routed = True
                    break
                    
            if not routed:
                # Overflow to next available expert
                for expert_id in range(self.num_experts):
                    if self.expert_loads[expert_id] < self.expert_capacities[expert_id]:
                        expert_assignments[expert_id].append(token)
                        self.expert_loads[expert_id] += 1
                        break
                        
        return expert_assignments
        
    def _get_expert_scores(self, token: int) -> List[float]:
        """Get expert preference scores for a token."""
        # Simulate expert gating network
        scores = np.random.random(self.num_experts)
        return scores.tolist()
        
    def get_load_balance_metrics(self) -> Dict:
        """Get load balancing metrics."""
        loads = list(self.expert_loads.values())
        return {
            "mean_load": np.mean(loads),
            "std_load": np.std(loads),
            "max_load": max(loads),
            "min_load": min(loads),
            "load_imbalance": max(loads) - min(loads)
        }

class SpeculativeDecoder:
    """Demonstrates speculative decoding techniques."""
    
    def __init__(self, draft_model_size: int = 1_000_000_000):
        self.draft_model_size = draft_model_size
        self.acceptance_rate = 0.7  # 70% acceptance rate
        
    def speculative_decode(self, prompt: str, target_tokens: int = 10) -> List[str]:
        """Generate tokens using speculative decoding."""
        print(f"Speculative decoding: Generating {target_tokens} tokens")
        
        generated_tokens = []
        current_position = 0
        
        while len(generated_tokens) < target_tokens:
            # Draft phase: Generate multiple candidate tokens
            draft_tokens = self._draft_phase(prompt, generated_tokens, num_draft=4)
            
            # Verify phase: Check draft tokens against target model
            accepted_tokens = self._verify_phase(prompt, generated_tokens, draft_tokens)
            
            # Accept correct prefix
            generated_tokens.extend(accepted_tokens)
            current_position += len(accepted_tokens)
            
            # If no tokens accepted, fall back to single token generation
            if len(accepted_tokens) == 0:
                single_token = self._generate_single_token(prompt, generated_tokens)
                generated_tokens.append(single_token)
                current_position += 1
                
        return generated_tokens
        
    def _draft_phase(self, prompt: str, context: List[str], num_draft: int) -> List[str]:
        """Generate draft tokens using smaller model."""
        print(f"Draft phase: Generating {num_draft} candidate tokens")
        
        # Simulate draft model generation
        vocab = ["the", "a", "is", "was", "in", "on", "at", "to", "for", "with"]
        draft_tokens = np.random.choice(vocab, num_draft).tolist()
        
        return draft_tokens
        
    def _verify_phase(self, prompt: str, context: List[str], draft_tokens: List[str]) -> List[str]:
        """Verify draft tokens against target model."""
        print(f"Verify phase: Checking {len(draft_tokens)} draft tokens")
        
        accepted_tokens = []
        
        for i, draft_token in enumerate(draft_tokens):
            # Simulate verification with target model
            if np.random.random() < self.acceptance_rate:
                accepted_tokens.append(draft_token)
            else:
                # Stop at first rejection
                break
                
        print(f"Accepted {len(accepted_tokens)} out of {len(draft_tokens)} draft tokens")
        return accepted_tokens
        
    def _generate_single_token(self, prompt: str, context: List[str]) -> str:
        """Generate single token using target model."""
        vocab = ["the", "a", "is", "was", "in", "on", "at", "to", "for", "with"]
        return np.random.choice(vocab)

class ParallelismManager:
    """Manages different parallelism strategies."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.strategy = None
        
    def setup_tensor_parallelism(self):
        """Setup tensor parallelism across GPUs."""
        print("Setting up tensor parallelism...")
        
        # Split model layers across GPUs
        num_layers_per_gpu = 6 // self.config.num_gpus
        
        for gpu_id in range(self.config.num_gpus):
            start_layer = gpu_id * num_layers_per_gpu
            end_layer = start_layer + num_layers_per_gpu
            
            print(f"GPU {gpu_id}: Layers {start_layer}-{end_layer}")
            
    def setup_pipeline_parallelism(self):
        """Setup pipeline parallelism across GPUs."""
        print("Setting up pipeline parallelism...")
        
        # Each GPU handles different stages
        stages_per_gpu = 1
        total_stages = self.config.num_gpus * stages_per_gpu
        
        for gpu_id in range(self.config.num_gpus):
            stage_start = gpu_id * stages_per_gpu
            stage_end = stage_start + stages_per_gpu
            
            print(f"GPU {gpu_id}: Pipeline stages {stage_start}-{stage_end}")
            
    def setup_expert_parallelism(self):
        """Setup expert parallelism for MoE."""
        print("Setting up expert parallelism...")
        
        experts_per_gpu = self.config.num_experts // self.config.num_gpus
        
        for gpu_id in range(self.config.num_gpus):
            start_expert = gpu_id * experts_per_gpu
            end_expert = start_expert + experts_per_gpu
            
            print(f"GPU {gpu_id}: Experts {start_expert}-{end_expert}")
            
    def setup_data_parallelism(self):
        """Setup data parallelism."""
        print("Setting up data parallelism...")
        
        # Each GPU gets a full model replica
        for gpu_id in range(self.config.num_gpus):
            print(f"GPU {gpu_id}: Full model replica")
            
    def setup_context_parallelism(self):
        """Setup context parallelism for long sequences."""
        print("Setting up context parallelism...")
        
        # Split sequence across GPUs
        tokens_per_gpu = self.config.sequence_length // self.config.num_gpus
        
        for gpu_id in range(self.config.num_gpus):
            start_token = gpu_id * tokens_per_gpu
            end_token = start_token + tokens_per_gpu
            
            print(f"GPU {gpu_id}: Tokens {start_token}-{end_token}")

def benchmark_inference_system():
    """Benchmark the disaggregated inference system."""
    print("=== Multi-Node Inference Benchmark ===\n")
    
    # Configuration - will automatically detect available GPUs
    config = InferenceConfig(
        model_size=7_000_000_000,
        batch_size=32,
        sequence_length=2048,
        num_experts=8,
        top_k_experts=2,
        capacity_factor=1.2,
        use_speculative=True,
        use_disaggregated=True
    )
    
    print(f"Using {config.num_gpus} GPU(s) for inference")
    
    # Test disaggregated inference
    print("1. Testing Disaggregated Prefill-Decode Architecture")
    system = DisaggregatedInferenceSystem(config)
    system.setup_prefill_workers()
    system.setup_decode_workers()
    
    prompt = "The quick brown fox jumps over the lazy dog. " * 50
    response = system.process_request(prompt)
    print(f"Response: {response[:100]}...\n")
    
    # Test MoE routing
    print("2. Testing MoE Routing and Load Balancing")
    router = MoERouter(
        num_experts=config.num_experts,
        top_k=config.top_k_experts,
        capacity_factor=config.capacity_factor
    )
    
    tokens = list(range(100))  # Simulate 100 tokens
    assignments = router.route_tokens(tokens)
    
    print("Expert assignments:")
    for expert_id, assigned_tokens in assignments.items():
        print(f"Expert {expert_id}: {len(assigned_tokens)} tokens")
        
    metrics = router.get_load_balance_metrics()
    print(f"Load balance metrics: {metrics}\n")
    
    # Test speculative decoding
    print("3. Testing Speculative Decoding")
    decoder = SpeculativeDecoder()
    speculative_tokens = decoder.speculative_decode(prompt, target_tokens=20)
    print(f"Speculative generation: {' '.join(speculative_tokens)}\n")
    
    # Test parallelism strategies
    print("4. Testing Parallelism Strategies")
    manager = ParallelismManager(config)
    
    print("Tensor Parallelism:")
    manager.setup_tensor_parallelism()
    
    print("\nPipeline Parallelism:")
    manager.setup_pipeline_parallelism()
    
    print("\nExpert Parallelism:")
    manager.setup_expert_parallelism()
    
    print("\nData Parallelism:")
    manager.setup_data_parallelism()
    
    print("\nContext Parallelism:")
    manager.setup_context_parallelism()
    
    print("\n=== Benchmark Complete ===")

if __name__ == "__main__":
    benchmark_inference_system()

# Architecture-specific optimizations
if torch.cuda.is_available():
    inductor = getattr(torch, "_inductor", None)
    triton_cfg = getattr(getattr(inductor, "config", None), "triton", None) if inductor else None

    if _ARCH_CFG.arch in {"blackwell", "grace_blackwell"} and triton_cfg is not None:
        try:
            if hasattr(triton_cfg, "use_blackwell_optimizations"):
                triton_cfg.use_blackwell_optimizations = True
            if hasattr(triton_cfg, "hbm3e_optimizations"):
                triton_cfg.hbm3e_optimizations = True
            if hasattr(triton_cfg, "tma_support"):
                triton_cfg.tma_support = True
            if hasattr(triton_cfg, "stream_ordered_memory"):
                triton_cfg.stream_ordered_memory = True
        except AttributeError:
            print("Blackwell optimizations not available in this PyTorch build")

    if triton_cfg is not None and hasattr(triton_cfg, "unique_kernel_names"):
        triton_cfg.unique_kernel_names = True
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.automatic_dynamic_shapes = True
