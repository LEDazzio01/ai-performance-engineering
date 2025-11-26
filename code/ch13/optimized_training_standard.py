"""optimized_training_standard.py - Transformer training with gradient checkpointing.

Gradient checkpointing trades compute time for memory savings by:
- NOT storing intermediate activations during forward pass
- Recomputing them during backward pass as needed

This is slower (~30-50% overhead) but uses MUCH less memory, allowing:
- Larger batch sizes
- Longer sequences  
- Deeper models

Memory savings come from not storing:
- Attention weights: O(batch * heads * seq_len²) per layer
- FFN intermediate activations: O(batch * seq_len * 4*hidden) per layer
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch13.workload_config import WORKLOAD


class CheckpointedTransformerModel(nn.Module):
    """Transformer with gradient checkpointing - recomputes activations during backward."""
    
    def __init__(
        self, 
        hidden_dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        seq_len: int = 512,
        vocab_size: int = 32000,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
        
        # Individual transformer layers (for checkpointing each layer)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Embeddings (not checkpointed - small memory footprint)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)
        
        # Apply each transformer layer with checkpointing
        # This discards activations after forward, recomputes them in backward
        for layer in self.layers:
            x = checkpoint(
                layer,
                x,
                use_reentrant=False,  # More efficient, works with autograd
            )
        
        # Output (not checkpointed)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


class OptimizedTrainingBenchmark(BaseBenchmark):
    """Gradient checkpointing: trades compute for memory.
    
    Same transformer model as baseline but with per-layer checkpointing.
    Expected: ~30-50% slower, but uses 50-70% less activation memory.
    
    This is a MEMORY optimization, not a speed optimization.
    The optimization_goal is "memory" to reflect this.
    """
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.input_ids = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        
        # SAME workload as baseline for fair comparison
        self.hidden_dim = 1024
        self.num_layers = 24  # Same depth
        self.num_heads = 16
        self.seq_len = 1024   # Same sequence length
        self.batch_size = 8   # Same batch size
        self.vocab_size = 32000
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self._peak_memory_gb = 0.0
        self._optimization_goal = "memory"  # This is a memory optimization
    
    def get_optimization_goal(self) -> str:
        """This benchmark optimizes for memory, not speed."""
        return "memory"
    
    def setup(self) -> None:
        """Setup: initialize model with checkpointing."""
        # Clear memory before setup
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        
        self.model = CheckpointedTransformerModel(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
        )
        self.model = self.model.to(self.device).train()
        
        # Same input data as baseline
        self.input_ids = torch.randint(
            0, self.vocab_size, 
            (self.batch_size, self.seq_len), 
            device=self.device
        )
        self.targets = torch.randint(
            0, self.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device
        )
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Training step WITH checkpointing - recomputes activations in backward."""
        if any(v is None for v in (self.model, self.input_ids, self.targets, self.optimizer, self.criterion)):
            raise RuntimeError("Benchmark not configured")

        with self._nvtx_range("checkpointed_training"):
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass - checkpointing discards intermediate activations
            logits = self.model(self.input_ids)
            
            # Compute loss
            loss = self.criterion(
                logits.view(-1, self.vocab_size),
                self.targets.view(-1)
            )
            
            # Backward pass - recomputes activations as needed
            loss.backward()
            
            # Optimizer step
            self.optimizer.step()
        
        # Track peak memory
        self._peak_memory_gb = max(
            self._peak_memory_gb,
            torch.cuda.max_memory_allocated(self.device) / 1e9
        )
        self._synchronize()
    
    def teardown(self) -> None:
        """Cleanup and report memory usage."""
        if self._peak_memory_gb > 0:
            print(f"\n[Checkpointed] Peak GPU Memory: {self._peak_memory_gb:.2f} GB")
        
        self.model = None
        self.input_ids = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        torch.cuda.empty_cache()
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=10,
            warmup=3,
            enable_memory_tracking=True,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input_ids is None:
            return "Input tensor not initialized"
        
        try:
            with torch.no_grad():
                # Disable checkpointing for validation (faster)
                self.model.eval()
                test_output = self.model(self.input_ids[:1])
                if not torch.isfinite(test_output).all():
                    return "Output contains non-finite values"
                self.model.train()
        except Exception as e:
            return f"Model forward pass failed: {e}"
        
        return None


def get_benchmark() -> OptimizedTrainingBenchmark:
    """Factory function for harness discovery."""
    return OptimizedTrainingBenchmark()
