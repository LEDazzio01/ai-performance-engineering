"""optimized_speculative_decoding.py - Optimized speculative decoding in inference/profiling.

Demonstrates speculative decoding for parallel token generation.
Speculative decoding: Uses draft model to predict multiple tokens in parallel.
Accepts/rejects tokens based on target model verification for speedup.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")

class OptimizedSpeculativeDecodingBenchmark(Benchmark):
    """Optimized: Speculative decoding with draft/target coordination."""
    
    def __init__(self):
        self.device = resolve_device()
        self.embedding = None
        self.target_decoder = None
        self.draft_decoder = None
        self.output_head = None
        self.input_ids = None
        self.target_hidden = None
        self.draft_hidden = None
        self.max_length = 64
        self.speculative_length = 4
        self.hidden_dim = 512
        self.vocab_size = 16000
        self.batch_size = 8
        self.seq_len = 64
        self.target_layers = 1
        self.draft_layers = 1
    
    def setup(self) -> None:
        """Setup: Initialize target/draft models and seed hidden states."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim, device=self.device, dtype=dtype)
        self.target_decoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.target_layers,
            batch_first=True,
        ).to(self.device, dtype=dtype).eval()
        self.draft_decoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.draft_layers,
            batch_first=True,
        ).to(self.device, dtype=dtype).eval()
        self.output_head = nn.Linear(self.hidden_dim, self.vocab_size, device=self.device, dtype=dtype)

        try:
            self.target_decoder = torch.compile(self.target_decoder, mode="reduce-overhead")
            self.draft_decoder = torch.compile(self.draft_decoder, mode="reduce-overhead")
            self.output_head = torch.compile(self.output_head, mode="reduce-overhead")
        except Exception:
            pass

        self.input_ids = torch.randint(
            0,
            self.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device,
            dtype=torch.long,
        )
        prompt_embeds = self.embedding(self.input_ids)
        self.target_hidden = torch.zeros(
            self.target_layers,
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=dtype,
        )
        self.draft_hidden = torch.zeros(
            self.draft_layers,
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=dtype,
        )
        with torch.no_grad():
            _, self.target_hidden = self.target_decoder(prompt_embeds, self.target_hidden)
            _, self.draft_hidden = self.draft_decoder(prompt_embeds, self.draft_hidden)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Speculative decoding."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_speculative_decoding", enable=enable_nvtx):
            with torch.no_grad():
                generated = self.input_ids.clone()
                target_hidden = self.target_hidden.clone()
                draft_hidden = self.draft_hidden.clone()
                target_len = generated.size(1) + self.max_length

                while generated.size(1) < target_len:
                    proposals = []
                    proposal_states = []
                    running_hidden = draft_hidden
                    prev_token = generated[:, -1:]
                    for _ in range(self.speculative_length):
                        draft_embed = self.embedding(prev_token)
                        draft_out, running_hidden = self.draft_decoder(draft_embed, running_hidden)
                        logits = self.output_head(draft_out[:, -1, :])
                        next_token = logits.argmax(dim=-1, keepdim=True)
                        proposals.append(next_token)
                        proposal_states.append(running_hidden.clone())
                        prev_token = next_token

                    proposal_tensor = torch.cat(proposals, dim=1)
                    proposal_embeds = self.embedding(proposal_tensor)
                    target_seq, seq_hidden = self.target_decoder(proposal_embeds, target_hidden)
                    target_logits = self.output_head(target_seq)

                    accepted_tokens = 0
                    for idx in range(proposal_tensor.size(1)):
                        proposal = proposal_tensor[:, idx : idx + 1]
                        target_choice = target_logits[:, idx, :].argmax(dim=-1, keepdim=True)
                        if torch.all(target_choice == proposal):
                            generated = torch.cat([generated, proposal], dim=1)
                            draft_hidden = proposal_states[idx]
                            target_hidden = target_seq[:, idx, :].unsqueeze(0)
                            accepted_tokens += 1
                        else:
                            generated = torch.cat([generated, target_choice], dim=1)
                            prev_state = proposal_states[idx - 1] if idx > 0 else draft_hidden
                            correction_embed = self.embedding(target_choice)
                            _, draft_hidden = self.draft_decoder(correction_embed, prev_state)
                            target_hidden = target_seq[:, idx, :].unsqueeze(0)
                            break
                        if generated.size(1) >= target_len:
                            break

                    if accepted_tokens == proposal_tensor.size(1):
                        target_hidden = seq_hidden
                        if generated.size(1) >= target_len:
                            break
        torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.target_decoder = None
        self.draft_decoder = None
        self.embedding = None
        self.output_head = None
        self.input_ids = None
        self.target_hidden = None
        self.draft_hidden = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.target_decoder is None or self.draft_decoder is None or self.output_head is None:
            return "Models not initialized"
        if self.input_ids is None:
            return "Input IDs not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedSpeculativeDecodingBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedSpeculativeDecodingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: speculative_decoding")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
