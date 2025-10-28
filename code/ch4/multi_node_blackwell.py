"""
Multi-Node Distributed Training for Blackwell GPUs

Demonstrates multi-node distributed training optimizations for Blackwell
B200/B300 GPUs with NVLink-C2C and NCCL. Includes tensor parallelism,
FSDP, and gradient compression for multi-node scaling.
"""
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.device_mesh import init_device_mesh
import torch.nn.functional as F
from typing import Tuple, Optional
import time


# ============================================================================
# Environment Setup for Blackwell Multi-Node
# ============================================================================

def detect_gb200_gb300() -> bool:
    """Detect if running on GB200/GB300 Grace-Blackwell Superchip."""
    try:
        import platform
        if platform.machine() == 'aarch64':
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'ARM' in cpuinfo or 'Neoverse' in cpuinfo:
                    return True
    except:
        pass
    return False


def setup_8xb200_optimized(
    backend: str = "nccl",
    tp_size: int = 2,
    dp_size: int = 4,
) -> Tuple[int, int, int, int]:
    """
    Initialize distributed environment optimized for 8x B200 GPUs.
    
    8x B200 Configuration:
    - Total: 1184 SMs, 1.44 TB HBM3e, 62.4 TB/s bandwidth
    - NVLink 5.0: 1800 GB/s bidirectional per GPU pair
    - Recommended: TP=2/4, DP=4/2 for hybrid parallelism
    
    Args:
        backend: Communication backend (default: nccl)
        tp_size: Tensor parallel size (2 or 4 for 8 GPUs)
        dp_size: Data parallel size (4 or 2 for 8 GPUs)
        
    Returns: (rank, local_rank, world_size, local_world_size)
    """
    assert tp_size * dp_size == 8, f"tp_size * dp_size must equal 8 for 8x B200, got {tp_size} * {dp_size}"
    
    # Use optimized NCCL config
    try:
        from ch4.nccl_blackwell_config import configure_nccl_for_8xB200, detect_8xb200_topology
        topology = detect_8xb200_topology()
        
        if topology.get("is_8xb200", False):
            # Detect if Grace-Blackwell
            is_grace = detect_gb200_gb300()
            if is_grace:
                from ch4.nccl_blackwell_config import configure_nccl_for_gb200_gb300
                configure_nccl_for_gb200_gb300(verbose=False)
            else:
                configure_nccl_for_8xB200(num_channels=8, verbose=False)
    except ImportError:
        pass  # Fall back to manual config
    
    return setup_blackwell_distributed(backend=backend)


def setup_blackwell_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> Tuple[int, int, int, int]:
    """Initialize distributed environment optimized for Blackwell.
    
    Returns: (rank, local_rank, world_size, local_world_size)
    """
    # Get environment variables
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = torch.cuda.device_count()
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Configure NCCL for Blackwell
    if backend == "nccl":
        # Detect 8x B200 configuration
        is_8gpu = local_world_size == 8
        
        # Blackwell-specific NCCL optimizations
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")  # Adjust as needed
        os.environ.setdefault("NCCL_IB_DISABLE", "0")  # Enable InfiniBand if available
        os.environ.setdefault("NCCL_NET_GDR_LEVEL", "5")  # GPU Direct RDMA
        os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")  # NVLink for intra-node
        os.environ.setdefault("NCCL_NVLS_ENABLE", "1")  # NVLink-C2C
        os.environ.setdefault("NCCL_PROTO", "Simple")  # Protocol
        
        # Optimize for 8x B200 (148 SMs per GPU = 1184 total)
        if is_8gpu:
            os.environ.setdefault("NCCL_ALGO", "Tree,Ring,NVLS")  # Add NVLS for 8 GPUs
            os.environ.setdefault("NCCL_NCHANNELS_PER_NET_PEER", "16")  # More channels for 8 GPUs
            os.environ.setdefault("NCCL_BUFFSIZE", str(64 * 1024 * 1024))  # 64MB for 1.44TB total
            os.environ.setdefault("NCCL_MIN_NCHANNELS", "8")
            os.environ.setdefault("NCCL_MAX_NCHANNELS", "32")
        else:
            os.environ.setdefault("NCCL_ALGO", "Tree,Ring")
            os.environ.setdefault("NCCL_NCHANNELS_PER_NET_PEER", "8")
            os.environ.setdefault("NCCL_BUFFSIZE", "8388608")  # 8MB
        
        if rank == 0:
            print("=" * 70)
            print(f"NCCL Configuration for Blackwell ({'8x B200' if is_8gpu else 'Standard'})")
            print("=" * 70)
            if is_8gpu:
                print("  🚀 8x B200 Optimizations Enabled:")
                print(f"     Total SMs: 1184 (148 per GPU)")
                print(f"     Total Memory: 1.44 TB HBM3e")
                print(f"     Aggregate Bandwidth: 62.4 TB/s")
                print(f"     NVLink 5.0: 1800 GB/s per pair")
            print(f"  P2P Level: {os.environ.get('NCCL_P2P_LEVEL', 'default')}")
            print(f"  NVLS (NVLink-C2C): {os.environ.get('NCCL_NVLS_ENABLE', '0')}")
            print(f"  Algorithms: {os.environ.get('NCCL_ALGO', 'default')}")
            print(f"  Channels per peer: {os.environ.get('NCCL_NCHANNELS_PER_NET_PEER', 'default')}")
            print(f"  Buffer size: {int(os.environ.get('NCCL_BUFFSIZE', '2097152')) / 1024 / 1024:.1f} MB")
            if detect_gb200_gb300():
                print("  ✓ GB200/GB300 Grace-Blackwell detected")
                print("    → CPU-GPU: 900 GB/s NVLink-C2C")
            print("=" * 70)
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("Distributed Training Setup")
        print("=" * 70)
        print(f"  World size: {world_size}")
        print(f"  Ranks per node: {local_world_size}")
        print(f"  Number of nodes: {world_size // local_world_size}")
        print(f"  Backend: {backend}")
        if local_world_size == 8:
            print("\n  Recommended Parallelism for 8x B200:")
            print("    - TP=2, DP=4 (moderate model size, <20B params)")
            print("    - TP=4, DP=2 (large model, 20B-100B params)")
            print("    - TP=8, DP=1 (very large model, >100B params)")
        print("=" * 70 + "\n")
    
    return rank, local_rank, world_size, local_world_size


# ============================================================================
# Multi-Node Model with Hybrid Parallelism
# ============================================================================

class MultiNodeTransformerBlock(nn.Module):
    """Transformer block with hybrid parallelism for multi-node training."""
    
    def __init__(
        self,
        d_model: int = 1024,
        num_heads: int = 16,
        d_ff: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Multi-head attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Feed-forward
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        residual = x
        x = self.attn_norm(x)
        
        # Multi-head attention (simplified)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.1 if self.training else 0.0,
        )
        
        x = self.o_proj(attn_out)
        x = self.dropout(x)
        x = x + residual
        
        # Feed-forward block
        residual = x
        x = self.ff_norm(x)
        x = self.ff1(x)
        x = F.gelu(x)
        x = self.ff2(x)
        x = self.dropout(x)
        x = x + residual
        
        return x


class MultiNodeTransformer(nn.Module):
    """
    Transformer model for multi-node training.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        d_ff: int = 4096,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            MultiNodeTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        
        # Embeddings
        x = self.embedding(input_ids)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(pos)
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


def create_8xb200_device_mesh(tp_size: int = 2, dp_size: int = 4):
    """
    Create optimal 2D device mesh for 8x B200 GPUs.
    
    8x B200 Recommended Configurations:
    - TP=2, DP=4: Moderate models (<20B params), balanced communication
    - TP=4, DP=2: Large models (20-100B params), more model parallelism
    - TP=8, DP=1: Very large models (>100B params), maximum model parallelism
    
    Args:
        tp_size: Tensor parallel size (2, 4, or 8)
        dp_size: Data parallel size (4, 2, or 1)
        
    Returns:
        2D device mesh with "dp" and "tp" dimensions
    """
    assert tp_size * dp_size == 8, f"tp_size * dp_size must equal 8, got {tp_size} * {dp_size}"
    
    # Create 2D mesh: [data_parallel, tensor_parallel]
    # TP uses intra-node NVLink (low latency, high bandwidth)
    # DP uses inter-node or remaining GPUs (optimized by NCCL)
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    
    rank = dist.get_rank()
    if rank == 0:
        print("\n" + "=" * 70)
        print(f"8x B200 Device Mesh Configuration")
        print("=" * 70)
        print(f"  Mesh shape: ({dp_size}, {tp_size}) - (DP, TP)")
        print(f"  Total GPUs: {dp_size * tp_size}")
        print(f"\nCommunication Patterns:")
        print(f"  Tensor Parallel: {tp_size} GPUs via NVLink 5.0")
        print(f"    → Within-group: {1800 * (tp_size - 1) // 2:.0f} GB/s aggregate")
        print(f"  Data Parallel: {dp_size} groups")
        print(f"    → Cross-group: NCCL AllReduce")
        print("\nMemory Distribution:")
        print(f"  Per TP group: {180 * tp_size:.0f} GB (model sharded {tp_size}x)")
        print(f"  Total capacity: {180 * 8:.0f} GB HBM3e")
        print("=" * 70)
    
    return device_mesh


def apply_tensor_parallelism(
    model: nn.Module,
    device_mesh,
) -> nn.Module:
    """
    Apply tensor parallelism to model using PyTorch 2.9 API.
    
    Optimized for intra-node parallelism via NVLink-C2C.
    """
    # Parallelize attention projections
    for i, layer in enumerate(model.layers):
        # Column-wise parallel for Q, K, V projections
        parallelize_module(
            layer,
            device_mesh["tp"],
            {
                "q_proj": ColwiseParallel(),
                "k_proj": ColwiseParallel(),
                "v_proj": ColwiseParallel(),
            },
        )
        
        # Row-wise parallel for output projection
        parallelize_module(
            layer,
            device_mesh["tp"],
            {"o_proj": RowwiseParallel()},
        )
        
        # Column-wise for FF1, Row-wise for FF2
        parallelize_module(
            layer,
            device_mesh["tp"],
            {
                "ff1": ColwiseParallel(),
                "ff2": RowwiseParallel(),
            },
        )
    
    return model


def apply_fsdp(
    model: nn.Module,
    device_mesh,
    mixed_precision: bool = True,
) -> FSDP:
    """
    Apply FSDP (Fully Sharded Data Parallel) for data parallelism.
    
    Optimized for inter-node communication.
    """
    # Mixed precision policy
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mp_policy = None
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        device_mesh=device_mesh["dp"],
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        use_orig_params=True,  # For optimizer compatibility
    )
    
    return model


# ============================================================================
# Multi-Node Training Loop
# ============================================================================

def train_multi_node(
    model: nn.Module,
    train_data: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    gradient_accumulation_steps: int = 4,
    use_compile: bool = True,
) -> dict:
    """
    Multi-node distributed training with optimizations.
    
    Optimizations:
    - Gradient accumulation
    - Mixed precision (BF16)
    - torch.compile
    - Communication overlap
    - Gradient compression (future)
    
    Returns:
        Training statistics
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()
    
    # Compile model for Blackwell
    if use_compile:
        model = torch.compile(
            model,
            mode="max-autotune",
            backend="inductor",
        )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_data),
    )
    
    # Training stats
    stats = {
        'losses': [],
        'throughputs': [],
        'step_times': [],
    }
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("Starting Multi-Node Training")
        print("=" * 70)
        print(f"  Nodes: {world_size // torch.cuda.device_count()}")
        print(f"  GPUs per node: {torch.cuda.device_count()}")
        print(f"  Total GPUs: {world_size}")
        print(f"  Mixed precision: BF16")
        print(f"  torch.compile: {use_compile}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print("=" * 70 + "\n")
    
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        
        for step, batch in enumerate(train_data):
            step_start = time.time()
            
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            
            # Backward pass with gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
            
            step_time = time.time() - step_start
            
            # Log statistics
            if rank == 0 and step % 10 == 0:
                tokens_per_step = input_ids.numel() * world_size
                throughput = tokens_per_step / step_time
                
                stats['losses'].append(loss.item() * gradient_accumulation_steps)
                stats['throughputs'].append(throughput)
                stats['step_times'].append(step_time)
                
                print(f"Epoch {epoch} | Step {step:4d} | "
                      f"Loss: {loss.item()*gradient_accumulation_steps:.4f} | "
                      f"Throughput: {throughput/1e6:.2f}M tokens/s | "
                      f"Time: {step_time*1000:.1f}ms")
            
            epoch_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        
        if rank == 0:
            avg_loss = epoch_loss / len(train_data)
            print(f"\nEpoch {epoch} complete:")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Epoch time: {epoch_time:.2f}s")
            print(f"  Throughput: {len(train_data) * input_ids.numel() * world_size / epoch_time / 1e6:.2f}M tokens/s")
    
    return stats


# ============================================================================
# Bandwidth Benchmark
# ============================================================================

def benchmark_8xb200_bandwidth(
    sizes: list[int] = [1024*1024, 10*1024*1024, 100*1024*1024, 500*1024*1024],
    num_iters: int = 100,
) -> dict:
    """
    Comprehensive bandwidth benchmark for 8x B200 GPUs.
    
    Tests:
    - All-reduce (NVLS, Ring, Tree algorithms)
    - Point-to-point (NVLink 5.0)
    - All-gather
    - Reduce-scatter
    
    Expected Performance:
    - All-reduce: 700-800 GB/s bus bandwidth
    - P2P: 800-900 GB/s per pair (NVLink 5.0)
    - Scaling efficiency: 85-95%
    
    Returns:
        Bandwidth measurements
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()
    
    if world_size != 8:
        if rank == 0:
            print(f"Warning: This benchmark is optimized for 8 GPUs, got {world_size}")
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("8x B200 Bandwidth Benchmark")
        print("=" * 70)
        print(f"GPUs: {world_size}")
        print(f"Expected: 700-800 GB/s bus bandwidth (NVLink 5.0)")
        print("=" * 70)
    
    results = {}
    
    # Test 1: All-reduce (primary collective for training)
    if rank == 0:
        print("\n[1/4] All-Reduce Benchmark:")
    
    for size in sizes:
        tensor = torch.randn(size, device=device, dtype=torch.float32)
        bytes_transferred = tensor.numel() * tensor.element_size()
        
        # Warmup
        for _ in range(10):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        
        # All-reduce benchmark
        start = time.time()
        for _ in range(num_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Calculate bandwidth (algorithm bandwidth: 2(N-1)/N factor)
        algo_bw_factor = 2.0 * (world_size - 1) / world_size
        bandwidth_gbs = (bytes_transferred * algo_bw_factor * num_iters) / elapsed / 1e9
        
        results[f"allreduce_{bytes_transferred/1e6:.1f}MB"] = bandwidth_gbs
        
        if rank == 0:
            print(f"  Size: {bytes_transferred/1e6:8.1f} MB | "
                  f"Bandwidth: {bandwidth_gbs:6.2f} GB/s | "
                  f"Time: {elapsed*1000/num_iters:6.2f} ms")
    
    # Test 2: Point-to-point (NVLink 5.0 test)
    if rank == 0:
        print("\n[2/4] P2P Bandwidth (GPU 0 ↔ GPU 1):")
    
    if rank == 0 or rank == 1:
        for size in sizes[:3]:  # Fewer sizes for P2P
            tensor = torch.randn(size, device=device, dtype=torch.float32)
            bytes_transferred = tensor.numel() * tensor.element_size()
            
            # Warmup
            for _ in range(10):
                if rank == 0:
                    dist.send(tensor, dst=1)
                else:
                    dist.recv(tensor, src=0)
            torch.cuda.synchronize()
            
            # P2P benchmark
            start = time.time()
            for _ in range(num_iters):
                if rank == 0:
                    dist.send(tensor, dst=1)
                else:
                    dist.recv(tensor, src=0)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            bandwidth_gbs = (bytes_transferred * num_iters) / elapsed / 1e9
            
            if rank == 0:
                print(f"  Size: {bytes_transferred/1e6:8.1f} MB | "
                      f"Bandwidth: {bandwidth_gbs:6.2f} GB/s")
    
    dist.barrier()
    
    # Test 3: All-gather
    if rank == 0:
        print("\n[3/4] All-Gather Benchmark:")
    
    for size in sizes[1:3]:  # Medium sizes
        tensor = torch.randn(size, device=device, dtype=torch.float32)
        output_list = [torch.empty_like(tensor) for _ in range(world_size)]
        bytes_transferred = tensor.numel() * tensor.element_size()
        
        # Warmup
        for _ in range(10):
            dist.all_gather(output_list, tensor)
        torch.cuda.synchronize()
        
        # All-gather benchmark
        start = time.time()
        for _ in range(num_iters):
            dist.all_gather(output_list, tensor)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Bandwidth: each GPU receives (N-1) * size
        bandwidth_gbs = (bytes_transferred * (world_size - 1) * num_iters) / elapsed / 1e9
        
        if rank == 0:
            print(f"  Size: {bytes_transferred/1e6:8.1f} MB | "
                  f"Bandwidth: {bandwidth_gbs:6.2f} GB/s")
    
    # Test 4: Reduce-scatter
    if rank == 0:
        print("\n[4/4] Reduce-Scatter Benchmark:")
    
    for size in sizes[1:3]:
        # Reduce-scatter splits input across GPUs
        tensor = torch.randn(size * world_size, device=device, dtype=torch.float32)
        output = torch.empty(size, device=device, dtype=torch.float32)
        input_list = list(tensor.chunk(world_size))
        bytes_transferred = tensor.numel() * tensor.element_size()
        
        # Warmup
        for _ in range(10):
            dist.reduce_scatter(output, input_list)
        torch.cuda.synchronize()
        
        # Reduce-scatter benchmark
        start = time.time()
        for _ in range(num_iters):
            dist.reduce_scatter(output, input_list)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Bandwidth calculation
        bandwidth_gbs = (bytes_transferred * num_iters) / elapsed / 1e9
        
        if rank == 0:
            print(f"  Size: {bytes_transferred/1e6:8.1f} MB | "
                  f"Bandwidth: {bandwidth_gbs:6.2f} GB/s")
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("Benchmark Complete!")
        avg_allreduce = sum(v for k, v in results.items() if "allreduce" in k) / len([k for k in results if "allreduce" in k])
        print(f"Average All-Reduce Bandwidth: {avg_allreduce:.2f} GB/s")
        print(f"Expected Range: 700-800 GB/s for 8x B200")
        efficiency = (avg_allreduce / 750) * 100  # 750 GB/s target
        print(f"Efficiency: {efficiency:.1f}% of target")
        print("=" * 70 + "\n")
    
    return results


def benchmark_multi_node_bandwidth(
    sizes: list[int] = [1024*1024, 10*1024*1024, 100*1024*1024],
    num_iters: int = 100,
) -> dict:
    """
    Benchmark inter-node and intra-node bandwidth.
    
    Tests:
    - All-reduce (ring algorithm)
    - All-gather
    - Reduce-scatter
    - Point-to-point
    
    Returns:
        Bandwidth measurements
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()
    
    # Use 8x B200 optimized benchmark if applicable
    local_world_size = torch.cuda.device_count()
    if local_world_size == 8 and world_size == 8:
        return benchmark_8xb200_bandwidth(sizes, num_iters)
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("Multi-Node Bandwidth Benchmark")
        print("=" * 70)
    
    results = {}
    
    for size in sizes:
        tensor = torch.randn(size, device=device, dtype=torch.float32)
        bytes_transferred = tensor.numel() * tensor.element_size()
        
        # Warmup
        for _ in range(10):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        
        # All-reduce benchmark
        start = time.time()
        for _ in range(num_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Calculate bandwidth
        # All-reduce transfers 2(N-1)/N * data_size
        algo_bw_factor = 2.0 * (world_size - 1) / world_size
        bandwidth_gbs = (bytes_transferred * algo_bw_factor * num_iters) / elapsed / 1e9
        
        results[f"{bytes_transferred/1e6:.1f}MB"] = bandwidth_gbs
        
        if rank == 0:
            print(f"  Size: {bytes_transferred/1e6:8.1f} MB | "
                  f"Bandwidth: {bandwidth_gbs:6.2f} GB/s | "
                  f"Time: {elapsed*1000/num_iters:6.2f} ms")
    
    if rank == 0:
        print("=" * 70 + "\n")
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main entry point for multi-node training demonstration.
    
    Launch with:
        torchrun --nnodes=2 --nproc_per_node=8 \\
                 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
                 multi_node_blackwell.py
    """
    # Setup distributed
    rank, local_rank, world_size, local_world_size = setup_blackwell_distributed()
    
    # Run bandwidth benchmark
    if rank == 0:
        print("\n[1/3] Running bandwidth benchmark...")
    bandwidth_results = benchmark_multi_node_bandwidth()
    
    # Create device mesh for hybrid parallelism
    if rank == 0:
        print("\n[2/3] Setting up hybrid parallelism...")
    
    # Use optimized 8x B200 mesh if applicable
    if local_world_size == 8 and world_size == 8:
        # Optimal configuration for 8x B200
        device_mesh = create_8xb200_device_mesh(tp_size=2, dp_size=4)
    else:
        # 2D mesh: [data_parallel, tensor_parallel]
        # Tensor parallel within nodes (via NVLink-C2C)
        # Data parallel across nodes (via InfiniBand)
        num_nodes = world_size // local_world_size
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(num_nodes, local_world_size),
            mesh_dim_names=("dp", "tp"),
        )
    
    # Create model
    model = MultiNodeTransformer(
        vocab_size=50257,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=4096,
    ).cuda()
    
    # Apply tensor parallelism (intra-node)
    model = apply_tensor_parallelism(model, device_mesh)
    
    # Apply FSDP (inter-node)
    model = apply_fsdp(model, device_mesh, mixed_precision=True)
    
    if rank == 0:
        print(" Model parallelization complete")
        print(f"  Tensor Parallel: {local_world_size}x (intra-node)")
        print(f"  Data Parallel: {num_nodes}x (inter-node)")
    
    # Create dummy dataset
    if rank == 0:
        print("\n[3/3] Starting training...")
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000, seq_len=512):
            self.size = size
            self.seq_len = seq_len
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 50257, (self.seq_len,)),
                'labels': torch.randint(0, 50257, (self.seq_len,)),
            }
    
    dataset = DummyDataset(size=100, seq_len=512)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=0,  # Set to 0 for demo
    )
    
    # Train
    stats = train_multi_node(
        model,
        dataloader,
        num_epochs=2,
        gradient_accumulation_steps=4,
        use_compile=True,
    )
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("Multi-Node Training Complete!")
        print("=" * 70)
        print(" Bandwidth benchmark: completed")
        print(" Hybrid parallelism: configured")
        print(" Training: completed")
        print(f" Average throughput: {sum(stats['throughputs'])/len(stats['throughputs'])/1e6:.2f}M tokens/s")
        print("=" * 70 + "\n")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
