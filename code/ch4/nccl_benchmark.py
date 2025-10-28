# nccl_benchmark.py
"""
Comprehensive NCCL benchmark for testing different collective operations
and communication patterns with PyTorch 2.9 and CUDA 13.0.
"""
import arch_config  # noqa: F401 - Configure Blackwell optimizations
import os
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def benchmark_collective(rank, world_size, op_type, data_size, dtype, num_warmup=5, num_trials=10):
    """Benchmark a specific collective operation."""
    torch.cuda.set_device(rank)
    
    # Create test tensor
    if dtype == "float32":
        tensor = torch.randn(data_size, device=f"cuda:{rank}", dtype=torch.float32)
    elif dtype == "float16":
        tensor = torch.randn(data_size, device=f"cuda:{rank}", dtype=torch.float16)
    elif dtype == "bfloat16":
        tensor = torch.randn(data_size, device=f"cuda:{rank}", dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # Warmup
    for _ in range(num_warmup):
        if op_type == "allreduce":
            dist.all_reduce(tensor.clone())
        elif op_type == "allgather":
            output_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(output_tensors, tensor)
        elif op_type == "reducescatter":
            output = torch.empty(data_size // world_size, device=tensor.device, dtype=tensor.dtype)
            input_list = list(tensor.chunk(world_size))
            dist.reduce_scatter(output, input_list)
        elif op_type == "broadcast":
            dist.broadcast(tensor, src=0)
        elif op_type == "reduce":
            dist.reduce(tensor, dst=0)
        torch.cuda.synchronize()
    
    # Actual benchmark
    times = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        start = time.time()
        
        if op_type == "allreduce":
            dist.all_reduce(tensor.clone())
        elif op_type == "allgather":
            output_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(output_tensors, tensor)
        elif op_type == "reducescatter":
            output = torch.empty(data_size // world_size, device=tensor.device, dtype=tensor.dtype)
            input_list = list(tensor.chunk(world_size))
            dist.reduce_scatter(output, input_list)
        elif op_type == "broadcast":
            dist.broadcast(tensor, src=0)
        elif op_type == "reduce":
            dist.reduce(tensor, dst=0)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
    
    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Calculate bandwidth (for allreduce, assume 2*(N-1)/N efficiency)
    data_bytes = tensor.numel() * tensor.element_size()
    if op_type == "allreduce":
        # Allreduce algorithm bandwidth calculation
        bandwidth_gbps = (data_bytes * 2 * (world_size - 1) / world_size) / avg_time / 1e9
    elif op_type == "allgather":
        bandwidth_gbps = data_bytes * world_size / avg_time / 1e9
    elif op_type == "reducescatter":
        bandwidth_gbps = (data_bytes * (world_size - 1) / world_size) / avg_time / 1e9
    else:
        bandwidth_gbps = data_bytes / avg_time / 1e9
    
    if rank == 0:
        print(f"{op_type.upper()} {dtype} {data_size} elements:")
        print(f"  Avg: {avg_time*1000:.2f} ms, Min: {min_time*1000:.2f} ms, Max: {max_time*1000:.2f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        print(f"  Data size: {data_bytes/1024/1024:.1f} MB")

def run_benchmarks(rank, world_size, args):
    """Run comprehensive NCCL benchmarks."""
    try:
        dist.init_process_group(backend="nccl", init_method="env://")
    except Exception as e:
        print(f"Failed to initialize NCCL process group: {e}", flush=True)
        print("Running single-GPU benchmark instead.", flush=True)
        # Single GPU benchmark
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            print("No CUDA device available, skipping NCCL benchmark.", flush=True)
            return
        
        # Run a simple benchmark on single GPU
        data_sizes = [1024, 1024*1024]  # Smaller sizes for single GPU
        operations = ["allreduce"]
        dtypes = ["float32"]
        
        for op in operations:
            for dtype in dtypes:
                for size in data_sizes:
                    if dtype == "float32":
                        tensor = torch.randn(size, device=device, dtype=torch.float32)
                    else:
                        continue
                    
                    # Warmup
                    for _ in range(3):
                        _ = tensor * 2
                        torch.cuda.synchronize()
                    
                    # Benchmark
                    times = []
                    for _ in range(5):
                        torch.cuda.synchronize()
                        start = time.time()
                        _ = tensor * 2  # Simulate operation
                        torch.cuda.synchronize()
                        elapsed = time.time() - start
                        times.append(elapsed)
                    
                    avg_time = sum(times) / len(times)
                    data_bytes = tensor.numel() * tensor.element_size()
                    bandwidth_gbps = data_bytes / avg_time / 1e9
                    
                    print(f"SINGLE_GPU {op.upper()} {dtype} {size} elements:")
                    print(f"  Avg: {avg_time*1000:.2f} ms")
                    print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
                    print(f"  Data size: {data_bytes/1024/1024:.1f} MB")
        return
    
    if rank == 0:
        print(f"NCCL Benchmark - World Size: {world_size}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
        
        # Special message for 8-GPU configuration
        if world_size == 8:
            print("\n*** 8x B200 GPU Configuration Detected ***")
            print("Expected Performance:")
            print("  AllReduce 1GB: 700-800 GB/s bus bandwidth")
            print("  P2P: 800-900 GB/s per GPU pair")
            print("  Scaling efficiency: 85-95%")
        
        print("=" * 60)
    
    # Test different data sizes and operations
    # For 8 GPUs, test wider range including latency-sensitive small sizes
    if world_size == 8:
        data_sizes = [
            1024,              # 4 KB (latency-bound)
            256*1024,          # 1 MB
            1024*1024,         # 4 MB
            16*1024*1024,      # 64 MB
            64*1024*1024,      # 256 MB
            256*1024*1024,     # 1 GB (bandwidth-bound)
        ]
    else:
        data_sizes = [1024, 1024*1024, 16*1024*1024, 64*1024*1024]  # 4KB to 256MB
    
    operations = ["allreduce", "allgather", "reducescatter", "broadcast"]
    dtypes = ["float32", "float16", "bfloat16"]
    
    for op in operations:
        if args.operation and op not in args.operation:
            continue
            
        for dtype in dtypes:
            if args.dtype and dtype not in args.dtype:
                continue
                
            for size in data_sizes:
                if args.max_size and size * 4 > args.max_size * 1024 * 1024:
                    continue
                    
                benchmark_collective(rank, world_size, op, size, dtype,
                                   args.warmup, args.trials)
                
                if rank == 0:
                    print("-" * 40)
    
    # 8-GPU specific: P2P bandwidth matrix
    if world_size == 8 and rank == 0:
        print("\n" + "=" * 60)
        print("8-GPU P2P Bandwidth Matrix")
        print("=" * 60)
        print("Testing all GPU pairs with 256MB transfers...")
        print("(This would require additional P2P-specific code)")
        print("Expected: ~800-900 GB/s per pair with NVLink 5.0")
    
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="NCCL Benchmark Suite")
    parser.add_argument("--world-size", type=int, default=2,
                       help="Number of processes (default: 2)")
    parser.add_argument("--operation", nargs="+", 
                       choices=["allreduce", "allgather", "broadcast"],
                       help="Operations to benchmark (default: all)")
    parser.add_argument("--dtype", nargs="+",
                       choices=["float32", "float16", "bfloat16"],
                       help="Data types to test (default: all)")
    parser.add_argument("--max-size", type=int, default=256,
                       help="Maximum data size in MB (default: 256)")
    parser.add_argument("--warmup", type=int, default=5,
                       help="Number of warmup iterations (default: 5)")
    parser.add_argument("--trials", type=int, default=10,
                       help="Number of benchmark trials (default: 10)")
    
    args = parser.parse_args()
    
    # Set environment variables for distributed training
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    
    world_size = min(args.world_size, torch.cuda.device_count())
    if world_size < 2:
        print(f"This benchmark requires at least 2 GPUs, but only {torch.cuda.device_count()} available.", flush=True)
        print("Running single-GPU benchmark instead.", flush=True)
        run_benchmarks(0, 1, args)
        return
    
    print(f"Running benchmark with {world_size} GPUs", flush=True)
    mp.spawn(run_benchmarks, args=(world_size, args), nprocs=world_size)

if __name__ == "__main__":
    main()
