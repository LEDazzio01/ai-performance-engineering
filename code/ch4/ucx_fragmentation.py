import arch_config  # noqa: F401 - Configure Blackwell optimizations
# ucx_fragmentation.py
import torch
import torch.distributed as dist
import time
import os

def log_mem(iteration):
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    print(f"[Iter {iteration:02d}] Reserved: {reserved/1e9:.3f} GB, "
          f"Allocated: {allocated/1e9:.3f} GB", flush=True)

def run(rank, world_size):
    # Check if we have enough GPUs for distributed training
    if torch.cuda.device_count() < world_size:
        print(f"Warning: Only {torch.cuda.device_count()} GPU(s) available, but {world_size} requested.", flush=True)
        print("Falling back to single-GPU simulation.", flush=True)
        # Single GPU simulation
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        
        # Pre-allocate a big buffer that UCX will register once and hold
        big_buffer = torch.empty(int(2e8), device=device)  # ~0.8 GB
        log_mem(0)
        
        for i in range(10):
            # Simulate variable-size allocations
            small_tensor = torch.randn(1000 + i * 100, 1000, device=device)
            
            # Simulate all-reduce operation
            _ = small_tensor * 2  # Simulate operation
            
            # Clear cache periodically to avoid fragmentation
            if i % 3 == 0:
                torch.cuda.empty_cache()
            
            log_mem(i + 1)
            
            # Important: del to release reference
            del small_tensor
        
        print("Single-GPU simulation completed.", flush=True)
        return
    
    # Standard DDP / UCX init
    try:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}", flush=True)
        print("Falling back to single-GPU simulation.", flush=True)
        # Single GPU simulation
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        
        # Pre-allocate a big buffer that UCX will register once and hold
        big_buffer = torch.empty(int(2e8), device=device)  # ~0.8 GB
        log_mem(0)
        
        for i in range(10):
            # Simulate variable-size allocations
            small_tensor = torch.randn(1000 + i * 100, 1000, device=device)
            
            # Simulate all-reduce operation
            _ = small_tensor * 2  # Simulate operation
            
            # Clear cache periodically to avoid fragmentation
            if i % 3 == 0:
                torch.cuda.empty_cache()
            
            log_mem(i + 1)
            
            # Important: del to release reference
            del small_tensor
        
        print("Single-GPU simulation completed.", flush=True)
        return
    
    # Pre-allocate a big buffer that UCX will register once and hold
    big_buffer = torch.empty(int(2e8), device=local_rank)  # ~0.8 GB
    log_mem(0)
    
    for i in range(10):
        # Simulate variable-size allocations
        small_tensor = torch.randn(1000 + i * 100, 1000, device=local_rank)
        
        # All-reduce to trigger UCX/RDMA registration
        dist.all_reduce(small_tensor)
        
        # Clear cache periodically to avoid fragmentation
        if i % 3 == 0:
            torch.cuda.empty_cache()
        
        log_mem(i + 1)
        
        # Important: del to release reference
        del small_tensor
    
    # Final cleanup
    dist.destroy_process_group()

def main():
    import torch.multiprocessing as mp
    world_size = min(2, torch.cuda.device_count())
    
    # Set environment variables for UCX (if using)
    os.environ.setdefault("NCCL_NET_GDR_LEVEL", "3")
    os.environ.setdefault("NCCL_IB_DISABLE", "0")
    
    if world_size == 1:
        print("Only 1 GPU available, running single-GPU simulation.", flush=True)
        run(0, 1)
    else:
        print(f"Running distributed training with {world_size} GPUs.", flush=True)
        mp.spawn(run, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main()
