import arch_config  # noqa: F401 - Configure Blackwell optimizations
# barrier_straggler.py
import os
import torch
import torch.distributed as dist
import time

def run(rank, world_size):
    try:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        
        # Simulate some work that might vary across ranks
        if rank == 1:
            # Simulate a straggler by adding extra delay
            time.sleep(2.0)
        
        # ... your forward/backward work here ...
        # For demo purposes, just do a simple operation
        dummy_tensor = torch.randn(1000, 1000, device=f"cuda:{local_rank}")
        result = torch.mm(dummy_tensor, dummy_tensor.t())
        
        # Before syncing at end of iteration, use a monitored barrier:
        try:
            # Wait up to 30 seconds for all ranks—if one lags, you'll get a timeout on that rank
            start_time = time.time()
            dist.monitored_barrier(timeout=30.0)
            barrier_time = time.time() - start_time
            print(f"Rank {rank} completed barrier in {barrier_time:.2f}s", flush=True)
        except RuntimeError as e:
            print(f"Rank {rank} timed out at barrier: {e}", flush=True)
        
        # Now proceed knowing all ranks are roughly in sync
        dist.destroy_process_group()
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}", flush=True)
        print("Running single-GPU simulation instead.", flush=True)
        # Single GPU simulation
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dummy_tensor = torch.randn(1000, 1000, device=device)
        result = torch.mm(dummy_tensor, dummy_tensor.t())
        print(f"Single-GPU simulation completed. Result shape: {result.shape}", flush=True)

def main():
    import torch.multiprocessing as mp
    world_size = min(2, torch.cuda.device_count())
    
    if world_size == 1:
        print("Only 1 GPU available, running single-GPU simulation.", flush=True)
        run(0, 1)
    else:
        print(f"Running distributed training with {world_size} GPUs.", flush=True)
        mp.spawn(run, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main()
