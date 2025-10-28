import arch_config  # noqa: F401 - Configure Blackwell optimizations
# after_reinit_comm.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os

def run(rank, world_size):
    # Check if we have enough GPUs for distributed training
    if torch.cuda.device_count() < world_size:
        print(f"Warning: Only {torch.cuda.device_count()} GPU(s) available, but {world_size} requested.", flush=True)
        print("Falling back to single-GPU simulation.", flush=True)
        # Single GPU simulation
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Simulate one-time initialization
        start_init = time.time()
        # Simulate initialization work
        time.sleep(0.001)  # Simulate initialization time
        init_time = time.time() - start_init
        print(f"One-time initialization took {init_time*1000:.2f} ms", flush=True)
        
        # Now run iterations with the same "communicator"
        for i in range(5):
            start_iter = time.time()
            # Simulate some work
            tensor = torch.ones(1, device=device)
            _ = tensor * 2  # Simulate operation
            iter_time = time.time() - start_iter
            print(f"Iter {i} done (simulation took {iter_time*1000:.2f} ms)", flush=True)
        return
    
    # Pin GPU
    torch.cuda.set_device(rank)
    
    # Initialize NCCL communicator once
    start_init = time.time()
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:45678",
            world_size=world_size,
            rank=rank
        )
        init_time = time.time() - start_init
        
        if rank == 0:
            print(f"One-time initialization took {init_time*1000:.2f} ms", flush=True)
        
        # Now run iterations with the same communicator
        for i in range(5):
            start_iter = time.time()
            
            # do a tiny all-reduce to simulate some work
            tensor = torch.ones(1).cuda(rank)
            dist.all_reduce(tensor)
            
            iter_time = time.time() - start_iter
            
            if rank == 0:
                print(f"Iter {i} done (all-reduce took {iter_time*1000:.2f} ms)", flush=True)
        
        # Clean up once at the end
        dist.destroy_process_group()
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}", flush=True)
        print("Falling back to single-GPU simulation.", flush=True)
        # Single GPU simulation
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Simulate one-time initialization
        start_init = time.time()
        time.sleep(0.001)  # Simulate initialization time
        init_time = time.time() - start_init
        print(f"One-time initialization took {init_time*1000:.2f} ms", flush=True)
        
        # Now run iterations with the same "communicator"
        for i in range(5):
            start_iter = time.time()
            # Simulate some work
            tensor = torch.ones(1, device=device)
            _ = tensor * 2  # Simulate operation
            iter_time = time.time() - start_iter
            print(f"Iter {i} done (simulation took {iter_time*1000:.2f} ms)", flush=True)

def main():
    world_size = min(2, torch.cuda.device_count())
    if world_size == 1:
        print("Only 1 GPU available, running single-GPU simulation.", flush=True)
        run(0, 1)
    else:
        print(f"Running distributed training with {world_size} GPUs.", flush=True)
        mp.spawn(run, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main()
