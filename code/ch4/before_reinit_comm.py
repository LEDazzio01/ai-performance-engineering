import arch_config  # noqa: F401 - Configure Blackwell optimizations
# before_reinit_comm.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def run(rank, world_size):
    # Check if we have enough GPUs for distributed training
    if torch.cuda.device_count() < world_size:
        print(f"Warning: Only {torch.cuda.device_count()} GPU(s) available, but {world_size} requested.", flush=True)
        print("Falling back to single-GPU simulation.", flush=True)
        # Single GPU simulation
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i in range(5):  # simulate 5 iterations
            start_iter = time.time()
            # Simulate some work
            tensor = torch.ones(1, device=device)
            _ = tensor * 2  # Simulate operation
            iter_time = time.time() - start_iter
            print(f"Iter {i} done (simulation took {iter_time*1000:.2f} ms)", flush=True)
        return
    
    torch.cuda.set_device(rank)
    
    for i in range(5):  # simulate 5 iterations
        # This naive approach re-initializes NCCL each
        # iteration. THIS IS EXTREMELY SLOW AND NOT
        # RECOMMENDED!!!
        start_init = time.time()
        try:
            dist.init_process_group("nccl", init_method="tcp://127.0.0.1:45678",
                                   world_size=world_size, rank=rank)
            init_time = time.time() - start_init
            
            # do a tiny all-reduce to simulate some work
            tensor = torch.ones(1).cuda(rank)
            dist.all_reduce(tensor)
            
            if rank == 0:
                print(f"Iter {i} done (init took {init_time*1000:.2f} ms)", flush=True)
            
            dist.destroy_process_group()
        except Exception as e:
            print(f"Failed to initialize process group on iteration {i}: {e}", flush=True)
            break

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
