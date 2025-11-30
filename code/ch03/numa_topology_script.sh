#!/bin/bash

# NUMA topology binding script
# Dynamically queries the topology using nvidia-smi topo and binds processes to GPUs using the local NUMA node

for GPU in 0 1 2 3; do
    # Query NUMA node for this GPU
    NODE=$(nvidia-smi topo -m -i $GPU \
          | awk '/NUMA Affinity/ {print $NF}')
    
    # Launch the training process pinned to that NUMA node
    numactl --cpunodebind=$NODE --membind=$NODE \
            bash -c "CUDA_VISIBLE_DEVICES=$GPU python train.py --gpu $GPU" &
done

# Wait for all background processes to complete
wait
