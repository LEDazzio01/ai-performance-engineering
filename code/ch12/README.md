# Chapter 12: CUDA Graphs and Dynamic Parallelism

## Overview

CUDA Graphs eliminate kernel launch overhead by capturing and replaying entire workflows, while dynamic parallelism enables kernels to launch other kernels. This chapter teaches you how to use these advanced features for ultra-low latency execution and adaptive workloads.

## Learning Objectives

After completing this chapter, you can:

- ✅ Capture and replay CUDA graphs for repeatable workloads
- ✅ Use conditional graphs for dynamic execution paths
- ✅ Implement dynamic parallelism for adaptive algorithms
- ✅ Apply graph instantiation for parameter updates
- ✅ Optimize launch overhead from 5-20 μs to <1 μs
- ✅ Choose between graphs, streams, and dynamic parallelism

## Prerequisites

**Previous chapters**:
- [Chapter 11: CUDA Streams](../ch11/README.md) - async execution model
- [Chapter 6: CUDA Basics](../ch6/README.md) - kernel launches

**Required**: Understanding of kernel launch mechanics and async operations

## Why CUDA Graphs?

### The Launch Overhead Problem

**Traditional kernel launch**:
```cpp
for (int i = 0; i < 1000; i++) {
    small_kernel<<<1, 32>>>(data);  // Each launch: 5-20 μs overhead
}
// Total overhead: 5-20 ms!
```

**With CUDA Graphs**:
```cpp
// Capture once
cudaGraph_t graph;
cudaStreamBeginCapture(stream);
for (int i = 0; i < 1000; i++) {
    small_kernel<<<1, 32, 0, stream>>>(data);
}
cudaStreamEndCapture(stream, &graph);

// Instantiate
cudaGraphExec_t instance;
cudaGraphInstantiate(&instance, graph);

// Replay (single launch, <1 μs overhead!)
cudaGraphLaunch(instance, stream);
// Total overhead: <1 ms
```

**Speedup**: **10-20x** for workloads with many small kernels!

---

## Examples

### 1. `cuda_graphs.cu` - Basic Graph Capture and Replay

**Purpose**: Demonstrate fundamental graph operations.

#### Manual Graph Creation

```cpp
// Create graph manually
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// Add nodes
cudaGraphNode_t memcpy_node, kernel_node;

// H2D copy node
cudaMemcpy3DParms memcpy_params = {0};
// ... configure params ...
cudaGraphAddMemcpyNode(&memcpy_node, graph, NULL, 0, &memcpy_params);

// Kernel node
cudaKernelNodeParams kernel_params = {0};
kernel_params.func = (void*)my_kernel;
kernel_params.gridDim = dim3(blocks);
kernel_params.blockDim = dim3(threads);
kernel_params.kernelParams = args;
cudaGraphAddKernelNode(&kernel_node, graph, &memcpy_node, 1, &kernel_params);

// Instantiate and launch
cudaGraphExec_t instance;
cudaGraphInstantiate(&instance, graph);
cudaGraphLaunch(instance, stream);
```

#### Stream Capture (Easier!)

```cpp
cudaGraph_t graph;
cudaGraphExec_t instance;
cudaStream_t stream;

cudaStreamCreate(&stream);

// Begin capture
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// Execute operations (captured, not run yet)
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
kernel<<<blocks, threads, 0, stream>>>(d_data);
cudaMemcpyAsync(h_result, d_result, size, cudaMemcpyDeviceToHost, stream);

// End capture
cudaStreamEndCapture(stream, &graph);

// Instantiate
cudaGraphInstantiate(&instance, graph);

// Replay many times (each replay <1 μs!)
for (int i = 0; i < 1000; i++) {
    cudaGraphLaunch(instance, stream);
}
```

**How to run**:
```bash
make cuda_graphs
./cuda_graphs_sm100
```

**Expected output**:
```
Traditional launches (1000 kernels): 18.2 ms
CUDA Graph replay (1000 kernels): 1.1 ms
Speedup: 16.5x ✅
```

---

### 2. `cuda_graphs_conditional.cu` - Conditional Graphs

**Purpose**: Execute different paths based on runtime conditions.

**Conditional nodes** (CUDA 12.4+):

```cpp
__global__ void evaluate_condition(int* input, int* output) {
    *output = (*input > 100) ? 1 : 0;
}

// Create graph with conditional
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// Condition evaluation node
cudaGraphNode_t cond_node;
// ... add evaluate_condition kernel ...

// True branch
cudaGraph_t true_graph;
cudaGraphCreate(&true_graph, 0);
// ... add kernels for true path ...

// False branch
cudaGraph_t false_graph;
cudaGraphCreate(&false_graph, 0);
// ... add kernels for false path ...

// Conditional node
cudaGraphConditionalHandle handle;
cudaGraphNodeParams params;
params.type = cudaGraphNodeTypeConditional;
params.conditional.handle = handle;
params.conditional.ctx = ...;
params.conditional.size = 1;
cudaGraphAddNode(&cond_node, graph, ...);

// Add conditional branches
cudaGraphConditionalHandleSetGraph(handle, true_graph, 0);
cudaGraphConditionalHandleSetGraph(handle, false_graph, 1);
```

**Use cases**:
- Early exit in iterative algorithms
- Adaptive batch sizes
- Dynamic routing in inference

**How to run**:
```bash
make cuda_graphs_conditional
./cuda_graphs_conditional_sm100
```

---

### 3. `dynamic_parallelism.cu` - Device-Side Kernel Launches

**Purpose**: Enable kernels to launch other kernels, adapting to data.

**Why dynamic parallelism?**
- ✅ Adaptive algorithms (tree traversal, AMR, sorting)
- ✅ Data-dependent parallelism
- ✅ Recursive patterns
- ❌ Not for regular workloads (overhead!)

**Example: Adaptive Tree Traversal**

```cpp
__global__ void traverse_tree_adaptive(TreeNode* node, int depth) {
    if (node == NULL) return;
    
    // Process current node
    process_node(node);
    
    // Adaptively decide whether to recurse
    if (needs_refinement(node) && depth < MAX_DEPTH) {
        // Launch child kernels from device!
        traverse_tree_adaptive<<<1, 32>>>(node->left, depth + 1);
        traverse_tree_adaptive<<<1, 32>>>(node->right, depth + 1);
        
        // Wait for children
        cudaDeviceSynchronize();
    }
}

// Host launches root
traverse_tree_adaptive<<<1, 32>>>(root, 0);
```

**Requirements**:
- Compile with `--relocatable-device-code=true`
- Link with `-lcudadevrt`
- Use `cudaDeviceSynchronize()` to wait for child kernels

**How to run**:
```bash
make dynamic_parallelism
./dynamic_parallelism_sm100
```

---

### 4. `dp_device_launched.cu` vs `dp_host_launched.cu` - Comparison

#### Host-Launched (Traditional)

```cpp
void traverse_tree_host(TreeNode* node) {
    if (node == NULL) return;
    
    // Launch kernel for this node
    process_node_kernel<<<1, 32>>>(node);
    cudaDeviceSynchronize();  // Wait
    
    // Host decides next steps
    if (needs_refinement_host(node)) {
        traverse_tree_host(node->left);   // Recursive host call
        traverse_tree_host(node->right);
    }
}
```

**Problem**: CPU-GPU round-trip for each decision → High latency!

#### Device-Launched (Dynamic Parallelism)

```cpp
__global__ void traverse_tree_device(TreeNode* node, int depth) {
    process_node(node);
    
    // Device decides and launches (no CPU involvement!)
    if (needs_refinement_device(node) && depth < MAX_DEPTH) {
        traverse_tree_device<<<1, 32>>>(node->left, depth + 1);
        traverse_tree_device<<<1, 32>>>(node->right, depth + 1);
    }
}
```

**Benefit**: No CPU round-trips → **10-100x lower latency** for adaptive algorithms.

**How to run**:
```bash
make dp_device_launched dp_host_launched
./dp_host_launched_sm100      # Baseline (CPU involvement)
./dp_device_launched_sm100    # Optimized (GPU-only)
```

---

### 5. `uneven_static.cu` → `uneven_dynamic.cu` - Load Balancing

#### Static Workload (Baseline)

```cpp
__global__ void process_static(int* work, int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Fixed work per thread
        output[idx] = expensive_computation(work[idx]);
    }
}
```

**Problem**: If `work[i]` varies greatly, some threads finish early, others take long → Load imbalance!

#### Dynamic Workload (with Dynamic Parallelism)

```cpp
__global__ void process_dynamic(WorkQueue* queue, int* output) {
    // Pull work from queue dynamically
    while (true) {
        int work_item = queue->dequeue();
        if (work_item == -1) break;
        
        // Expensive item? Launch more threads!
        if (is_expensive(work_item)) {
            parallel_process<<<4, 32>>>(work_item, output);
            cudaDeviceSynchronize();
        } else {
            output[work_item] = simple_process(work_item);
        }
    }
}
```

**Benefit**: Automatic load balancing → **2-3x better utilization** for uneven workloads.

**How to run**:
```bash
make uneven_static uneven_dynamic
./uneven_static_sm100
./uneven_dynamic_sm100
```

---

### 6. `atomic_work_queue.cu` - Device-Side Work Queue

**Purpose**: Implement work-stealing queue for dynamic parallelism.

```cpp
struct WorkQueue {
    int* items;
    int* head;
    int* tail;
    int capacity;
    
    __device__ int dequeue() {
        int idx = atomicAdd(head, 1);
        if (idx >= *tail) return -1;
        return items[idx];
    }
    
    __device__ void enqueue(int item) {
        int idx = atomicAdd(tail, 1);
        items[idx] = item;
    }
};

__global__ void work_stealing_kernel(WorkQueue* queue, int* output) {
    int tid = threadIdx.x;
    
    while (true) {
        int work = queue->dequeue();
        if (work == -1) break;
        
        // Process work
        int result = process(work);
        output[work] = result;
        
        // Generate more work?
        if (generates_more_work(result)) {
            queue->enqueue(new_work_item);
        }
    }
}
```

**Use cases**: Breadth-first search, dynamic task graphs, irregular computations.

**How to run**:
```bash
make atomic_work_queue
./atomic_work_queue_sm100
```

---

## Graph Instantiation and Updates

### Updating Graph Parameters

**Problem**: Recreating graphs is expensive. Update parameters instead!

```cpp
cudaGraphExec_t instance;
cudaGraphInstantiate(&instance, graph);

// First launch
cudaGraphLaunch(instance, stream);

// Update kernel parameters (no re-instantiation!)
cudaKernelNodeParams new_params;
new_params.kernelParams[0] = &new_data;  // New pointer
cudaGraphExecKernelNodeSetParams(instance, kernel_node, &new_params);

// Launch with updated parameters
cudaGraphLaunch(instance, stream);
```

**Speedup**: **100x faster** than recreating graph.

### When Graphs Can't Be Updated

Some changes require re-instantiation:
- ❌ Changing topology (adding/removing nodes)
- ❌ Changing kernel function
- ❌ Changing grid/block dimensions
- ✅ Updating kernel parameters (OK!)
- ✅ Updating memcpy addresses (OK!)

---

## When to Use What

| Feature | Use When | Launch Overhead | Flexibility |
|---------|----------|-----------------|-------------|
| **Traditional Launches** | One-off operations | 5-20 μs | Maximum |
| **CUDA Streams** | Independent concurrent ops | 5-20 μs | High |
| **CUDA Graphs** | Repeatable workflows | <1 μs | Medium |
| **Dynamic Parallelism** | Data-dependent algorithms | 10-30 μs | Maximum |

### Decision Tree

```
Does workload repeat exactly?
├─ Yes → Use CUDA Graphs
└─ No → Is it data-dependent?
    ├─ Yes → Use Dynamic Parallelism
    └─ No → Are operations independent?
        ├─ Yes → Use CUDA Streams
        └─ No → Traditional launches
```

---

## How to Run All Examples

```bash
cd ch12

# Build all examples
make

# Basic graphs
./cuda_graphs_sm100                    # Capture and replay
./cuda_graphs_conditional_sm100        # Conditional execution

# Dynamic parallelism
./dynamic_parallelism_sm100            # Recursive kernels
./dp_device_launched_sm100             # Device-side launch
./dp_host_launched_sm100               # Host-side launch (baseline)

# Load balancing
./uneven_static_sm100                  # Static workload
./uneven_dynamic_sm100                 # Dynamic adaptation
./atomic_work_queue_sm100              # Work stealing

# Profile to see launch overhead reduction
../../common/profiling/profile_cuda.sh ./cuda_graphs baseline
```

---

## Key Takeaways

1. **Graphs eliminate launch overhead**: <1 μs vs 5-20 μs → 10-20x faster for many small kernels.

2. **Stream capture is easiest**: Use `cudaStreamBeginCapture` instead of manual graph construction.

3. **Graphs are for repeatable workloads**: If workflow changes every iteration, graphs won't help.

4. **Update parameters, don't recreate**: `cudaGraphExecKernelNodeSetParams` is 100x faster than re-instantiation.

5. **Dynamic parallelism for adaptivity**: When data determines parallelism structure, device-side launches eliminate CPU round-trips.

6. **Dynamic parallelism has overhead**: 10-30 μs per device-side launch. Only use for irregular/adaptive workloads.

7. **Profile to validate**: Measure actual launch overhead reduction with Nsight Systems.

---

## Common Pitfalls

### Pitfall 1: Using Graphs for Dynamic Workloads
**Problem**: Workflow changes every iteration → Can't capture as graph.

**Solution**: Use streams or dynamic parallelism instead.

### Pitfall 2: Recreating Graphs Instead of Updating
**Problem**: `cudaGraphInstantiate` on every iteration → Loses benefit.

**Solution**: Update parameters with `cudaGraphExecKernelNodeSetParams`.

### Pitfall 3: Excessive Device-Side Launches
**Problem**: Launching 1000 child kernels → 10-30 ms overhead!

**Solution**: Batch work when possible. Only use dynamic parallelism when truly adaptive.

### Pitfall 4: Not Checking Graph Compatibility
**Problem**: Attempting to update incompatible parameters → Silent failure or crash.

**Solution**: Check `cudaGraphExecUpdate` return value.

### Pitfall 5: Memory Dependencies in Captured Graphs
**Problem**: Using `cudaMalloc` during capture → Undefined behavior.

**Solution**: Allocate memory before capture, or use memory nodes in manual graph.

---

## Next Steps

**PyTorch profiling and optimization** → [Chapter 13: PyTorch Profiling](../ch13/README.md)

Learn about:
- PyTorch profiler for bottleneck identification
- Memory profiling and optimization
- FSDP (Fully Sharded Data Parallel)
- Custom autograd functions

**Back to streams** → [Chapter 11: CUDA Streams](../ch11/README.md)

---

## Additional Resources

- **CUDA Graphs**: [Programming Guide - Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- **Dynamic Parallelism**: [Programming Guide - Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism)
- **Conditional Graphs**: [CUDA 12.4+ Feature](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#conditional-graph-nodes)
- **Graph Best Practices**: [CUDA Best Practices - Graphs](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#cuda-graphs)

---

**Chapter Status**: ✅ Complete  
**Last Updated**: November 3, 2025  
**Tested On**: NVIDIA B200 GPU, CUDA 13.0, sm_100 architecture

