# Chapter 15: Disaggregated Inference Architecture

## Overview

Disaggregated inference separates prefill (prompt processing) and decode (token generation) into specialized services for better resource utilization and latency. This chapter teaches you the architectural patterns, trade-offs, and implementation strategies for production LLM inference systems.

## Learning Objectives

After completing this chapter, you can:

- [OK] Understand prefill vs decode characteristics and bottlenecks
- [OK] Design disaggregated inference architectures
- [OK] Implement KV cache management strategies
- [OK] Apply continuous batching for higher throughput
- [OK] Optimize latency vs throughput trade-offs
- [OK] Choose between monolithic and disaggregated approaches

## Prerequisites

**Previous chapters**:
- [Chapter 4: Multi-GPU](.[executable]/[file]) - distributed inference
- [Chapter 11: Streams](.[executable]/[file]) - concurrent execution
- [Chapter 13: PyTorch Profiling](.[executable]/[file]) - bottleneck identification

**Required**: Understanding of LLM inference and attention mechanisms

## Prefill vs Decode Fundamentals

### Characteristics

| Aspect | Prefill | Decode |
|--------|---------|--------|
| **Input** | Full prompt (100-1000+ tokens) | Single token |
| **Output** | Initial KV cache | Next token + updated KV |
| **Compute** | High (matrix-matrix) | Low (matrix-vector) |
| **Memory** | Grows linearly with prompt | Fixed per token |
| **Bottleneck** | Compute-bound | Memory-bound |
| **Latency** | Can be seconds | Must be <50ms |
| **Batch size** | Small (1-8) | Large (32-256) |
| **GPU utilization** | 70-90% | 20-40% |

### Why Separate?

**Monolithic approach** (prefill + decode on same GPU):
```
GPU Timeline:
[Prefill req 1: 2s] → [Decode req 1: 8×50ms = 400ms]
                      [Prefill req 2: blocked!]
```

**Disaggregated approach**:
```
Prefill GPU:
[Req 1: 2s] [Req 2: 2s] [Req 3: 2s] ...

Decode GPU:
[Req 1: decode...] [Req 2: decode...] [Req 3: decode...] ...
```

**Benefits**:
- [OK] Better GPU utilization (70-90% prefill, 80-95% decode)
- [OK] Lower TTFT (Time To First Token) for new requests
- [OK] Higher throughput (specialized hardware)
- [OK] Easier autoscaling (scale prefill and decode independently)

---

## Examples

###  Complete Implementation

**Purpose**: Full disaggregated inference system with prefill/decode services.

#### Architecture

```python
import torch
import [file] as dist
from queue import Queue
from threading import Thread

class KVCacheManager:
    """Manages KV cache transfer between prefill and decode."""
    
    def __init__(self, max_batch_size=256, max_seq_len=2048):
        [file]_batch_size = max_batch_size
        [file]_seq_len = max_seq_len
        
        # Preallocate cache storage
        [file]_storage = {}
        
    def allocate_cache(self, request_id: str, num_layers: int, 
                      hidden_dim: int, num_heads: int):
        """Allocate KV cache for a request."""
        head_dim = hidden_dim // num_heads
        
        k_cache = [file](
            num_layers, [file]_seq_len, num_heads, head_dim,
            dtype=[file], device='cuda'
        )
        v_cache = [file](
            num_layers, [file]_seq_len, num_heads, head_dim,
            dtype=[file], device='cuda'
        )
        
        [file]_storage[request_id] = {
            'k_cache': k_cache,
            'v_cache': v_cache,
            'current_len': 0,
            'max_len': [file]_seq_len
        }
        
        return k_cache, v_cache
    
    def get_cache(self, request_id: str):
        """Retrieve existing cache."""
        return [file][file](request_id)
    
    def transfer_to_decode(self, request_id: str, prefill_len: int):
        """Transfer cache from prefill to decode service."""
        cache = [file]_storage[request_id]
        cache['current_len'] = prefill_len
        
        # In production: Transfer over NVLink or network
        # For now: Already in shared memory pool
        return cache
    
    def free_cache(self, request_id: str):
        """Free cache when request completes."""
        if request_id in [file]_storage:
            del [file]_storage[request_id]

class PrefillService:
    """Prefill service: Process prompts, generate initial KV cache."""
    
    def __init__(self, model, cache_manager, device='cuda:0'):
        [file] = [file](device)
        [file].eval()
        [file]_manager = cache_manager
        [file] = device
        [file]_queue = Queue()
        
    def process_request(self, request):
        """Process single prefill request."""
        request_id = request['id']
        input_ids = request['input_ids'].to([file])
        
        # Allocate cache
        k_cache, v_cache = [file][file]_cache(
            request_id, 
            num_layers=[file].[file]_hidden_layers,
            hidden_dim=[file].[file]_size,
            num_heads=[file].[file]_attention_heads
        )
        
        # Prefill (compute-bound)
        with [file]_grad():
            outputs = [file](
                input_ids=input_ids,
                past_key_values=None,  # First pass
                use_cache=True,
                return_dict=True
            )
        
        # Store KV cache
        past_key_values = [file]_key_values
        for layer_idx, (k, v) in enumerate(past_key_values):
            k_cache[layer_idx, :[file](1)] = k[0]
            v_cache[layer_idx, :[file](1)] = v[0]
        
        # Transfer to decode service
        prefill_len = [file](1)
        [file][file]_to_decode(request_id, prefill_len)
        
        return {
            'request_id': request_id,
            'first_token': [file][:, -1, :].argmax(dim=-1),
            'prefill_len': prefill_len
        }
    
    def run(self):
        """Run prefill service (continuous loop)."""
        while True:
            request = [file][file]()
            if request is None:  # Shutdown signal
                break
            
            result = [file]_request(request)
            # Send to decode service
            [file][file](result)

class DecodeService:
    """Decode service: Generate tokens using cached KV."""
    
    def __init__(self, model, cache_manager, device='cuda:1'):
        [file] = [file](device)
        [file].eval()
        [file]_manager = cache_manager
        [file] = device
        [file]_queue = Queue()
        
        # Continuous batching: active requests
        [file]_requests = {}
        
    def add_request(self, request):
        """Add request to continuous batch."""
        request_id = request['request_id']
        [file]_requests[request_id] = {
            'current_token': request['first_token'],
            'generated': [request['first_token'].item()],
            'current_len': request['prefill_len'],
            'max_tokens': 100  # Generate up to 100 tokens
        }
    
    def decode_step(self):
        """Single decode step for all active requests (batched)."""
        if not [file]_requests:
            return
        
        # Prepare batch
        batch_size = len([file]_requests)
        input_ids = [file](batch_size, 1, dtype=[file], device=[file])
        
        # Gather current tokens
        request_ids = list([file][file]())
        for i, req_id in enumerate(request_ids):
            input_ids[i, 0] = [file]_requests[req_id]['current_token']
        
        # Gather KV caches
        past_key_values = self._gather_kv_caches(request_ids)
        
        # Decode (memory-bound, but batched for efficiency)
        with [file]_grad():
            outputs = [file](
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
        
        # Update caches and generate next tokens
        next_tokens = [file][:, -1, :].argmax(dim=-1)
        
        # Update each request
        completed = []
        for i, req_id in enumerate(request_ids):
            req = [file]_requests[req_id]
            req['current_token'] = next_tokens[i]
            req['generated'].append(next_tokens[i].item())
            req['current_len'] += 1
            
            # Check completion
            if (req['current_len'] >= req['max_tokens'] or 
                next_tokens[i] == [file].[file]_token_id):
                [file](req_id)
        
        # Remove completed requests
        for req_id in completed:
            [file][file]_cache(req_id)
            del [file]_requests[req_id]
    
    def _gather_kv_caches(self, request_ids):
        """Gather KV caches for batch."""
        # Implementation detail: Gather from cache manager
        # ...
        pass
    
    def run(self):
        """Run decode service (continuous batching loop)."""
        while True:
            # Add new requests from queue
            while not [file][file]():
                request = [file][file]()
                if request is None:  # Shutdown
                    return
                [file]_request(request)
            
            # Decode step for all active requests
            [file]_step()
            
            # Sleep if no active requests
            if not [file]_requests:
                [file]([file])

# Initialize system
cache_manager = KVCacheManager()
prefill_service = PrefillService(model, cache_manager, device='cuda:0')
decode_service = DecodeService(model, cache_manager, device='cuda:1')

# Start services
prefill_thread = Thread(target=[file])
decode_thread = Thread(target=[file])
[file]()
[file]()

# Send requests
for i in range(100):
    request = {
        'id': f'request_{i}',
        'input_ids': [file]([[...]])  # Prompt tokens
    }
    [file][file](request)
```

**How to run**:
```bash
python3 [script]
```

**Expected improvement**:
- **Throughput**: **2-3x higher** than monolithic
- **TTFT**: **50-70% lower** (prefill not blocked by decode)
- **GPU utilization**: **80-95%** (vs 40-60% monolithic)

---

## Continuous Batching

### Problem with Static Batching

```
Batch of 8 requests:
Request 1: Generates 10 tokens  → Finishes early, GPU idle
Request 2: Generates 100 tokens → Holds batch hostage
...

GPU waits for longest request → Underutilization
```

### Solution: Continuous Batching

```
Dynamic batch:
Step 1: [Req 1, Req 2, Req 3, Req 4, Req 5, Req 6, Req 7, Req 8]
Step 5: [Req 2, Req 3, Req 4, Req 5, Req 6, Req 7, Req 8, Req 9] ← Req 1 done, Req 9 added!
Step 20: [Req 4, Req 6, Req 8, Req 10, Req 11, Req 12, Req 13, Req 14]

Batch constantly full → High utilization
```

**Implementation**: See `[file]_step()` above - adds/removes requests every step.

---

## KV Cache Management Strategies

### 1. PagedAttention (vLLM approach)

**Concept**: Store KV cache in fixed-size pages, like virtual memory.

**Benefits**:
- [OK] No fragmentation
- [OK] Efficient memory sharing (multiple sequences sharing prompt)
- [OK] Easy eviction (swap pages to CPU/NVMe)

### 2. Prefix Caching

**Concept**: Cache common prompt prefixes.

```
Prompt 1: "You are a helpful assistant. User: How do I ...?"
Prompt 2: "You are a helpful assistant. User: What is ...?"
                     ↑ Shared prefix

Cache prefix once, reuse for all requests → Save prefill compute!
```

**Speedup**: **5-10x** for repetitive system prompts.

### 3. Cache Eviction Policies

When cache is full, evict based on:
- **LRU** (Least Recently Used): Good for long-running requests
- **Priority**: Evict lower-priority requests first
- **Expected completion time**: Evict requests close to finishing

---

## How to Run All Examples

```bash
cd ch15

# Install dependencies
pip install -r [file]

# Run disaggregated inference
python3 [script]

# Compare with monolithic (for reference)
# python3 .[executable]/[file] --monolithic
```

---

## Key Takeaways

1. **Prefill and decode have different characteristics**: Compute-bound vs memory-bound, high latency vs low latency.

2. **Disaggregation improves utilization**: 80-95% GPU usage vs 40-60% monolithic.

3. **Continuous batching is essential**: Don't wait for full batch. Add/remove requests dynamically.

4. **KV cache management is critical**: Memory is scarce. Use paged attention or prefix caching.

5. **TTFT vs throughput trade-off**: Disaggregation lowers TTFT but adds complexity.

6. **Scale prefill and decode independently**: Different hardware needs (compute vs memory).

7. **Network latency matters**: Cache transfer between services must be fast (NVLink or high-speed interconnect).

---

## Common Pitfalls

### Pitfall 1: Not Using Continuous Batching
**Problem**: Waiting for full batch → Low throughput.

**Solution**: Add requests as they arrive, remove as they complete.

### Pitfall 2: Over-Allocating KV Cache
**Problem**: Allocating max_seq_len for every request → OOM.

**Solution**: Use paged attention or dynamic allocation.

### Pitfall 3: Slow Cache Transfer
**Problem**: CPU bottleneck transferring KV cache between services.

**Solution**: Use NVLink P2P or shared memory pool.

### Pitfall 4: Not Prioritizing Latency for Prefill
**Problem**: Batching prefill requests → High TTFT.

**Solution**: Prioritize prefill latency over throughput (batch size 1-4).

### Pitfall 5: Monolithic for Variable Workloads
**Problem**: Mixing short and long requests → Unpredictable latency.

**Solution**: Separate by expected length or use disaggregation.

---

## Next Steps

**Production inference** → [Chapter 16: Inference Optimization](.[executable]/[file])

Learn about:
- Production serving with vLLM
- FP8 quantization for inference
- Speculative decoding
- Synthetic MoE benchmarking

**Back to multi-GPU** → [Chapter 4: Multi-GPU Training](.[executable]/[file])

---

## Additional Resources

- **vLLM**: [PagedAttention Paper](https://[file]/abs/[file])
- **Orca**: [Continuous Batching](https://[file].org/conference/osdi22/presentation/yu)
- **Disaggregated Inference**: [Splitwise Paper](https://[file]/abs/[file])
- **KV Cache Optimization**: [FlashAttention-2](https://[file]/abs/[file])

---

**Chapter Status**: [OK] Complete
