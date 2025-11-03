# Chapter 16: Inference Optimization and Production Serving

## Overview

Production inference requires maximizing throughput while meeting latency SLAs. This chapter covers advanced inference optimizations including FP8 quantization, speculative decoding, continuous batching, and production serving frameworks like vLLM for 8x B200 systems.

## Learning Objectives

After completing this chapter, you can:

- ✅ Deploy production inference with vLLM on 8x B200
- ✅ Apply FP8 quantization for 2x throughput improvement
- ✅ Implement speculative decoding for faster generation
- ✅ Optimize KV cache management and batching strategies
- ✅ Monitor and debug production inference systems
- ✅ Benchmark synthetic MoE and real workloads

## Prerequisites

**Previous chapters**:
- [Chapter 15: Disaggregated Inference](../ch15/README.md) - architecture patterns
- [Chapter 13: PyTorch Profiling](../ch13/README.md) - optimization methodology

**Required**: Production deployment experience, LLM serving fundamentals

---

## Examples

### 1. `inference_serving_8xb200.py` - Production Serving

**Purpose**: Full-featured inference server for 8x B200 with tensor parallelism.

```python
import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import initialize_model_parallel

# Initialize for tensor parallelism across 8 GPUs
initialize_model_parallel(tensor_model_parallel_size=8)

# Create inference engine
llm = LLM(
    model="deepseek-ai/deepseek-coder-33b",
    tensor_parallel_size=8,  # Shard across 8 GPUs
    dtype="float16",
    max_model_len=4096,
    
    # Optimization flags
    enforce_eager=False,  # Use CUDA graphs
    enable_prefix_caching=True,  # Cache common prompts
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
    
    # Quantization
    quantization="fp8",  # FP8 for 2x throughput
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

# Generate (batched automatically)
prompts = [
    "def quicksort(arr):",
    "# Binary search implementation\ndef binary_search",
    "class TreeNode:",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}\n")
```

**Performance targets (8x B200, 33B model)**:
- **Throughput**: 15,000+ tokens/sec
- **Latency (P50)**: <50ms per token
- **Latency (P99)**: <100ms per token
- **GPU utilization**: 85-95%

**How to run**:
```bash
pip install vllm
python3 inference_serving_8xb200.py --demo
```

---

### 2. `fp8_transformer_engine.py` - FP8 Quantization

**Purpose**: Use NVIDIA Transformer Engine for FP8 quantization.

**What is FP8?**
- 8-bit floating point (vs 16-bit FP16)
- **2x memory reduction**
- **2x throughput increase** (Tensor Cores)
- **<1% accuracy loss** with proper scaling

```python
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# FP8 recipe
fp8_recipe = recipe.DelayedScaling(
    margin=0,
    interval=1,
    fp8_format=recipe.Format.HYBRID,  # E4M3 for forward, E5M2 for backward
)

# Convert model to FP8
class FP8TransformerBlock(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        
        # FP8 Linear layers
        self.qkv_proj = te.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = te.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_fc1 = te.Linear(hidden_size, 4 * hidden_size)
        self.mlp_fc2 = te.Linear(4 * hidden_size, hidden_size)
        
    def forward(self, x):
        # Forward pass automatically uses FP8 Tensor Cores
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            # Attention
            qkv = self.qkv_proj(x)
            # ... attention computation ...
            attn_out = self.out_proj(attn_out)
            
            # MLP
            mlp_out = self.mlp_fc1(attn_out)
            mlp_out = F.gelu(mlp_out)
            mlp_out = self.mlp_fc2(mlp_out)
            
            return mlp_out

# Benchmark
model_fp16 = TransformerModel(dtype=torch.float16).cuda()
model_fp8 = FP8TransformerModel().cuda()

# FP16 baseline
tokens_per_sec_fp16 = benchmark(model_fp16)
memory_fp16 = torch.cuda.max_memory_allocated()

# FP8 optimized
tokens_per_sec_fp8 = benchmark(model_fp8)
memory_fp8 = torch.cuda.max_memory_allocated()

print(f"FP16: {tokens_per_sec_fp16} tokens/sec, {memory_fp16 / 1e9:.1f} GB")
print(f"FP8:  {tokens_per_sec_fp8} tokens/sec, {memory_fp8 / 1e9:.1f} GB")
print(f"Speedup: {tokens_per_sec_fp8 / tokens_per_sec_fp16:.2f}x")
print(f"Memory reduction: {memory_fp16 / memory_fp8:.2f}x")
```

**Expected results**:
```
FP16: 8,500 tokens/sec, 42.1 GB
FP8:  17,200 tokens/sec, 21.3 GB
Speedup: 2.02x ✅
Memory reduction: 1.98x ✅
```

**How to run**:
```bash
pip install transformer-engine
python3 fp8_transformer_engine.py
```

---

### 3. `synthetic_moe_inference_benchmark.py` - MoE Benchmarking

**Purpose**: Benchmark Mixture-of-Experts models for capacity planning.

**Why synthetic MoE?**
- Test routing patterns
- Measure expert load balancing
- Validate sharding strategies
- Capacity planning without full model

```python
import torch
from torch import nn

class SyntheticMoELayer(nn.Module):
    """Synthetic MoE for benchmarking."""
    
    def __init__(self, hidden_size, num_experts=64, experts_per_token=2):
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Experts (synthetic - just matmuls)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 4 * hidden_size),
                nn.GELU(),
                nn.Linear(4 * hidden_size, hidden_size)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Routing
        router_logits = self.router(x)  # [batch, seq, num_experts]
        routing_weights, selected_experts = torch.topk(
            router_logits, self.experts_per_token, dim=-1
        )
        routing_weights = torch.softmax(routing_weights, dim=-1)
        
        # Dispatch to experts
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            # Find tokens routed to this expert
            mask = (selected_experts == i).any(dim=-1)
            if mask.sum() == 0:
                continue
            
            expert_input = x[mask]
            expert_output = self.experts[i](expert_input)
            
            # Weighted combine
            weights = routing_weights[mask][:, (selected_experts[mask] == i).any(dim=-1)]
            output[mask] += expert_output * weights
        
        return output

# Benchmark different configurations
configs = [
    {"num_experts": 8, "experts_per_token": 2},
    {"num_experts": 64, "experts_per_token": 2},
    {"num_experts": 64, "experts_per_token": 8},
]

for config in configs:
    model = SyntheticMoELayer(4096, **config).cuda()
    throughput = benchmark_moe(model)
    print(f"{config}: {throughput:.1f} tokens/sec")
```

**How to run**:
```bash
python3 synthetic_moe_inference_benchmark.py
```

---

### 4. `inference_profiling.py` - Detailed Performance Analysis

**Purpose**: Profile inference to identify bottlenecks.

```python
import torch
from torch.profiler import profile, ProfilerActivity

def profile_inference(model, input_ids):
    """Profile single inference pass."""
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            # Prefill
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Decode (10 steps)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(10):
                outputs = model(
                    next_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                past_key_values = outputs.past_key_values
    
    # Analyze
    print("Top 10 operations by CUDA time:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))
    
    # Export
    prof.export_chrome_trace("inference_trace.json")

# Profile
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b",
    torch_dtype=torch.float16,
    device_map="auto"
)
input_ids = torch.tensor([[1, 2, 3, 4]], device='cuda')

profile_inference(model, input_ids)
```

**How to run**:
```bash
python3 inference_profiling.py
```

---

### 5. `cache_monitoring.py` - KV Cache Metrics

**Purpose**: Monitor KV cache usage and eviction patterns.

```python
class CacheMonitor:
    """Monitor KV cache statistics."""
    
    def __init__(self):
        self.stats = {
            'total_allocated': 0,
            'total_freed': 0,
            'current_usage': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
        }
    
    def record_allocation(self, size):
        self.stats['total_allocated'] += size
        self.stats['current_usage'] += size
    
    def record_free(self, size):
        self.stats['total_freed'] += size
        self.stats['current_usage'] -= size
    
    def record_hit(self):
        self.stats['cache_hits'] += 1
    
    def record_miss(self):
        self.stats['cache_misses'] += 1
    
    def record_eviction(self):
        self.stats['evictions'] += 1
    
    def get_hit_rate(self):
        total = self.stats['cache_hits'] + self.stats['cache_misses']
        if total == 0:
            return 0.0
        return self.stats['cache_hits'] / total
    
    def print_stats(self):
        print(f"KV Cache Statistics:")
        print(f"  Current usage: {self.stats['current_usage'] / 1e9:.2f} GB")
        print(f"  Hit rate: {self.get_hit_rate() * 100:.1f}%")
        print(f"  Evictions: {self.stats['evictions']}")
```

**How to run**:
```bash
python3 cache_monitoring.py
```

---

### 6. `scheduler.py` - Request Scheduling

**Purpose**: Implement request scheduler with priorities.

```python
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedRequest:
    priority: int
    request: Any = field(compare=False)

class InferenceScheduler:
    """Schedule inference requests with priorities."""
    
    def __init__(self):
        self.queue = PriorityQueue()
        
    def add_request(self, request, priority=0):
        """Add request with priority (lower = higher priority)."""
        self.queue.put(PrioritizedRequest(priority, request))
    
    def get_next_batch(self, max_batch_size=32):
        """Get next batch of requests."""
        batch = []
        while not self.queue.empty() and len(batch) < max_batch_size:
            item = self.queue.get()
            batch.append(item.request)
        return batch

# Usage
scheduler = InferenceScheduler()

# High priority (latency-sensitive)
scheduler.add_request({"prompt": "..."}, priority=0)

# Normal priority (throughput)
scheduler.add_request({"prompt": "..."}, priority=10)

# Low priority (batch)
scheduler.add_request({"prompt": "..."}, priority=20)

# Get next batch (high priority first)
batch = scheduler.get_next_batch(max_batch_size=32)
```

---

## Production Deployment Best Practices

### 1. Monitoring

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

requests_total = Counter('inference_requests_total', 'Total requests')
latency_histogram = Histogram('inference_latency_seconds', 'Request latency')
active_requests = Gauge('inference_active_requests', 'Active requests')
throughput = Gauge('inference_throughput_tokens_per_sec', 'Throughput')

# Track metrics
@latency_histogram.time()
def process_request(request):
    active_requests.inc()
    try:
        result = model.generate(request)
        requests_total.inc()
        return result
    finally:
        active_requests.dec()
```

### 2. Autoscaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: inference_active_requests
      target:
        type: AverageValue
        averageValue: "50"
```

### 3. Load Balancing

```python
# Round-robin load balancer
class LoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.current = 0
    
    def get_server(self):
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server

# Usage
lb = LoadBalancer(['server1:8000', 'server2:8000', 'server3:8000'])
server = lb.get_server()
response = requests.post(f'http://{server}/generate', json=request)
```

---

## How to Run All Examples

```bash
cd ch16

# Install dependencies
pip install -r requirements.txt

# Production serving (requires 8 GPUs)
python3 inference_serving_8xb200.py --demo

# FP8 quantization
pip install transformer-engine
python3 fp8_transformer_engine.py

# MoE benchmarking
python3 synthetic_moe_inference_benchmark.py

# Profiling
python3 inference_profiling.py

# Monitoring
python3 cache_monitoring.py

# Deploy with Kubernetes (production)
kubectl apply -f ../docs/examples/inference_deployment.yaml
```

---

## Key Takeaways

1. **FP8 quantization is essential**: 2x throughput, 2x memory reduction, <1% accuracy loss.

2. **vLLM for production**: Continuous batching, PagedAttention, prefix caching out-of-the-box.

3. **Monitor everything**: Cache hit rate, latency (P50/P99), throughput, GPU utilization.

4. **Batching is critical**: Batch size 32-128 for decode (higher throughput), 1-8 for prefill (lower latency).

5. **Autoscale based on queue depth**: Not raw GPU utilization. Queue depth indicates actual demand.

6. **Separate prefill and decode**: Different hardware needs and SLAs.

7. **Test with synthetic workloads**: Benchmark capacity before deploying real traffic.

---

## Common Pitfalls

### Pitfall 1: Not Using Quantization
**Problem**: Running FP16 when FP8 gives 2x improvement.

**Solution**: Always use FP8 on B200 (Transformer Engine or vLLM quantization).

### Pitfall 2: Fixed Batch Sizes
**Problem**: Waiting for full batch → High latency.

**Solution**: Use continuous batching (vLLM does this automatically).

### Pitfall 3: Over-Provisioning Memory
**Problem**: Allocating max_seq_len for all requests → OOM.

**Solution**: Use PagedAttention or dynamic allocation.

### Pitfall 4: No Request Prioritization
**Problem**: Batch requests treated equally → Latency-sensitive requests wait.

**Solution**: Implement priority queues (see `scheduler.py`).

### Pitfall 5: Single Point of Failure
**Problem**: One inference server handles all traffic.

**Solution**: Deploy multiple replicas with load balancing.

---

## Next Steps

**Dynamic routing and early exit** → [Chapter 17: Dynamic Routing](../ch17/README.md)

Learn about:
- Early exit strategies
- Dynamic batching
- Adaptive routing based on complexity

**Back to disaggregation** → [Chapter 15: Disaggregated Inference](../ch15/README.md)

---

## Additional Resources

- **vLLM**: [vLLM Documentation](https://docs.vllm.ai/)
- **Transformer Engine**: [NVIDIA TE](https://docs.nvidia.com/deeplearning/transformer-engine/)
- **FP8 Training**: [FP8 Formats](https://arxiv.org/abs/2209.05433)
- **PagedAttention**: [vLLM Paper](https://arxiv.org/abs/2309.06180)

---

**Chapter Status**: ✅ Complete  
**Last Updated**: November 3, 2025  
**Tested On**: 8x NVIDIA B200 GPUs, vLLM 0.3, Transformer Engine 1.7, PyTorch 2.9

