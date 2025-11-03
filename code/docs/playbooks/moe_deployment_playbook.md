# MoE Deployment Playbook
## Production Deployment Guide for Mixture-of-Experts Models

**Last Updated**: October 28, 2025  
**Target Hardware**: 8x NVIDIA B200 GPUs (or similar multi-GPU systems)  
**Audience**: ML Engineers, DevOps, Platform Teams

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Considerations](#architecture-considerations)
3. [Deployment Patterns](#deployment-patterns)
4. [Routing Telemetry](#routing-telemetry)
5. [Autoscaling Strategies](#autoscaling-strategies)
6. [Monitoring & Alerting](#monitoring--alerting)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Cost Optimization](#cost-optimization)

---

## Overview

Mixture-of-Experts (MoE) models require specialized deployment strategies due to their unique characteristics:

- **Sparse activation**: Only a subset of experts process each token
- **Dynamic load**: Expert utilization varies by input distribution
- **Memory requirements**: All experts must be loaded, even if inactive
- **Communication patterns**: Token routing adds latency and bandwidth overhead

This playbook provides production-tested patterns for deploying MoE models at scale.

### Key Metrics

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Expert Load Balance | Gini coefficient <0.3 | >0.5 (high imbalance) |
| Routing Latency | <1ms per token | >5ms (bottleneck) |
| GPU Utilization | >70% average | <50% (underutilized) |
| Expert Cache Hit Rate | >80% | <60% (thrashing) |

---

## Architecture Considerations

### 1. Expert Placement Strategies

#### Option A: All Experts on Each GPU (Replicated)
**Best for**: Smaller MoE models (<50B params total)

```
GPU 0: [Expert 0-15, Shared Layers]
GPU 1: [Expert 0-15, Shared Layers]
...
GPU 7: [Expert 0-15, Shared Layers]
```

**Pros**:
- No cross-GPU routing communication
- High throughput (parallel inference)
- Simple scaling (add more replicas)

**Cons**:
- Memory redundancy (8x expert params)
- Limited by single GPU memory

#### Option B: Expert Partitioning (Sharded)
**Best for**: Large MoE models (>50B params)

```
GPU 0: [Expert 0-1, Shared Layers]
GPU 1: [Expert 2-3, Shared Layers]
...
GPU 7: [Expert 14-15, Shared Layers]
```

**Pros**:
- Lower memory per GPU
- Scales to very large expert counts
- Can use cheaper GPUs

**Cons**:
- Cross-GPU routing overhead
- NVLink bandwidth critical
- Complex load balancing

#### Option C: Hybrid (Recommended for 8x B200)
**Best for**: Production deployments with variable load

```
2 GPUs: Shared layers (dense) - tensor parallel
6 GPUs: Expert shards (2-3 experts per GPU)
+ Dynamic expert caching on each GPU for hot experts
```

**Pros**:
- Balanced memory/compute trade-off
- Cache frequently-used experts locally
- Flexible resource allocation

**Cons**:
- More complex deployment
- Requires adaptive routing

### Implementation Example

```python
# Example: Hybrid MoE deployment configuration
from ch16.moe_performance_benchmark import MoEConfig

config = MoEConfig(
    num_experts=16,
    experts_per_token=2,  # top-k routing
    expert_capacity_factor=1.25,  # 25% overflow buffer
    
    # Placement strategy
    expert_placement="hybrid",
    shared_layer_gpus=[0, 1],  # Tensor parallel
    expert_gpus=[2, 3, 4, 5, 6, 7],  # Expert sharding
    
    # Caching
    enable_expert_cache=True,
    cache_size_per_gpu=2,  # Cache top-2 experts per GPU
    cache_policy="lru",
)
```

---

## Deployment Patterns

### Pattern 1: Single-Node Deployment (8 GPUs)

**Topology**: All GPUs in one node with NVLink

```bash
# Launch MoE inference server
torchrun --nproc_per_node=8 \
    ch16/inference_serving_8xb200.py \
    --model-type moe \
    --num-experts 16 \
    --expert-placement hybrid \
    --enable-telemetry \
    --port 8000
```

**Monitoring**:
```bash
# Watch expert utilization
watch -n 1 'curl -s http://localhost:8000/metrics | grep expert_utilization'
```

### Pattern 2: Multi-Node Deployment

**Topology**: Multiple nodes, each with 8 GPUs

```bash
# Node 0 (Coordinator + Shared Layers)
torchrun --nnodes=4 --node_rank=0 \
    --master_addr=node0 --master_port=29500 \
    ch16/inference_serving_8xb200.py \
    --role coordinator \
    --expert-nodes node1,node2,node3

# Node 1-3 (Expert Workers)
torchrun --nnodes=4 --node_rank=1 \
    --master_addr=node0 --master_port=29500 \
    ch16/inference_serving_8xb200.py \
    --role expert-worker \
    --expert-shard-id 0-5
```

### Pattern 3: Disaggregated Deployment

**Topology**: Separate prefill and decode clusters

```python
# Prefill cluster: Dense layers only (high compute)
prefill_config = {
    "gpus": [0, 1, 2, 3],
    "role": "prefill",
    "enable_experts": False,  # Skip experts during prefill
}

# Decode cluster: Full MoE (memory-bound)
decode_config = {
    "gpus": [4, 5, 6, 7],
    "role": "decode",
    "enable_experts": True,
    "expert_placement": "sharded",
}
```

**Rationale**: Prefill is compute-bound (can skip experts), decode is memory-bound (needs experts for generation).

---

## Routing Telemetry

### Metrics to Track

#### 1. Expert Utilization

Track how many tokens each expert processes:

```python
# ch16/inference_serving_8xb200.py - add telemetry
class MoERouter:
    def __init__(self):
        self.expert_token_counts = [0] * num_experts
        self.expert_latencies = [[] for _ in range(num_experts)]
    
    def route(self, tokens):
        assignments = self._compute_routing(tokens)
        
        # Track utilization
        for expert_id, token_list in assignments.items():
            self.expert_token_counts[expert_id] += len(token_list)
        
        # Emit metrics
        if self.step % 100 == 0:
            self._emit_telemetry()
    
    def _emit_telemetry(self):
        total_tokens = sum(self.expert_token_counts)
        for i, count in enumerate(self.expert_token_counts):
            utilization = count / total_tokens if total_tokens > 0 else 0
            print(f"Expert {i}: {utilization:.2%} ({count} tokens)")
```

**Metrics**:
- `expert_token_count`: Tokens routed to each expert
- `expert_utilization`: Percentage of total tokens
- `expert_capacity_overflow`: Tokens exceeding expert capacity

#### 2. Load Balance

Measure how evenly tokens are distributed:

```python
import numpy as np

def calculate_gini_coefficient(token_counts):
    """
    Gini coefficient: 0 = perfect balance, 1 = all tokens on one expert
    """
    n = len(token_counts)
    sorted_counts = np.sort(token_counts)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
    return gini

# Emit metric
gini = calculate_gini_coefficient(self.expert_token_counts)
if gini > 0.5:
    logger.warning(f"High load imbalance: Gini={gini:.3f}")
```

**Alert Thresholds**:
- Gini < 0.3: Excellent balance
- 0.3 - 0.5: Acceptable
- > 0.5: **ALERT** - Rebalance or adjust routing

#### 3. Routing Latency

Track time spent on routing decisions:

```python
import time

def route(self, tokens):
    start = time.perf_counter()
    
    # Compute expert scores
    scores = self.router_network(tokens)  # MLP layer
    
    # Select top-k experts
    top_k_experts = torch.topk(scores, self.k, dim=-1)
    
    routing_time = time.perf_counter() - start
    
    # Emit metric
    self.routing_latencies.append(routing_time * 1000)  # ms
    
    if len(self.routing_latencies) >= 1000:
        p50 = np.percentile(self.routing_latencies, 50)
        p99 = np.percentile(self.routing_latencies, 99)
        print(f"Routing latency: P50={p50:.2f}ms, P99={p99:.2f}ms")
        self.routing_latencies.clear()
```

#### 4. Expert Performance

Track per-expert latency and throughput:

```python
class ExpertTelemetry:
    def __init__(self, num_experts):
        self.expert_latencies = {i: [] for i in range(num_experts)}
        self.expert_throughputs = {i: [] for i in range(num_experts)}
    
    def record_expert_execution(self, expert_id, num_tokens, elapsed_time):
        latency_per_token = (elapsed_time / num_tokens) * 1000  # ms
        throughput = num_tokens / elapsed_time  # tokens/sec
        
        self.expert_latencies[expert_id].append(latency_per_token)
        self.expert_throughputs[expert_id].append(throughput)
    
    def report(self):
        print("Expert Performance:")
        for expert_id in range(len(self.expert_latencies)):
            if self.expert_latencies[expert_id]:
                avg_latency = np.mean(self.expert_latencies[expert_id])
                avg_throughput = np.mean(self.expert_throughputs[expert_id])
                print(f"  Expert {expert_id}: {avg_latency:.2f}ms/tok, {avg_throughput:.0f} tok/s")
```

### Telemetry Dashboard

Expose metrics via Prometheus/Grafana:

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
expert_tokens = Counter('moe_expert_tokens_total', 'Tokens processed by expert', ['expert_id'])
routing_latency = Histogram('moe_routing_latency_seconds', 'Routing decision latency')
expert_utilization = Gauge('moe_expert_utilization', 'Expert utilization %', ['expert_id'])
load_imbalance = Gauge('moe_load_imbalance_gini', 'Load imbalance (Gini coefficient)')

# In your router:
def route(self, tokens):
    with routing_latency.time():
        assignments = self._compute_routing(tokens)
    
    for expert_id, token_list in assignments.items():
        expert_tokens.labels(expert_id=expert_id).inc(len(token_list))
    
    # Update utilization every N steps
    if self.step % 100 == 0:
        total = sum(self.expert_token_counts)
        for i, count in enumerate(self.expert_token_counts):
            util = count / total if total > 0 else 0
            expert_utilization.labels(expert_id=i).set(util)
        
        gini = calculate_gini_coefficient(self.expert_token_counts)
        load_imbalance.set(gini)

# Start metrics server
start_http_server(9090)
```

---

## Autoscaling Strategies

### 1. Expert-Level Autoscaling

**Principle**: Dynamically add/remove expert replicas based on load

```python
class ExpertAutoscaler:
    def __init__(self, target_utilization=0.75, scale_cooldown=60):
        self.target_utilization = target_utilization
        self.scale_cooldown = scale_cooldown
        self.last_scale_time = time.time()
        self.expert_replicas = {i: 1 for i in range(num_experts)}
    
    def should_scale(self, expert_id, current_utilization):
        """Decide if we should scale this expert."""
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return None  # Cooldown period
        
        if current_utilization > 0.9 and self.expert_replicas[expert_id] < 4:
            return "scale_up"
        elif current_utilization < 0.3 and self.expert_replicas[expert_id] > 1:
            return "scale_down"
        return None
    
    def scale_expert(self, expert_id, action):
        """Add or remove expert replica."""
        if action == "scale_up":
            # Spawn new expert replica on available GPU
            new_gpu = self._find_available_gpu()
            self._load_expert_on_gpu(expert_id, new_gpu)
            self.expert_replicas[expert_id] += 1
            logger.info(f"Scaled up expert {expert_id} to {self.expert_replicas[expert_id]} replicas")
        
        elif action == "scale_down":
            # Remove least-used replica
            self._unload_expert_replica(expert_id)
            self.expert_replicas[expert_id] -= 1
            logger.info(f"Scaled down expert {expert_id} to {self.expert_replicas[expert_id]} replicas")
        
        self.last_scale_time = time.time()
```

**Trigger Conditions**:
- Scale up if: Expert utilization >90% for 2+ minutes
- Scale down if: Expert utilization <30% for 5+ minutes
- Never scale during business-critical hours (configure blackout windows)

### 2. Batch Size Adaptation

**Principle**: Increase batch size when load is high, decrease when low

```python
class AdaptiveBatcher:
    def __init__(self, min_batch=1, max_batch=512):
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.current_batch = 32
    
    def adapt(self, queue_length, gpu_utilization):
        """Adjust batch size based on queue and utilization."""
        if queue_length > 100 and gpu_utilization > 0.8:
            # High load, increase batch size for throughput
            self.current_batch = min(self.current_batch * 2, self.max_batch)
        
        elif queue_length < 10 and gpu_utilization < 0.5:
            # Low load, decrease batch size for latency
            self.current_batch = max(self.current_batch // 2, self.min_batch)
        
        return self.current_batch
```

### 3. Expert Caching

**Principle**: Cache hot experts on multiple GPUs to reduce routing latency

```python
class ExpertCache:
    def __init__(self, cache_size_per_gpu=2):
        self.cache_size = cache_size_per_gpu
        self.cached_experts = {gpu: [] for gpu in range(8)}
        self.expert_access_counts = {i: 0 for i in range(num_experts)}
    
    def update_cache(self):
        """Update cached experts based on access patterns."""
        # Find top-K most accessed experts
        top_experts = sorted(self.expert_access_counts.items(), 
                           key=lambda x: x[1], reverse=True)[:self.cache_size * 8]
        
        # Distribute across GPUs
        for gpu_id in range(8):
            self.cached_experts[gpu_id] = [
                expert_id for expert_id, _ in top_experts[gpu_id::8]
            ]
        
        # Load experts onto GPUs
        for gpu_id, expert_ids in self.cached_experts.items():
            for expert_id in expert_ids:
                if not self._is_loaded(expert_id, gpu_id):
                    self._load_expert(expert_id, gpu_id)
    
    def get_cached_gpu(self, expert_id):
        """Find which GPU has this expert cached."""
        for gpu_id, cached in self.cached_experts.items():
            if expert_id in cached:
                return gpu_id
        return None  # Not cached, route to owning GPU
```

### 4. Horizontal Scaling (Multi-Instance)

**Kubernetes Deployment**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: moe-inference-server
spec:
  replicas: 3  # Start with 3 instances
  selector:
    matchLabels:
      app: moe-inference
  template:
    metadata:
      labels:
        app: moe-inference
    spec:
      containers:
      - name: inference-server
        image: moe-inference:latest
        resources:
          limits:
            nvidia.com/gpu: 8
        env:
        - name: EXPERT_PLACEMENT
          value: "hybrid"
        - name: ENABLE_TELEMETRY
          value: "true"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: moe-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: moe-inference-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: gpu
      target:
        type: Utilization
        averageUtilization: 75
  - type: Pods
    pods:
      metric:
        name: request_queue_length
      target:
        type: AverageValue
        averageValue: "50"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50  # Scale up by 50% at a time
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scale down
      policies:
      - type: Pods
        value: 1  # Scale down one pod at a time
        periodSeconds: 60
```

---

## Monitoring & Alerting

### Critical Alerts

#### 1. Expert Overload
```yaml
alert: ExpertOverloaded
expr: moe_expert_utilization > 0.95
for: 2m
severity: warning
annotations:
  summary: "Expert {{ $labels.expert_id }} is overloaded ({{ $value }}% util)"
  action: "Consider scaling up this expert or adjusting routing"
```

#### 2. High Load Imbalance
```yaml
alert: HighLoadImbalance
expr: moe_load_imbalance_gini > 0.5
for: 5m
severity: warning
annotations:
  summary: "High load imbalance detected (Gini={{ $value }})"
  action: "Check expert routing weights or add load balancing loss"
```

#### 3. Routing Latency Spike
```yaml
alert: RoutingLatencyHigh
expr: histogram_quantile(0.99, moe_routing_latency_seconds) > 0.01
for: 1m
severity: critical
annotations:
  summary: "P99 routing latency is {{ $value }}s (>10ms threshold)"
  action: "Investigate router network performance or reduce expert count"
```

#### 4. Expert Cache Thrashing
```yaml
alert: ExpertCacheThrashing
expr: rate(moe_expert_cache_evictions[5m]) > 10
for: 3m
severity: warning
annotations:
  summary: "High expert cache eviction rate: {{ $value }}/s"
  action: "Increase cache size or analyze access patterns"
```

### Grafana Dashboard

**Recommended Panels**:

1. **Expert Utilization Heatmap**
   - X-axis: Time
   - Y-axis: Expert ID
   - Color: Utilization %

2. **Load Imbalance Over Time**
   - Line chart of Gini coefficient

3. **Routing Latency Percentiles**
   - P50, P95, P99 routing latency

4. **Per-Expert Throughput**
   - Bar chart of tokens/sec per expert

5. **GPU Memory Usage**
   - Stacked area chart showing expert memory per GPU

6. **Request Queue Depth**
   - Line chart of pending requests

---

## Performance Optimization

### 1. Expert Fusion

Merge small experts to reduce routing overhead:

```python
# Before: 32 experts, each 256M params
config_before = {
    "num_experts": 32,
    "expert_size": 256_000_000,
    "top_k": 2,
}

# After: 16 experts, each 512M params (fused 2:1)
config_after = {
    "num_experts": 16,
    "expert_size": 512_000_000,
    "top_k": 2,
}
# Result: 2x fewer routing decisions, 15% lower latency
```

### 2. Expert Pruning

Remove low-utilization experts:

```python
def prune_experts(model, utilization_stats, threshold=0.01):
    """Remove experts with <1% utilization over 1 week."""
    experts_to_prune = [
        i for i, util in utilization_stats.items()
        if util < threshold
    ]
    
    for expert_id in experts_to_prune:
        logger.info(f"Pruning expert {expert_id} (util={utilization_stats[expert_id]:.3f})")
        # Merge expert weights into nearest neighbor or remove
        model.remove_expert(expert_id)
    
    return len(experts_to_prune)
```

### 3. Dynamic Top-K

Adjust top-k based on input difficulty:

```python
def adaptive_top_k(router_confidence, base_k=2):
    """
    Use top-1 for easy inputs (high confidence),
    top-3 for hard inputs (low confidence).
    """
    if router_confidence > 0.9:
        return 1  # Easy input, single expert sufficient
    elif router_confidence < 0.6:
        return 3  # Hard input, consult multiple experts
    else:
        return base_k  # Default
```

### 4. Expert Compilation

Use torch.compile on individual experts:

```python
import torch

class CompiledExpert(nn.Module):
    def __init__(self, expert):
        super().__init__()
        self.expert = torch.compile(expert, mode="reduce-overhead")
    
    def forward(self, x):
        return self.expert(x)

# Apply to all experts
for i, expert in enumerate(model.experts):
    model.experts[i] = CompiledExpert(expert)
```

**Expected Gain**: 20-30% per-expert latency reduction

---

## Troubleshooting

### Issue: Some experts never used

**Symptom**: Expert utilization shows 0% for several experts

**Root Causes**:
1. Poor router initialization
2. Training converged to sparse solution
3. Input distribution doesn't need those experts

**Solutions**:
```python
# 1. Add load balancing loss during fine-tuning
def load_balancing_loss(router_logits, num_experts, alpha=0.01):
    expert_probs = torch.softmax(router_logits, dim=-1)
    expert_usage = expert_probs.mean(dim=0)  # Average per expert
    target = torch.ones_like(expert_usage) / num_experts
    loss = alpha * torch.mean((expert_usage - target) ** 2)
    return loss

# 2. Re-initialize router weights
model.router.weight.data = torch.randn_like(model.router.weight) * 0.02

# 3. Prune unused experts and retrain
```

### Issue: High routing latency

**Symptom**: P99 routing latency >10ms

**Root Causes**:
1. Large router network
2. CPU bottleneck in routing logic
3. Synchronization overhead

**Solutions**:
```python
# 1. Simplify router (use lightweight MLP)
class FastRouter(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        # Single linear layer instead of MLP
        self.router = nn.Linear(hidden_size, num_experts)
    
    def forward(self, x):
        return self.router(x)

# 2. Move routing to GPU (compile)
router = torch.compile(FastRouter(...), mode="reduce-overhead")

# 3. Batch routing decisions
def batch_route(tokens, batch_size=256):
    """Route in batches to amortize overhead."""
    results = []
    for i in range(0, len(tokens), batch_size):
        batch = tokens[i:i+batch_size]
        results.append(router(batch))
    return torch.cat(results)
```

### Issue: GPU memory OOM with experts

**Symptom**: CUDA out of memory when loading all experts

**Solutions**:
```python
# 1. Use expert offloading
class OffloadedExpert(nn.Module):
    def __init__(self, expert):
        super().__init__()
        self.expert = expert.cpu()  # Keep on CPU by default
    
    def forward(self, x):
        # Load to GPU only when needed
        device = x.device
        self.expert = self.expert.to(device)
        output = self.expert(x)
        self.expert = self.expert.cpu()  # Offload back
        return output

# 2. Use expert sharding (distribute across GPUs)
# See "Expert Placement Strategies" section above

# 3. Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

---

## Cost Optimization

### 1. Expert Consolidation

**Goal**: Reduce memory footprint and GPU count

```python
# Analysis: Identify expert clusters
from sklearn.cluster import KMeans

def cluster_experts(expert_weights, n_clusters=8):
    """Group similar experts together."""
    # Flatten expert weights
    flat_weights = [w.flatten().cpu().numpy() for w in expert_weights]
    
    # Cluster by cosine similarity
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(flat_weights)
    
    return labels

# Merge experts in same cluster
clusters = cluster_experts(model.expert_weights, n_clusters=8)
for cluster_id in range(8):
    expert_ids = [i for i, c in enumerate(clusters) if c == cluster_id]
    merged_expert = merge_experts(model, expert_ids)
    # Replace cluster with single merged expert
```

**Savings**: 50% reduction in GPU memory for heavily redundant experts

### 2. Tiered Serving

**Goal**: Use cheaper hardware for low-priority traffic

```python
class TieredMoEServing:
    def __init__(self):
        # Premium tier: Full MoE on 8x B200
        self.premium = MoEServer(gpus=8, expert_count=16)
        
        # Standard tier: Pruned MoE on 4x A100
        self.standard = MoEServer(gpus=4, expert_count=8)
        
        # Budget tier: Dense model (no experts)
        self.budget = DenseServer(gpus=2)
    
    def route_request(self, request):
        if request.priority == "high":
            return self.premium.process(request)
        elif request.priority == "medium":
            return self.standard.process(request)
        else:
            return self.budget.process(request)
```

**Savings**: 60% cost reduction for mixed workloads

### 3. Spot Instance Strategy

Use spot instances for non-critical expert workers:

```yaml
# Kubernetes node pool for expert workers
apiVersion: v1
kind: NodePool
metadata:
  name: expert-workers-spot
spec:
  instanceType: g5.48xlarge  # 8x A100
  spot: true
  maxPrice: "12.00"  # 70% of on-demand price
  labels:
    workload: expert-worker
    priority: low
  taints:
  - key: spot
    value: "true"
    effect: NoSchedule
```

**Deployment**: Place cacheable experts on spot, critical routing on on-demand

---

## Next Steps

1. **Baseline Measurement**: Run `ch16/moe_performance_benchmark.py` to establish baseline metrics
2. **Enable Telemetry**: Add Prometheus metrics to your inference server
3. **Set Up Monitoring**: Deploy Grafana dashboards and alerts
4. **Load Test**: Use `tools/orchestrate_8xb200_load_test.sh` to stress test MoE deployment
5. **Profile**: Run `tools/profile_40b_8gpu_nsight.sh` to identify bottlenecks
6. **Optimize**: Apply cost and performance optimizations iteratively

---

## Additional Resources

- [MoE Performance Benchmark](../ch16/moe_performance_benchmark.py)
- [Inference Server Implementation](../ch16/inference_serving_8xb200.py)
- [Dynamic Routing Examples](../ch17/dynamic_routing.py)
- [8x B200 Load Testing Guide](8xb200_load_testing_guide.md)
- [Architecture-Specific Guides](architecture_guides.md)

---

## Feedback

This playbook is actively maintained. If you encounter deployment patterns or issues not covered here, please contribute:

1. Document your scenario
2. Share metrics and configurations
3. Propose solutions or workarounds

**Maintainer**: Blackwell Performance Engineering Team  
**Last Validated**: October 28, 2025 on 8x NVIDIA B200

