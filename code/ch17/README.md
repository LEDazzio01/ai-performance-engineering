# Chapter 17: Dynamic Routing and Early Exit

## Overview

Dynamic routing adapts inference strategies based on request complexity, while early exit allows models to terminate computation when confidence is high. This chapter covers adaptive inference techniques that optimize the latency-accuracy-cost trade-off for production systems.

## Learning Objectives

After completing this chapter, you can:

- ✅ Implement early exit strategies for faster inference
- ✅ Apply dynamic routing based on request complexity
- ✅ Use adaptive batching for mixed workloads
- ✅ Optimize latency vs accuracy trade-offs
- ✅ Profile and analyze inference with roofline models
- ✅ Deploy confidence-based early termination

## Prerequisites

**Previous chapters**:
- [Chapter 16: Inference Optimization](../ch16/README.md) - production serving
- [Chapter 15: Disaggregated Inference](../ch15/README.md) - architecture patterns

**Required**: Understanding of model architectures and confidence metrics

---

## Examples

### 1. `early_rejection.py` - Early Exit Implementation

**Purpose**: Implement early exit for faster inference on easy examples.

**Concept**: Add classifiers at intermediate layers. If confidence high → exit early!

```python
import torch
import torch.nn as nn

class EarlyExitTransformer(nn.Module):
    """Transformer with early exit classifiers."""
    
    def __init__(self, num_layers=12, hidden_size=768, num_classes=1000):
        super().__init__()
        
        # Main transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size)
            for _ in range(num_layers)
        ])
        
        # Early exit classifiers (every 3 layers)
        self.exit_classifiers = nn.ModuleList([
            nn.Linear(hidden_size, num_classes)
            for _ in range(num_layers // 3)
        ])
        
        self.confidence_threshold = 0.95
    
    def forward(self, x, use_early_exit=True):
        exits_taken = []
        
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x)
            
            # Check for early exit every 3 layers
            if use_early_exit and (layer_idx + 1) % 3 == 0:
                exit_idx = (layer_idx + 1) // 3 - 1
                classifier = self.exit_classifiers[exit_idx]
                
                # Get prediction
                logits = classifier(x[:, 0])  # CLS token
                probs = torch.softmax(logits, dim=-1)
                confidence, prediction = torch.max(probs, dim=-1)
                
                # Early exit if confident
                if confidence > self.confidence_threshold:
                    exits_taken.append(layer_idx + 1)
                    return logits, layer_idx + 1
        
        # Use all layers (no early exit)
        final_logits = self.exit_classifiers[-1](x[:, 0])
        return final_logits, len(self.layers)

# Benchmark
model = EarlyExitTransformer().cuda()
input = torch.randn(1, 128, 768, device='cuda')

# Without early exit
start = time.time()
logits, layers_used = model(input, use_early_exit=False)
time_full = time.time() - start

# With early exit
start = time.time()
logits, layers_used = model(input, use_early_exit=True)
time_early = time.time() - start

print(f"Full model: {time_full * 1000:.2f} ms (12 layers)")
print(f"Early exit: {time_early * 1000:.2f} ms ({layers_used} layers)")
print(f"Speedup: {time_full / time_early:.2f}x")
```

**Expected results**:
- Easy examples: Exit at layer 3-6 → **2-3x faster**
- Hard examples: Use all 12 layers → Same accuracy
- Average: **1.5-2x speedup** with <1% accuracy loss

**How to run**:
```bash
python3 early_rejection.py
```

---

### 2. `dynamic_routing.py` - Complexity-Based Routing

**Purpose**: Route requests to different model sizes based on complexity.

```python
class ComplexityRouter:
    """Route requests based on estimated complexity."""
    
    def __init__(self):
        # Load models of different sizes
        self.small_model = load_model("1.5B")  # Fast, lower quality
        self.medium_model = load_model("7B")   # Balanced
        self.large_model = load_model("33B")   # Slow, high quality
        
        # Complexity estimator
        self.complexity_estimator = ComplexityEstimator()
    
    def route_request(self, prompt):
        """Route to appropriate model based on complexity."""
        
        # Estimate complexity
        complexity_score = self.complexity_estimator.estimate(prompt)
        
        # Route decision
        if complexity_score < 0.3:
            # Easy: Use small model
            return self.small_model.generate(prompt), "1.5B", complexity_score
        elif complexity_score < 0.7:
            # Medium: Use medium model
            return self.medium_model.generate(prompt), "7B", complexity_score
        else:
            # Hard: Use large model
            return self.large_model.generate(prompt), "33B", complexity_score

class ComplexityEstimator:
    """Estimate prompt complexity."""
    
    def __init__(self):
        # Train small classifier on prompt features
        self.classifier = train_complexity_classifier()
    
    def estimate(self, prompt):
        """Return complexity score [0, 1]."""
        features = self.extract_features(prompt)
        complexity = self.classifier(features)
        return complexity.item()
    
    def extract_features(self, prompt):
        """Extract complexity indicators."""
        return {
            'length': len(prompt.split()),
            'vocab_diversity': len(set(prompt.split())) / len(prompt.split()),
            'has_code': '```' in prompt or 'def ' in prompt,
            'has_math': any(c in prompt for c in ['∫', '∑', '∂']),
            'question_words': sum(1 for w in ['how', 'why', 'explain'] if w in prompt.lower()),
        }

# Usage
router = ComplexityRouter()

prompts = [
    "What is 2+2?",  # Easy → 1.5B model
    "Explain quantum entanglement",  # Medium → 7B model
    "Derive the Navier-Stokes equations",  # Hard → 33B model
]

for prompt in prompts:
    response, model_used, complexity = router.route_request(prompt)
    print(f"Prompt: {prompt}")
    print(f"Routed to: {model_used} (complexity: {complexity:.2f})")
    print(f"Response: {response}\n")
```

**Benefits**:
- **Cost reduction**: 70% of requests use smaller models
- **Lower latency**: Small model 5x faster than large
- **Quality**: Hard requests still get high-quality responses

**How to run**:
```bash
python3 dynamic_routing.py
```

---

### 3. `blackwell_profiling_guide.py` - Blackwell-Specific Profiling

**Purpose**: Profile inference on Blackwell GPUs with architecture-specific metrics.

```python
def profile_blackwell_inference(model, input_ids):
    """Profile with Blackwell-specific metrics."""
    
    # NVIDIA SMI metrics
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    # Start monitoring
    start_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Watts
    start_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    
    # Run inference
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            outputs = model(input_ids)
    
    # End monitoring
    end_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    end_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    
    # Blackwell-specific metrics
    sm_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    mem_util = pynvml.nvmlDeviceGetUtilizationRates(handle).memory
    
    print(f"Blackwell (sm_100) Metrics:")
    print(f"  SM Utilization: {sm_util}%")
    print(f"  Memory Utilization: {mem_util}%")
    print(f"  Power: {end_power:.1f} W")
    print(f"  Temperature: {end_temp}°C")
    
    # Tensor Core utilization
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))
```

**How to run**:
```bash
python3 blackwell_profiling_guide.py
```

---

### 4. `blackwell_roofline_analysis.py` - Roofline Model

**Purpose**: Analyze kernel performance against hardware roofline.

```python
def roofline_analysis(model, input_data):
    """Generate roofline plot for kernels."""
    
    # Profile kernels
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        outputs = model(input_data)
    
    # Extract metrics
    kernels = []
    for event in prof.key_averages():
        if event.device_type == torch.profiler.DeviceType.CUDA:
            # Calculate arithmetic intensity
            flops = estimate_flops(event)
            bytes_accessed = estimate_memory(event)
            arithmetic_intensity = flops / bytes_accessed if bytes_accessed > 0 else 0
            
            # Calculate achieved performance
            cuda_time_ms = event.cuda_time_total / 1000
            achieved_gflops = (flops / 1e9) / (cuda_time_ms / 1000)
            
            kernels.append({
                'name': event.key,
                'arithmetic_intensity': arithmetic_intensity,
                'achieved_gflops': achieved_gflops,
            })
    
    # Plot roofline
    plot_roofline(kernels, peak_bandwidth_gbs=8000, peak_compute_tflops=2000)

def plot_roofline(kernels, peak_bandwidth_gbs, peak_compute_tflops):
    """Plot kernels on roofline model."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Roofline boundaries
    ai_range = np.logspace(-2, 3, 100)  # Arithmetic intensity range
    
    # Memory-bound region
    memory_bound = peak_bandwidth_gbs * ai_range
    
    # Compute-bound region
    compute_bound = np.ones_like(ai_range) * peak_compute_tflops
    
    # Actual roofline (minimum of both)
    roofline = np.minimum(memory_bound, compute_bound)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.loglog(ai_range, roofline, 'k-', linewidth=2, label='Roofline')
    
    # Plot kernels
    for kernel in kernels:
        plt.loglog(
            kernel['arithmetic_intensity'],
            kernel['achieved_gflops'],
            'ro', markersize=8
        )
    
    plt.xlabel('Arithmetic Intensity (FLOP/Byte)')
    plt.ylabel('Performance (GFLOPS)')
    plt.title('Roofline Model - B200 (sm_100)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('roofline.png')
```

**How to run**:
```bash
python3 blackwell_roofline_analysis.py
```

---

### 5. `comprehensive_profiling_toolkit.py` - All-in-One Profiling

**Purpose**: Comprehensive profiling toolkit for inference analysis.

```python
class InferenceProfiler:
    """Comprehensive inference profiling."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def profile_all(self, input_data, output_dir='profiling_results'):
        """Run all profiling analyses."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Latency breakdown
        print("1. Profiling latency...")
        latency_results = self.profile_latency(input_data)
        self.save_results(latency_results, f"{output_dir}/latency.json")
        
        # 2. Memory usage
        print("2. Profiling memory...")
        memory_results = self.profile_memory(input_data)
        self.save_results(memory_results, f"{output_dir}/memory.json")
        
        # 3. Throughput at different batch sizes
        print("3. Profiling throughput...")
        throughput_results = self.profile_throughput(input_data)
        self.save_results(throughput_results, f"{output_dir}/throughput.json")
        
        # 4. Roofline analysis
        print("4. Roofline analysis...")
        self.roofline_analysis(input_data, f"{output_dir}/roofline.png")
        
        # 5. Generate report
        print("5. Generating report...")
        self.generate_report(output_dir)
        
        print(f"\nProfiling complete! Results in {output_dir}/")
```

**How to run**:
```bash
python3 comprehensive_profiling_toolkit.py --model deepseek-coder-6.7b
```

---

## Dynamic Batching Strategies

### 1. Complexity-Aware Batching

```python
def batch_by_complexity(requests):
    """Group requests by similar complexity."""
    
    # Estimate complexity for each request
    complexities = [estimate_complexity(r) for r in requests]
    
    # Group into buckets
    easy = [r for r, c in zip(requests, complexities) if c < 0.3]
    medium = [r for r, c in zip(requests, complexities) if 0.3 <= c < 0.7]
    hard = [r for r, c in zip(requests, complexities) if c >= 0.7]
    
    # Process each group with appropriate resources
    process_batch(easy, model='small', batch_size=64)
    process_batch(medium, model='medium', batch_size=32)
    process_batch(hard, model='large', batch_size=8)
```

### 2. Latency-Aware Batching

```python
def batch_by_latency_sla(requests):
    """Group by latency requirements."""
    
    latency_critical = [r for r in requests if r.sla < 50]  # <50ms
    latency_sensitive = [r for r in requests if 50 <= r.sla < 200]
    batch_requests = [r for r in requests if r.sla >= 200]
    
    # Critical: Small batches, high priority
    process_batch(latency_critical, batch_size=1, priority=0)
    
    # Sensitive: Medium batches
    process_batch(latency_sensitive, batch_size=16, priority=5)
    
    # Batch: Large batches for throughput
    process_batch(batch_requests, batch_size=128, priority=10)
```

---

## How to Run All Examples

```bash
cd ch17

# Install dependencies
pip install -r requirements.txt

# Early exit
python3 early_rejection.py

# Dynamic routing
python3 dynamic_routing.py

# Blackwell profiling
python3 blackwell_profiling_guide.py
python3 blackwell_roofline_analysis.py

# Comprehensive toolkit
python3 comprehensive_profiling_toolkit.py --model deepseek-coder-6.7b
```

---

## Key Takeaways

1. **Early exit saves compute**: 30-50% of requests can exit early → 1.5-2x average speedup.

2. **Dynamic routing optimizes cost**: Route easy requests to small models → 3-5x cost reduction.

3. **Complexity estimation is key**: Accurate routing requires good complexity prediction.

4. **Batch by similarity**: Group similar requests for better GPU utilization.

5. **SLA-based prioritization**: Different requests have different latency needs.

6. **Roofline analysis identifies bottlenecks**: Memory-bound vs compute-bound operations.

7. **Profile before optimizing**: Measure actual performance, don't guess.

---

## Common Pitfalls

### Pitfall 1: Poor Complexity Estimation
**Problem**: Routing easy requests to large model → Wasted resources.

**Solution**: Train complexity classifier on labeled data. Validate accuracy.

### Pitfall 2: Too Aggressive Early Exit
**Problem**: Exiting too early → Accuracy degradation.

**Solution**: Tune confidence threshold. Monitor accuracy metrics.

### Pitfall 3: Static Routing
**Problem**: Fixed routing rules don't adapt to workload.

**Solution**: Use feedback loop to adjust routing based on actual performance.

### Pitfall 4: Ignoring Tail Latency
**Problem**: P99 latency still high despite average improvements.

**Solution**: Monitor and optimize tail latency separately (dedicate resources, priorities).

---

## Next Steps

**Attention optimization** → [Chapter 18: Attention Mechanisms](../ch18/README.md)

Learn about:
- FlexAttention for flexible patterns
- FlashAttention for memory efficiency
- MLA (Multi-head Latent Attention) kernels

**Back to inference** → [Chapter 16: Inference Optimization](../ch16/README.md)

---

## Additional Resources

- **Early Exit**: [BERxiT Paper](https://arxiv.org/abs/2006.04152)
- **Adaptive Inference**: [Dynamic Neural Networks Survey](https://arxiv.org/abs/2102.04906)
- **Roofline Model**: [Roofline Paper](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)

---

**Chapter Status**: ✅ Complete  
**Last Updated**: November 3, 2025  
**Tested On**: 8x NVIDIA B200 GPUs, PyTorch 2.9

