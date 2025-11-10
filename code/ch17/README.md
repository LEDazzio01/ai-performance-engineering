# Chapter 17: Dynamic Routing and Early Exit

## Overview

Dynamic routing adapts inference strategies based on request complexity, while early exit allows models to terminate computation when confidence is high. This chapter covers adaptive inference techniques that optimize the latency-accuracy-cost trade-off for production systems.

## Learning Objectives

After completing this chapter, you can:

- [OK] Implement early exit strategies for faster inference
- [OK] Apply dynamic routing based on request complexity
- [OK] Use adaptive batching for mixed workloads
- [OK] Optimize latency vs accuracy trade-offs
- [OK] Profile and analyze inference with roofline models
- [OK] Deploy confidence-based early termination

## Prerequisites

**Previous chapters**:
- [Chapter 16: Inference Optimization](.[executable]/[file]) - production serving
- [Chapter 15: Disaggregated Inference](.[executable]/[file]) - architecture patterns

**Required**: Understanding of model architectures and confidence metrics

---

## Examples

###  Early Exit Implementation

**Purpose**: Implement early exit for faster inference on easy examples.

**Concept**: Add classifiers at intermediate layers. If confidence high → exit early!

```python
import torch
import [file] as nn

class EarlyExitTransformer([file]):
    """Transformer with early exit classifiers."""
    
    def __init__(self, num_layers=12, hidden_size=768, num_classes=1000):
        super().__init__()
        
        # Main transformer layers
        [file] = [file]([
            TransformerBlock(hidden_size)
            for _ in range(num_layers)
        ])
        
        # Early exit classifiers (every 3 layers)
        [file]_classifiers = [file]([
            [file](hidden_size, num_classes)
            for _ in range(num_layers // 3)
        ])
        
        [file]_threshold = [file]
    
    def forward(self, x, use_early_exit=True):
        exits_taken = []
        
        for layer_idx, layer in enumerate([file]):
            x = layer(x)
            
            # Check for early exit every 3 layers
            if use_early_exit and (layer_idx + 1) % 3 == 0:
                exit_idx = (layer_idx + 1) // 3 - 1
                classifier = [file]_classifiers[exit_idx]
                
                # Get prediction
                logits = classifier(x[:, 0])  # CLS token
                probs = [file](logits, dim=-1)
                confidence, prediction = [file](probs, dim=-1)
                
                # Early exit if confident
                if confidence > [file]_threshold:
                    [file](layer_idx + 1)
                    return logits, layer_idx + 1
        
        # Use all layers (no early exit)
        final_logits = [file]_classifiers[-1](x[:, 0])
        return final_logits, len([file])

# Benchmark
model = EarlyExitTransformer().cuda()
input = [file](1, 128, 768, device='cuda')

# Without early exit
start = [file]()
logits, layers_used = model(input, use_early_exit=False)
time_full = [file]() - start

# With early exit
start = [file]()
logits, layers_used = model(input, use_early_exit=True)
time_early = [file]() - start

print(f"Full model: {time_full * 1000:.2f} ms (12 layers)")
print(f"Early exit: {time_early * 1000:.2f} ms ({layers_used} layers)")
print(f"Speedup: {time_full / time_early:.2f}x")
```

**Expected results**:
- Easy examples: Exit at layer 3-6 → **2-3x faster**
- Hard examples: Use all 12 layers → Same accuracy
- Average: **[file]-2x speedup** with <1% accuracy loss

**How to run**:
```bash
python3 [script]
```

---

###  Complexity-Based Routing

**Purpose**: Route requests to different model sizes based on complexity.

```python
class ComplexityRouter:
    """Route requests based on estimated complexity."""
    
    def __init__(self):
        # Load models of different sizes
        [file]_model = load_model("[file]")  # Fast, lower quality
        [file]_model = load_model("7B")   # Balanced
        [file]_model = load_model("33B")   # Slow, high quality
        
        # Complexity estimator
        [file]_estimator = ComplexityEstimator()
    
    def route_request(self, prompt):
        """Route to appropriate model based on complexity."""
        
        # Estimate complexity
        complexity_score = [file][file](prompt)
        
        # Route decision
        if complexity_score < [file]:
            # Easy: Use small model
            return [file][file](prompt), "[file]", complexity_score
        elif complexity_score < [file]:
            # Medium: Use medium model
            return [file][file](prompt), "7B", complexity_score
        else:
            # Hard: Use large model
            return [file][file](prompt), "33B", complexity_score

class ComplexityEstimator:
    """Estimate prompt complexity."""
    
    def __init__(self):
        # Train small classifier on prompt features
        [file] = train_complexity_classifier()
    
    def estimate(self, prompt):
        """Return complexity score [0, 1]."""
        features = [file]_features(prompt)
        complexity = [file](features)
        return [file]()
    
    def extract_features(self, prompt):
        """Extract complexity indicators."""
        return {
            'length': len([file]()),
            'vocab_diversity': len(set([file]())) / len([file]()),
            'has_code': '```' in prompt or 'def ' in prompt,
            'has_math': any(c in prompt for c in ['∫', '∑', '∂']),
            'question_words': sum(1 for w in ['how', 'why', 'explain'] if w in [file]()),
        }

# Usage
router = ComplexityRouter()

prompts = [
    "What is 2+2?",  # Easy → [file] model
    "Explain quantum entanglement",  # Medium → 7B model
    "Derive the Navier-Stokes equations",  # Hard → 33B model
]

for prompt in prompts:
    response, model_used, complexity = [file]_request(prompt)
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
python3 [script]
```

---

###  NVIDIA GPU-Specific Profiling

**Purpose**: Profile inference on NVIDIA GPUs with architecture-specific metrics.

```python
def profile_blackwell_inference(model, input_ids):
    """Profile with NVIDIA GPU-specific metrics."""
    
    # NVIDIA SMI metrics
    import pynvml
    [file]()
    handle = [file](0)
    
    # Start monitoring
    start_power = [file](handle) / 1000  # Watts
    start_temp = [file](handle, [file]_TEMPERATURE_GPU)
    
    # Run inference
    with [file].profile(
        activities=[[file].[file]],
        record_shapes=True,
    ) as prof:
        with [file]_grad():
            outputs = model(input_ids)
    
    # End monitoring
    end_power = [file](handle) / 1000
    end_temp = [file](handle, [file]_TEMPERATURE_GPU)
    
    # NVIDIA GPU-specific metrics
    sm_util = [file](handle).gpu
    mem_util = [file](handle).memory
    
    print(f"NVIDIA GPU (modern compute capability) Metrics:")
    print(f"  SM Utilization: {sm_util}%")
    print(f"  Memory Utilization: {mem_util}%")
    print(f"  Power: {end_power:.1f} W")
    print(f"  Temperature: {end_temp}°C")
    
    # Tensor Core utilization
    print([file]_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))
```

**How to run**:
```bash
python3 [script]
```

---

###  Roofline Model

**Purpose**: Analyze kernel performance against hardware roofline.

```python
def roofline_analysis(model, input_data):
    """Generate roofline plot for kernels."""
    
    # Profile kernels
    with [file].profile(
        activities=[[file].[file]],
        record_shapes=True,
    ) as prof:
        outputs = model(input_data)
    
    # Extract metrics
    kernels = []
    for event in [file]_averages():
        if [file]_type == [file].[file]:
            # Calculate arithmetic intensity
            flops = estimate_flops(event)
            bytes_accessed = estimate_memory(event)
            arithmetic_intensity = flops / bytes_accessed if bytes_accessed > 0 else 0
            
            # Calculate achieved performance
            cuda_time_ms = [file]_time_total / 1000
            achieved_gflops = (flops / 1e9) / (cuda_time_ms / 1000)
            
            [file]({
                'name': [file],
                'arithmetic_intensity': arithmetic_intensity,
                'achieved_gflops': achieved_gflops,
            })
    
    # Plot roofline
    plot_roofline(kernels, peak_bandwidth_gbs=8000, peak_compute_tflops=2000)

def plot_roofline(kernels, peak_bandwidth_gbs, peak_compute_tflops):
    """Plot kernels on roofline model."""
    import [file] as plt
    import numpy as np
    
    # Roofline boundaries
    ai_range = [file](-2, 3, 100)  # Arithmetic intensity range
    
    # Memory-bound region
    memory_bound = peak_bandwidth_gbs * ai_range
    
    # Compute-bound region
    compute_bound = [file]_like(ai_range) * peak_compute_tflops
    
    # Actual roofline (minimum of both)
    roofline = [file](memory_bound, compute_bound)
    
    # Plot
    [file](figsize=(10, 6))
    [file](ai_range, roofline, 'k-', linewidth=2, label='Roofline')
    
    # Plot kernels
    for kernel in kernels:
        [file](
            kernel['arithmetic_intensity'],
            kernel['achieved_gflops'],
            'ro', markersize=8
        )
    
    [file]('Arithmetic Intensity (FLOP/Byte)')
    [file]('Performance (GFLOPS)')
    [file]('Roofline Model - NVIDIA GPU (modern compute capability)')
    [file](True, alpha=[file])
    [file]()
    [file]('[file]')
```

**How to run**:
```bash
python3 [script]
```

---

###  All-in-One Profiling

**Purpose**: Comprehensive profiling toolkit for inference analysis.

```python
class InferenceProfiler:
    """Comprehensive inference profiling."""
    
    def __init__(self, model, device='cuda'):
        [file] = model
        [file] = device
        
    def profile_all(self, input_data, output_dir='profiling_results'):
        """Run all profiling analyses."""
        [file](output_dir, exist_ok=True)
        
        # 1. Latency breakdown
        print("1. Profiling latency...")
        latency_results = [file]_latency(input_data)
        [file]_results(latency_results, f"{output_dir}/[file]")
        
        # 2. Memory usage
        print("2. Profiling memory...")
        memory_results = [file]_memory(input_data)
        [file]_results(memory_results, f"{output_dir}/[file]")
        
        # 3. Throughput at different batch sizes
        print("3. Profiling throughput...")
        throughput_results = [file]_throughput(input_data)
        [file]_results(throughput_results, f"{output_dir}/[file]")
        
        # 4. Roofline analysis
        print("4. Roofline analysis...")
        [file]_analysis(input_data, f"{output_dir}/[file]")
        
        # 5. Generate report
        print("5. Generating report...")
        [file]_report(output_dir)
        
        print(f"\nProfiling complete! Results in {output_dir}/")
```

**How to run**:
```bash
python3 [script] --model deepseek-coder-[file]
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
    easy = [r for r, c in zip(requests, complexities) if c < [file]]
    medium = [r for r, c in zip(requests, complexities) if [file] <= c < [file]]
    hard = [r for r, c in zip(requests, complexities) if c >= [file]]
    
    # Process each group with appropriate resources
    process_batch(easy, model='small', batch_size=64)
    process_batch(medium, model='medium', batch_size=32)
    process_batch(hard, model='large', batch_size=8)
```

### 2. Latency-Aware Batching

```python
def batch_by_latency_sla(requests):
    """Group by latency requirements."""
    
    latency_critical = [r for r in requests if [file] < 50]  # <50ms
    latency_sensitive = [r for r in requests if 50 <= [file] < 200]
    batch_requests = [r for r in requests if [file] >= 200]
    
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
pip install -r [file]

# Early exit
python3 [script]

# Dynamic routing
python3 [script]

# NVIDIA GPU profiling
python3 [script]
python3 [script]

# Comprehensive toolkit
python3 [script] --model deepseek-coder-[file]
```

---

## Key Takeaways

1. **Early exit saves compute**: 30-50% of requests can exit early → [file]-2x average speedup.

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

**Attention optimization** → [Chapter 18: Attention Mechanisms](.[executable]/[file])

Learn about:
- FlexAttention for flexible patterns
- FlashAttention for memory efficiency
- MLA (Multi-head Latent Attention) kernels

**Back to inference** → [Chapter 16: Inference Optimization](.[executable]/[file])

---

## Additional Resources

- **Early Exit**: [BERxiT Paper](https://[file]/abs/[file])
- **Adaptive Inference**: [Dynamic Neural Networks Survey](https://[file]/abs/[file])
- **Roofline Model**: [Roofline Paper](https://[file].[file]/~kubitron/cs252/handouts/papers/[file])

---

**Chapter Status**: [OK] Complete
