"""
Shared what-if scenario helpers (static for now, can be extended to dynamic sims).
"""

from __future__ import annotations

from typing import Dict, Any


def get_scenarios() -> Dict[str, Any]:
    scenarios = []

    scenarios.append({
        "id": "fp8",
        "name": "Use FP8 Precision",
        "description": "Switch from FP16/BF16 to FP8 for 2x compute throughput",
        "requirements": ["Blackwell GPU", "FP8-compatible model"],
        "estimated_speedup": 1.8,
        "memory_impact": -0.5,
        "accuracy_impact": "Minimal (<0.1% loss for inference)",
        "implementation_effort": "low",
        "code_example": "with torch.autocast(dtype=torch.float8_e4m3fn):\n    output = model(input)",
    })

    scenarios.append({
        "id": "flash_attention",
        "name": "Enable Flash Attention",
        "description": "Use memory-efficient attention for O(n) memory instead of O(n²)",
        "requirements": ["PyTorch 2.0+", "Attention-based model"],
        "estimated_speedup": 2.5,
        "memory_impact": -0.8,
        "accuracy_impact": "None (mathematically equivalent)",
        "implementation_effort": "low",
        "code_example": "from torch.nn.functional import scaled_dot_product_attention\noutput = scaled_dot_product_attention(q, k, v, is_causal=True)",
    })

    scenarios.append({
        "id": "torch_compile",
        "name": "Enable torch.compile",
        "description": "JIT compile model for kernel fusion and optimization",
        "requirements": ["PyTorch 2.0+"],
        "estimated_speedup": 1.4,
        "memory_impact": 0.1,
        "accuracy_impact": "None",
        "implementation_effort": "low",
        "code_example": "model = torch.compile(model, mode='max-autotune')",
    })

    scenarios.append({
        "id": "batch_size",
        "name": "Double Batch Size",
        "description": "Increase batch size for better GPU utilization",
        "requirements": ["Sufficient VRAM"],
        "estimated_speedup": 1.6,
        "memory_impact": 1.0,
        "accuracy_impact": "May need learning rate adjustment for training",
        "implementation_effort": "low",
        "code_example": "# Update dataloader\nbatch_size = current_batch_size * 2",
    })

    return {"scenarios": scenarios, "count": len(scenarios)}

