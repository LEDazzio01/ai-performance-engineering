"""
Inference Commands - vLLM, quantization, deployment configuration.

Uses tools/optimization_intelligence.py for vLLM config generation.
"""

from __future__ import annotations

import json

from tools.core.engine import get_engine


def _print_header(title: str, emoji: str = "🚀"):
    print(f"\n{emoji} {title}")
    print("=" * 70)


def vllm_config(args) -> int:
    """Generate optimized vLLM configuration."""
    _print_header("vLLM Configuration", "⚡")
    
    model = getattr(args, 'model', 'meta-llama/Llama-2-70b-hf')
    target = getattr(args, 'target', 'balanced')
    
    print(f"  Model: {model}")
    print(f"  Target: {target}")
    print("-" * 70)
    
    try:
        engine = get_engine()
        result = engine.inference.vllm_config(
            model=model,
            target=target,
        )
        
        if isinstance(result, dict):
            print("\n  Generated Configuration:")
            config = result.get("config", result)
            if isinstance(config, dict):
                for key, value in config.items():
                    if key not in ("launch_command", "expected_performance"):
                        print(f"    {key}: {value}")
            
            if "launch_command" in result:
                print("\n  Launch Command:")
                print(f"    {result['launch_command']}")
            
            perf = result.get("expected_performance", {})
            if perf:
                print(f"\n  Expected Performance:")
                if "throughput_tokens_per_sec" in perf:
                    print(f"    Throughput: ~{perf['throughput_tokens_per_sec']:.0f} tokens/sec")
                if "ttft_ms" in perf:
                    print(f"    TTFT: ~{perf['ttft_ms']:.0f} ms")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    return 0


def quantize(args) -> int:
    """Quantization recommendations and configuration."""
    _print_header("Quantization", "🔢")
    
    model_size = getattr(args, 'model_size', 70)
    target_memory = getattr(args, 'target_memory', None)
    
    print(f"  Model Size: {model_size}B parameters")
    
    try:
        result = get_engine().inference.quantization({"model_size": model_size, "target_memory_gb": target_memory})
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        print(f"  Error: {e}")
        return 1


def deploy_config(args) -> int:
    """Generate deployment configuration."""
    _print_header("Deployment Configuration", "📦")
    
    target = getattr(args, "target", "vllm")
    model = getattr(args, "model", "meta-llama/Llama-2-70b-hf")
    try:
        result = get_engine().inference.deploy_config({"target": target, "model": model})
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        print(f"  Error: {e}")
        return 1


def serve(args) -> int:
    """Start inference server."""
    _print_header("Inference Server", "🖥️")
    
    model = getattr(args, "model", "meta-llama/Llama-2-70b-hf")
    gpus = getattr(args, "gpus", 1)
    try:
        status = get_engine().inference.status()
        print("  Inference backend status:")
        print(json.dumps(status, indent=2))
        print("\n  Suggested launch (vLLM):")
        print(f"    python -m vllm.entrypoints.openai.api_server --model {model} --tensor-parallel-size {gpus}")
    except Exception as e:
        print(f"  Error: {e}")
        return 1
    return 0
