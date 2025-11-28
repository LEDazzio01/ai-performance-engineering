"""
HuggingFace Integration Commands - Model search, trending, configurations.

Provides commands for:
- Searching HuggingFace models
- Viewing trending models
- Getting model configurations and requirements
"""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
from typing import Optional, Dict, Any, List


def _print_header(title: str, emoji: str = "🤗"):
    print(f"\n{emoji} {title}")
    print("=" * 70)


def _hf_api_request(endpoint: str, params: Optional[Dict] = None) -> Dict:
    """Make a request to HuggingFace API."""
    base_url = "https://huggingface.co/api"
    url = f"{base_url}/{endpoint}"
    
    if params:
        url += "?" + urllib.parse.urlencode(params)
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "aisp-cli/2.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


# Model size estimates (parameters in billions)
MODEL_SIZES = {
    "llama": {"7b": 7, "13b": 13, "30b": 30, "65b": 65, "70b": 70},
    "mistral": {"7b": 7},
    "mixtral": {"8x7b": 47, "8x22b": 141},
    "qwen": {"0.5b": 0.5, "1.8b": 1.8, "7b": 7, "14b": 14, "72b": 72},
    "phi": {"2": 2.7, "3": 3.8},
    "gemma": {"2b": 2, "7b": 7, "9b": 9, "27b": 27},
    "falcon": {"7b": 7, "40b": 40, "180b": 180},
}


def search_models(args) -> int:
    """Search HuggingFace for models."""
    _print_header("HuggingFace Model Search", "🔍")
    
    query = getattr(args, 'query', None)
    if not query:
        print("  Usage: aisp hf search <query>")
        print("\n  Examples:")
        print("    aisp hf search llama")
        print("    aisp hf search 'code generation'")
        print("    aisp hf search mistral --task text-generation")
        return 1
    
    task = getattr(args, 'task', None)
    limit = getattr(args, 'limit', 10)
    
    print(f"  Searching for: {query}")
    
    params = {
        "search": query,
        "limit": limit,
        "sort": "downloads",
        "direction": "-1",
    }
    if task:
        params["pipeline_tag"] = task
    
    results = _hf_api_request("models", params)
    
    if "error" in results:
        print(f"  ❌ Error: {results['error']}")
        return 1
    
    if not results:
        print("  No models found.")
        return 0
    
    print(f"\n  Found {len(results)} models:\n")
    print(f"  {'Model':<45} {'Downloads':>12} {'Likes':>8}")
    print("-" * 70)
    
    for model in results:
        name = model.get('id', 'unknown')[:44]
        downloads = model.get('downloads', 0)
        likes = model.get('likes', 0)
        
        # Format downloads
        if downloads >= 1_000_000:
            dl_str = f"{downloads/1_000_000:.1f}M"
        elif downloads >= 1_000:
            dl_str = f"{downloads/1_000:.1f}K"
        else:
            dl_str = str(downloads)
        
        print(f"  {name:<45} {dl_str:>12} {likes:>8}")
    
    print(f"\n  View model: aisp hf model <model_id>")
    return 0


def trending_models(args) -> int:
    """Show trending models on HuggingFace."""
    _print_header("Trending Models", "📈")
    
    task = getattr(args, 'task', 'text-generation')
    limit = getattr(args, 'limit', 15)
    
    print(f"  Task: {task}")
    print(f"  Showing top {limit} by recent downloads\n")
    
    params = {
        "pipeline_tag": task,
        "sort": "downloads",
        "direction": "-1",
        "limit": limit,
    }
    
    results = _hf_api_request("models", params)
    
    if "error" in results:
        print(f"  ❌ Error: {results['error']}")
        return 1
    
    print(f"  {'#':<3} {'Model':<45} {'Downloads':>12}")
    print("-" * 65)
    
    for i, model in enumerate(results, 1):
        name = model.get('id', 'unknown')[:44]
        downloads = model.get('downloads', 0)
        
        if downloads >= 1_000_000:
            dl_str = f"{downloads/1_000_000:.1f}M"
        elif downloads >= 1_000:
            dl_str = f"{downloads/1_000:.1f}K"
        else:
            dl_str = str(downloads)
        
        print(f"  {i:<3} {name:<45} {dl_str:>12}")
    
    print(f"\n  Other tasks: --task [text-classification, summarization, translation, ...]")
    return 0


def model_info(args) -> int:
    """Get detailed model information and requirements."""
    _print_header("Model Information", "📋")
    
    model_id = getattr(args, 'model', None)
    if not model_id:
        print("  Usage: aisp hf model <model_id>")
        print("\n  Examples:")
        print("    aisp hf model meta-llama/Llama-2-70b-hf")
        print("    aisp hf model mistralai/Mistral-7B-v0.1")
        return 1
    
    print(f"  Fetching: {model_id}")
    
    result = _hf_api_request(f"models/{model_id}")
    
    if "error" in result:
        print(f"  ❌ Error: {result['error']}")
        return 1
    
    print(f"\n  Model: {result.get('id', 'unknown')}")
    print(f"  Author: {result.get('author', 'unknown')}")
    print(f"  Downloads: {result.get('downloads', 0):,}")
    print(f"  Likes: {result.get('likes', 0):,}")
    print(f"  Task: {result.get('pipeline_tag', 'unknown')}")
    
    # Tags
    tags = result.get('tags', [])
    if tags:
        print(f"  Tags: {', '.join(tags[:10])}")
    
    # Estimate model size
    model_lower = model_id.lower()
    estimated_size = None
    for family, sizes in MODEL_SIZES.items():
        if family in model_lower:
            for size_str, size_b in sizes.items():
                if size_str in model_lower:
                    estimated_size = size_b
                    break
    
    if estimated_size:
        print(f"\n  📊 Estimated Size: {estimated_size}B parameters")
        
        # Memory requirements
        fp32_mem = estimated_size * 4
        fp16_mem = estimated_size * 2
        int8_mem = estimated_size * 1
        int4_mem = estimated_size * 0.5
        
        print(f"\n  Memory Requirements (inference):")
        print(f"    FP32:  {fp32_mem:.0f} GB")
        print(f"    FP16:  {fp16_mem:.0f} GB")
        print(f"    INT8:  {int8_mem:.0f} GB")
        print(f"    INT4:  {int4_mem:.0f} GB")
        
        # GPU recommendations
        print(f"\n  GPU Recommendations:")
        if int4_mem <= 24:
            print(f"    ✅ Fits on single RTX 4090 (24GB) with INT4")
        if fp16_mem <= 80:
            print(f"    ✅ Fits on single A100/H100 (80GB) with FP16")
        if fp16_mem <= 180:
            print(f"    ✅ Fits on single B200 (180GB) with FP16")
        
        if fp16_mem > 80:
            tp_size = (fp16_mem // 80) + 1
            print(f"    💡 Recommended TP size for H100: {tp_size}")
    
    # Config link
    print(f"\n  🔗 https://huggingface.co/{model_id}")
    
    return 0


def model_config(args) -> int:
    """Generate optimal configuration for a model."""
    _print_header("Optimal Model Configuration", "⚙️")
    
    model_id = getattr(args, 'model', None)
    gpus = getattr(args, 'gpus', 1)
    
    if not model_id:
        print("  Usage: aisp hf config <model_id> [--gpus N]")
        return 1
    
    print(f"  Model: {model_id}")
    print(f"  GPUs: {gpus}")
    
    # Estimate size from model name
    model_lower = model_id.lower()
    estimated_size = 7  # default
    
    for family, sizes in MODEL_SIZES.items():
        if family in model_lower:
            for size_str, size_b in sizes.items():
                if size_str in model_lower:
                    estimated_size = size_b
                    break
    
    print(f"\n  Estimated size: {estimated_size}B parameters")
    
    # Generate configuration
    fp16_mem = estimated_size * 2
    
    print(f"\n  📋 Recommended Configuration:")
    print("-" * 70)
    
    # Determine precision
    if gpus >= 4 or estimated_size <= 13:
        precision = "bfloat16"
    elif estimated_size <= 30:
        precision = "float16"
    else:
        precision = "float8"  # FP8 for very large models
    
    print(f"    precision: {precision}")
    
    # Determine parallelism
    mem_per_gpu = 80  # Assume H100
    if fp16_mem / gpus <= mem_per_gpu * 0.85:
        tp = gpus
        pp = 1
    else:
        tp = min(gpus, 8)
        pp = gpus // tp
    
    print(f"    tensor_parallel: {tp}")
    print(f"    pipeline_parallel: {pp}")
    
    # vLLM config
    print(f"\n  🚀 vLLM Launch Command:")
    print(f"    python -m vllm.entrypoints.openai.api_server \\")
    print(f"      --model {model_id} \\")
    print(f"      --tensor-parallel-size {tp} \\")
    print(f"      --dtype {precision} \\")
    print(f"      --max-model-len 8192 \\")
    print(f"      --gpu-memory-utilization 0.9")
    
    return 0


# =============================================================================
# COMMAND REGISTRATION
# =============================================================================

def register_commands(subparsers):
    """Register HuggingFace commands."""
    hf_parser = subparsers.add_parser("hf", help="HuggingFace model integration")
    hf_subparsers = hf_parser.add_subparsers(dest="hf_command")
    
    # Search
    search_p = hf_subparsers.add_parser("search", help="Search for models")
    search_p.add_argument("query", nargs="?", help="Search query")
    search_p.add_argument("--task", help="Filter by task")
    search_p.add_argument("--limit", type=int, default=10, help="Max results")
    search_p.set_defaults(func=search_models)
    
    # Trending
    trend_p = hf_subparsers.add_parser("trending", help="Show trending models")
    trend_p.add_argument("--task", default="text-generation", help="Task type")
    trend_p.add_argument("--limit", type=int, default=15, help="Max results")
    trend_p.set_defaults(func=trending_models)
    
    # Model info
    model_p = hf_subparsers.add_parser("model", help="Get model information")
    model_p.add_argument("model", nargs="?", help="Model ID")
    model_p.set_defaults(func=model_info)
    
    # Config
    config_p = hf_subparsers.add_parser("config", help="Generate optimal config")
    config_p.add_argument("model", nargs="?", help="Model ID")
    config_p.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    config_p.set_defaults(func=model_config)


