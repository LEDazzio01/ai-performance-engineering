"""
AI Assistant Commands - LLM-powered Q&A, explanations, troubleshooting.

All commands that use the LLM backend for intelligent assistance.
Includes book citations from AI Systems Performance Engineering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, List


def _print_header(title: str, emoji: str = "🤖"):
    """Print a formatted header."""
    print(f"\n{emoji} {title}")
    print("=" * 70)


def _get_book_citations(query: str, max_results: int = 2) -> str:
    """Get relevant citations from the book."""
    try:
        from tools.core.book import get_citations, format_citations
        citations = get_citations(query, max_results)
        if citations:
            return format_citations(citations)
    except ImportError:
        pass
    except Exception as e:
        pass
    return ""


def _get_llm_engine():
    """Get LLM engine with proper error handling."""
    try:
        from tools.llm_engine import PerformanceAnalysisEngine
        return PerformanceAnalysisEngine()
    except RuntimeError as e:
        print(f"⚠️ LLM not available: {e}")
        print("\nTo enable LLM features:")
        print("  Option 1: Set API key in .env")
        print("    echo 'OPENAI_API_KEY=sk-...' >> .env")
        print("    # or")
        print("    echo 'ANTHROPIC_API_KEY=...' >> .env")
        print("")
        print("  Option 2: Start Ollama locally")
        print("    ollama serve")
        print("    ollama pull qwen2.5:7b")
        return None


# =============================================================================
# ASK QUESTION
# =============================================================================

def ask_question(args) -> int:
    """Ask a performance-related question with book citations."""
    _print_header("AI Performance Assistant", "🧠")
    
    # Get question from args or prompt
    if hasattr(args, 'question') and args.question:
        question = ' '.join(args.question)
    else:
        print("Enter your question (or 'quit' to exit):")
        question = input("> ").strip()
        if question.lower() in ('quit', 'exit', 'q'):
            return 0
    
    if not question:
        print("No question provided.")
        return 1
    
    print(f"\n  Question: {question}")
    print("-" * 70)
    
    # Get book citations first
    no_book = getattr(args, 'no_book', False)
    if not no_book:
        print("\n  Searching book for relevant references...")
        citations = _get_book_citations(question)
        if citations:
            print(citations)
    
    engine = _get_llm_engine()
    if not engine:
        return 1
    
    print("\n  🤖 LLM Analysis:\n")
    
    try:
        response = engine.ask(question)
        print(response)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    print()
    return 0


# =============================================================================
# EXPLAIN CONCEPT
# =============================================================================

def explain_concept(args) -> int:
    """Explain a performance concept with book references + LLM."""
    _print_header("Concept Explanation", "📚")
    
    # Common concepts we can explain
    concepts = {
        "flash-attention": "Flash Attention algorithm and memory efficiency",
        "tensor-parallelism": "Tensor Parallelism (TP) for distributed training",
        "pipeline-parallelism": "Pipeline Parallelism (PP) and micro-batching",
        "fsdp": "Fully Sharded Data Parallel training",
        "fp8": "FP8 precision training on Hopper/Blackwell",
        "cuda-graphs": "CUDA Graphs for reducing launch overhead",
        "kv-cache": "KV Cache for autoregressive inference",
        "continuous-batching": "Continuous batching for inference serving",
        "moe": "Mixture of Experts architecture and routing",
        "speculative-decoding": "Speculative decoding for faster inference",
        "occupancy": "GPU occupancy and SM utilization",
        "coalescing": "Memory coalescing for efficient access",
        "bank-conflicts": "Shared memory bank conflicts",
        "warp-divergence": "Warp divergence and branch efficiency",
        "triton": "Triton kernels and custom GPU code",
        "nccl": "NCCL collective communication",
        "zero": "ZeRO optimizer memory optimization",
        "quantization": "Model quantization (FP8/INT8/INT4)",
    }
    
    if hasattr(args, 'concept'):
        concept = args.concept
    else:
        print("Available concepts:")
        for key, desc in concepts.items():
            print(f"  {key:25} - {desc}")
        print("\nEnter concept name:")
        concept = input("> ").strip()
    
    if not concept:
        return 1
    
    print(f"\n  Explaining: {concept}")
    print("-" * 70)
    
    # Get book citations FIRST
    print("\n  📖 Book References:")
    citations = _get_book_citations(concept, max_results=3)
    if citations:
        print(citations)
    else:
        print("  No specific book section found for this concept.\n")
    
    # Then get LLM explanation
    engine = _get_llm_engine()
    if not engine:
        print("  (LLM not available for extended explanation)")
        return 0
    
    print("\n  🤖 LLM Explanation:\n")
    
    prompt = f"""Explain the concept of "{concept}" in the context of GPU performance optimization.

Include:
1. What it is and why it matters
2. When to use it
3. Key parameters/configurations
4. Common pitfalls
5. A simple code example if applicable

Keep the explanation practical and actionable for an ML engineer."""

    try:
        response = engine.ask(prompt)
        print(response)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    print()
    return 0


# =============================================================================
# TROUBLESHOOT
# =============================================================================

def troubleshoot(args) -> int:
    """Diagnose and help fix issues."""
    _print_header("Troubleshooting Assistant", "🔧")
    
    # Common issues
    common_issues = [
        "CUDA out of memory",
        "NCCL timeout",
        "Slow training throughput",
        "High GPU memory fragmentation",
        "torch.compile errors",
        "Flash Attention not working",
    ]
    
    if hasattr(args, 'issue') and args.issue:
        issue = ' '.join(args.issue)
    else:
        print("Common issues:")
        for i, iss in enumerate(common_issues, 1):
            print(f"  {i}. {iss}")
        print("\nDescribe your issue (or enter number):")
        user_input = input("> ").strip()
        
        if user_input.isdigit() and 1 <= int(user_input) <= len(common_issues):
            issue = common_issues[int(user_input) - 1]
        else:
            issue = user_input
    
    if not issue:
        return 1
    
    print(f"\n  Issue: {issue}")
    print("-" * 70)
    
    # Collect system context
    print("\n  Collecting system context...")
    
    context = {}
    try:
        import torch
        context["pytorch_version"] = torch.__version__
        context["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            context["gpu_name"] = torch.cuda.get_device_name(0)
            context["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    except ImportError:
        pass
    
    engine = _get_llm_engine()
    if not engine:
        return 1
    
    prompt = f"""I'm experiencing this issue with my GPU workload:

Issue: {issue}

System context:
{json.dumps(context, indent=2)}

Please help me:
1. Understand what's causing this issue
2. Provide specific steps to diagnose it
3. Suggest solutions in order of likelihood
4. Include any relevant code fixes or configuration changes"""

    print("\n  Analyzing...\n")
    
    try:
        response = engine.ask(prompt)
        print(response)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    print()
    return 0


# =============================================================================
# LLM STATUS
# =============================================================================

def llm_status(args) -> int:
    """Check LLM backend status and configuration."""
    _print_header("LLM Backend Status", "🔌")
    
    import os
    
    # Check environment
    print("\n  Environment Variables:")
    env_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "PERF_LLM_PROVIDER",
        "PERF_LLM_MODEL",
        "PERF_LLM_BASE_URL",
    ]
    
    for var in env_vars:
        value = os.environ.get(var, "")
        if "KEY" in var and value:
            value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
        print(f"    {var:25} = {value or '(not set)'}")
    
    # Check backends
    print("\n  Backend Availability:")
    
    # Ollama
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                import json
                data = json.loads(resp.read().decode())
                models = [m["name"] for m in data.get("models", [])]
                print(f"    ✅ Ollama: Running ({len(models)} models)")
                if models:
                    print(f"       Models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
            else:
                print("    ❌ Ollama: Not responding")
    except Exception:
        print("    ❌ Ollama: Not running")
    
    # vLLM local
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8000/v1/models")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                print("    ✅ Local vLLM: Running")
    except Exception:
        print("    ❌ Local vLLM: Not running")
    
    # API providers
    if os.environ.get("OPENAI_API_KEY"):
        print("    ✅ OpenAI: API key configured")
    else:
        print("    ❌ OpenAI: No API key")
    
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("    ✅ Anthropic: API key configured")
    else:
        print("    ❌ Anthropic: No API key")
    
    # Test connection
    print("\n  Connection Test:")
    engine = _get_llm_engine()
    if engine:
        print(f"    ✅ Connected to: {engine.config.provider}")
        print(f"       Model: {engine.config.model}")
    else:
        print("    ❌ No LLM backend available")
    
    print()
    return 0

