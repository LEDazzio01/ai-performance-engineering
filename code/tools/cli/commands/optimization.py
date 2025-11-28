"""
Optimization Commands - Recommendations, auto-optimization, what-if analysis.

Uses tools/optimization_intelligence.py and tools/llm_engine.py.
"""

from __future__ import annotations

import json


def _print_header(title: str, emoji: str = "⚡"):
    print(f"\n{emoji} {title}")
    print("=" * 70)


def recommend(args) -> int:
    """Get optimization recommendations for current setup."""
    _print_header("Optimization Recommendations", "💡")
    
    model_size = getattr(args, 'model_size', None)
    num_gpus = getattr(args, 'gpus', 1)
    goal = getattr(args, 'goal', 'throughput')
    
    print(f"  Model Size: {model_size or 'auto-detect'}B")
    print(f"  GPUs: {num_gpus}")
    print(f"  Goal: {goal}")
    print("-" * 70)
    
    try:
        from tools.core.engine import get_engine
        
        engine = get_engine()
        result = engine.optimize.recommend(
            model_size=model_size or 7,
            gpus=num_gpus,
            goal=goal,
        )
        
        if result.get("success"):
            print("\n  Recommended Techniques:")
            for i, tech in enumerate(result.get("techniques", []), 1):
                print(f"    {i}. {tech}")
            
            speedup = result.get("estimated_speedup", (1, 1))
            if isinstance(speedup, (list, tuple)):
                print(f"\n  Expected Speedup: {speedup[0]:.1f}x - {speedup[1]:.1f}x")
            else:
                print(f"\n  Expected Speedup: {speedup:.1f}x")
            print(f"  Memory Reduction: {result.get('estimated_memory_reduction', 0):.0f}%")
            print(f"  Confidence: {result.get('confidence', 0):.0%}")
            
            rationale = result.get("rationale", "")
            if rationale:
                print(f"\n  Rationale:")
                print(f"    {rationale[:200]}...")
            
            steps = result.get("implementation_steps", [])
            if steps:
                print("\n  Implementation Steps:")
                for step in steps[:5]:
                    print(f"    • {step}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"  Error: {e}")
        return 1
    
    print()
    return 0


def auto_optimize(args) -> int:
    """Auto-optimize with LLM guidance."""
    _print_header("Auto-Optimization", "🤖")
    
    try:
        from tools.core.engine import get_engine
        
        engine = get_engine()
        print("  Collecting system data...")
        
        # Get recommendations based on current system
        result = engine.optimize.recommend(
            model_size=getattr(args, 'model_size', 7),
            gpus=getattr(args, 'gpus', 1),
            goal=getattr(args, 'goal', 'throughput')
        )
        
        if result.get("success"):
            print("\n  Auto-optimization recommendations:")
            for i, tech in enumerate(result.get("techniques", []), 1):
                print(f"    {i}. {tech}")
            
            steps = result.get("implementation_steps", [])
            if steps:
                print("\n  Implementation Guide:")
                for step in steps:
                    print(f"    • {step}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"  Error: {e}")
        return 1
    
    return 0


def whatif(args) -> int:
    """What-if analysis for optimizations."""
    _print_header("What-If Analysis", "🔮")
    
    try:
        from tools.core.engine import get_engine

        engine = get_engine()
        result = engine.optimize.whatif({})
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        print(f"  Error: {e}")
        return 1


def stacking(args) -> int:
    """Compound optimization stacking analysis."""
    _print_header("Optimization Stacking", "📚")
    
    try:
        from tools.core.engine import get_engine

        engine = get_engine()
        result = engine.optimize.stacking()
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"  Error: {e}")
    
    return 0


def playbook(args) -> int:
    """Pre-built optimization playbooks."""
    _print_header("Optimization Playbooks", "📋")
    
    action = getattr(args, 'action', 'list')
    try:
        from tools.core.engine import get_engine

        engine = get_engine()
        playbooks = engine.optimize.playbook()

        if action == 'list':
            print("\n  Available Playbooks:")
            for key, pb in playbooks.items():
                print(f"    {key:25} - {pb.get('name', key)}")
            print("\n  Usage: aisp optimize playbook <name>")
        else:
            name = action
            pb = playbooks.get(name)
            if pb:
                print(f"\n  Playbook: {pb.get('name', name)}")
                print("-" * 50)
                for i, step in enumerate(pb.get('steps', []), 1):
                    print(f"    {i}. {step}")
            else:
                print(f"  Unknown playbook: {name}")
                return 1
        return 0
    except Exception as e:
        print(f"  Error: {e}")
        return 1
