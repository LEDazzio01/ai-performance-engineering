#!/usr/bin/env python3
"""Analyze if optimizations applied are relevant to each chapter's learning objectives."""

from pathlib import Path
import re

# Chapter topics from README files
CHAPTER_TOPICS = {
    "ch1": {
        "topics": ["profiling", "pinned memory", "batching", "CUDA Graphs", "fundamental optimizations"],
        "relevant_optimizations": ["torch.compile", "cuDNN benchmarking", "FP16", "TF32"],
        "notes": "Fundamental optimizations - all optimizations are relevant"
    },
    "ch2": {
        "topics": ["hardware architecture", "NVLink", "memory hierarchy", "HBM3e", "hardware specs"],
        "relevant_optimizations": ["TF32", "FP16"],  # Demonstrating hardware capabilities
        "irrelevant_optimizations": ["torch.compile", "cuDNN benchmarking"],
        "notes": "Hardware-focused - PyTorch optimizations less relevant"
    },
    "ch3": {
        "topics": ["NUMA binding", "CPU affinity", "Docker", "Kubernetes", "system tuning"],
        "relevant_optimizations": [],  # System-level, not PyTorch
        "irrelevant_optimizations": ["torch.compile", "cuDNN benchmarking", "FP16", "TF32"],
        "notes": "System-level tuning - PyTorch optimizations not relevant"
    },
    "ch4": {
        "topics": ["DDP", "NCCL", "tensor parallelism", "pipeline parallelism", "multi-GPU"],
        "relevant_optimizations": ["torch.compile"],  # Can help with multi-GPU
        "irrelevant_optimizations": ["cuDNN benchmarking", "FP16"],  # Less focus on these
        "notes": "Multi-GPU communication - some optimizations relevant"
    },
    "ch6": {
        "topics": ["CUDA kernels", "thread hierarchy", "occupancy", "ILP", "warp divergence"],
        "relevant_optimizations": ["FP16"],  # If demonstrating precision
        "irrelevant_optimizations": ["torch.compile", "cuDNN benchmarking"],
        "notes": "Low-level CUDA - PyTorch optimizations less relevant"
    },
    "ch10": {
        "topics": ["Tensor Cores", "TMA", "async pipelines", "warp specialization"],
        "relevant_optimizations": ["TF32", "FP16", "torch.compile"],
        "notes": "Tensor Cores - precision optimizations very relevant!"
    },
    "ch13": {
        "topics": ["PyTorch profiling", "compiled autograd", "FSDP", "mixed precision"],
        "relevant_optimizations": ["torch.compile", "cuDNN benchmarking", "FP16", "TF32"],
        "notes": "PyTorch optimization - all optimizations very relevant!"
    },
    "ch19": {
        "topics": ["FP4", "FP6", "FP8", "quantization", "low-precision"],
        "relevant_optimizations": ["FP16"],  # Related to precision
        "irrelevant_optimizations": ["torch.compile"],  # Less focus on compilation
        "notes": "Low-precision formats - precision optimizations relevant"
    },
}

def check_file_optimizations(file_path: Path) -> dict:
    """Check what optimizations are in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        optimizations = {
            "torch.compile": "torch.compile" in content,
            "cuDNN benchmarking": "torch.backends.cudnn.benchmark" in content,
            "TF32": "torch.backends.cuda.matmul.allow_tf32" in content or "torch.backends.cudnn.allow_tf32" in content,
            "FP16": ".half()" in content or "torch.float16" in content,
        }
        return optimizations
    except Exception:
        return {}

def analyze_chapter(chapter_dir: Path) -> dict:
    """Analyze a chapter's optimized files."""
    chapter_name = chapter_dir.name
    chapter_info = CHAPTER_TOPICS.get(chapter_name, {})
    
    optimized_files = list(chapter_dir.glob("optimized_*.py"))
    
    results = {
        "chapter": chapter_name,
        "topics": chapter_info.get("topics", []),
        "relevant": chapter_info.get("relevant_optimizations", []),
        "irrelevant": chapter_info.get("irrelevant_optimizations", []),
        "files": [],
        "issues": []
    }
    
    for file_path in optimized_files:
        optimizations = check_file_optimizations(file_path)
        file_results = {
            "file": file_path.name,
            "optimizations": optimizations,
            "has_irrelevant": False,
            "missing_relevant": []
        }
        
        # Check for irrelevant optimizations
        for opt in results["irrelevant"]:
            if optimizations.get(opt.replace("torch.compile", "torch.compile").replace("cuDNN benchmarking", "cuDNN benchmarking").replace("FP16", "FP16").replace("TF32", "TF32"), False):
                file_results["has_irrelevant"] = True
                file_results["irrelevant_found"] = opt
        
        # Check for missing relevant optimizations
        for opt in results["relevant"]:
            opt_key = opt.replace("torch.compile", "torch.compile").replace("cuDNN benchmarking", "cuDNN benchmarking").replace("FP16", "FP16").replace("TF32", "TF32")
            if not optimizations.get(opt_key, False):
                file_results["missing_relevant"].append(opt)
        
        if file_results["has_irrelevant"] or file_results["missing_relevant"]:
            results["files"].append(file_results)
            if file_results["has_irrelevant"]:
                results["issues"].append(f"{file_path.name}: Has irrelevant optimization '{file_results['irrelevant_found']}'")
            if file_results["missing_relevant"]:
                results["issues"].append(f"{file_path.name}: Missing relevant optimizations: {', '.join(file_results['missing_relevant'])}")
    
    return results

def main():
    """Main analysis."""
    repo_root = Path(__file__).parent.parent
    
    print("=" * 80)
    print("OPTIMIZATION RELEVANCE ANALYSIS")
    print("=" * 80)
    print()
    
    all_issues = []
    
    for chapter_dir in sorted(repo_root.glob("ch*")):
        if not chapter_dir.is_dir() or chapter_dir.name not in CHAPTER_TOPICS:
            continue
        
        results = analyze_chapter(chapter_dir)
        
        if results["issues"]:
            print(f"\n{results['chapter'].upper()}")
            print(f"Topics: {', '.join(results['topics'])}")
            print(f"Relevant optimizations: {', '.join(results['relevant']) if results['relevant'] else 'None'}")
            print(f"Irrelevant optimizations: {', '.join(results['irrelevant']) if results['irrelevant'] else 'None'}")
            print(f"Issues found: {len(results['issues'])}")
            for issue in results["issues"][:5]:  # Show first 5
                print(f"  - {issue}")
            if len(results["issues"]) > 5:
                print(f"  ... and {len(results['issues']) - 5} more")
            all_issues.extend(results["issues"])
        else:
            print(f"✓ {results['chapter']}: All optimizations are relevant")
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: Found {len(all_issues)} potential relevance issues")
    print("=" * 80)
    
    if all_issues:
        print("\nRecommendations:")
        print("1. Review Chapter 3 (System Tuning) - PyTorch optimizations may not be relevant")
        print("2. Review Chapter 2 (Hardware) - Focus should be on hardware capabilities, not PyTorch")
        print("3. Review Chapter 6 (CUDA Basics) - Low-level CUDA may not need PyTorch optimizations")
        print("4. Consider removing irrelevant optimizations or documenting why they're included")

if __name__ == "__main__":
    main()

