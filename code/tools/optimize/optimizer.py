"""
AutoOptimizer - Core optimization engine.

Profiles GPU code, uses LLM to analyze bottlenecks, generates and validates optimizations.
"""

import ast
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    success: bool
    original_code: str
    optimized_code: str
    original_time_ms: float
    optimized_time_ms: float
    speedup: float
    techniques_applied: List[str]
    explanation: str
    profile_data: Dict[str, Any]
    patches_applied: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class ProfileResult:
    """Profiling results for GPU code."""
    total_time_ms: float
    gpu_time_ms: float
    cpu_time_ms: float
    memory_peak_mb: float
    memory_allocated_mb: float
    kernel_times: Dict[str, float]
    bottlenecks: List[str]
    trace_path: Optional[Path] = None
    memory_snapshot_path: Optional[Path] = None
    flame_graph_data: Optional[Dict] = None


class AutoOptimizer:
    """
    Automatic GPU code optimizer.
    
    Profiles code, identifies bottlenecks, uses LLM to suggest optimizations,
    applies patches, and validates performance improvements.
    """
    
    def __init__(
        self,
        llm_provider: str = "anthropic",
        model: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        verbose: bool = True,
        max_iterations: int = 3,
        target_speedup: float = 1.2,
    ):
        self.llm_provider = llm_provider
        self.model = model or ("claude-sonnet-4-20250514" if llm_provider == "anthropic" else "gpt-4o")
        self.cache_dir = cache_dir or Path.home() / ".cache" / "auto-optimizer"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.target_speedup = target_speedup
        
        # Import analysis tools
        from tools.analysis.llm_profile_analyzer import LLMProfileAnalyzer
        from tools.analysis.llm_patch_applier import LLMPatchApplier
        
        self.analyzer = LLMProfileAnalyzer(provider=llm_provider, model=self.model)
        self.patch_applier = LLMPatchApplier()
    
    def optimize_file(
        self,
        file_path: Path,
        output_path: Optional[Path] = None,
        benchmark_fn: Optional[str] = None,
        setup_code: Optional[str] = None,
    ) -> OptimizationResult:
        """
        Optimize a single Python file containing GPU code.
        
        Args:
            file_path: Path to the Python file to optimize
            output_path: Where to write the optimized code (None = don't write)
            benchmark_fn: Name of function to benchmark (auto-detected if None)
            setup_code: Code to run before benchmarking
            
        Returns:
            OptimizationResult with before/after comparison
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        original_code = file_path.read_text()
        
        if self.verbose:
            print(f"🔍 Analyzing: {file_path}")
        
        # Step 1: Profile the original code
        profile = self._profile_code(original_code, benchmark_fn, setup_code)
        
        if self.verbose:
            print(f"   ⏱️  Original time: {profile.total_time_ms:.2f}ms")
            print(f"   🎯 Bottlenecks: {', '.join(profile.bottlenecks[:3])}")
        
        # Step 2: Use LLM to analyze and suggest optimizations
        optimized_code = original_code
        all_patches = []
        all_techniques = []
        all_errors = []
        best_time = profile.total_time_ms
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n🔄 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Get LLM suggestions
            analysis = self._analyze_with_llm(optimized_code, profile)
            
            if not analysis or "patches" not in analysis:
                if self.verbose:
                    print("   ⚠️  No patches suggested")
                break
            
            # Apply patches
            patched_code, patch_results = self._apply_patches(
                optimized_code, 
                analysis["patches"]
            )
            
            if not any(r["success"] for r in patch_results):
                all_errors.extend([r["error"] for r in patch_results if r.get("error")])
                if self.verbose:
                    print("   ❌ All patches failed")
                continue
            
            # Validate the patched code
            new_profile = self._profile_code(patched_code, benchmark_fn, setup_code)
            
            if new_profile.total_time_ms < best_time:
                improvement = (best_time - new_profile.total_time_ms) / best_time * 100
                if self.verbose:
                    print(f"   ✅ Improvement: {improvement:.1f}% ({best_time:.2f}ms → {new_profile.total_time_ms:.2f}ms)")
                
                optimized_code = patched_code
                best_time = new_profile.total_time_ms
                all_patches.extend([r for r in patch_results if r["success"]])
                all_techniques.extend(analysis.get("techniques", []))
                profile = new_profile
                
                # Check if we've reached target speedup
                current_speedup = profile.total_time_ms / new_profile.total_time_ms
                if current_speedup >= self.target_speedup:
                    if self.verbose:
                        print(f"   🎉 Target speedup reached: {current_speedup:.2f}x")
                    break
            else:
                if self.verbose:
                    print(f"   ⚠️  No improvement (was {best_time:.2f}ms, now {new_profile.total_time_ms:.2f}ms)")
        
        # Calculate final speedup
        original_time = self._profile_code(original_code, benchmark_fn, setup_code).total_time_ms
        speedup = original_time / best_time if best_time > 0 else 1.0
        
        # Generate explanation
        explanation = self._generate_explanation(
            original_code, optimized_code, all_techniques, speedup
        )
        
        # Write output if requested
        if output_path and optimized_code != original_code:
            output_path = Path(output_path)
            output_path.write_text(optimized_code)
            if self.verbose:
                print(f"\n📝 Wrote optimized code to: {output_path}")
        
        return OptimizationResult(
            success=speedup > 1.0,
            original_code=original_code,
            optimized_code=optimized_code,
            original_time_ms=original_time,
            optimized_time_ms=best_time,
            speedup=speedup,
            techniques_applied=list(set(all_techniques)),
            explanation=explanation,
            profile_data=profile.__dict__,
            patches_applied=all_patches,
            errors=all_errors,
        )
    
    def optimize_repo(
        self,
        repo_url: str,
        target_files: Optional[List[str]] = None,
        branch: str = "main",
        output_dir: Optional[Path] = None,
    ) -> Dict[str, OptimizationResult]:
        """
        Clone and optimize a GitHub repository.
        
        Args:
            repo_url: GitHub repository URL
            target_files: Specific files to optimize (None = auto-detect GPU code)
            branch: Branch to checkout
            output_dir: Where to write optimized files
            
        Returns:
            Dict mapping file paths to optimization results
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            if self.verbose:
                print(f"📦 Cloning {repo_url}...")
            
            # Clone the repo
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(tmpdir)],
                check=True,
                capture_output=True,
            )
            
            # Find GPU-related Python files if not specified
            if target_files is None:
                target_files = self._find_gpu_files(tmpdir)
                if self.verbose:
                    print(f"   Found {len(target_files)} GPU-related files")
            
            results = {}
            for file_path in target_files:
                full_path = tmpdir / file_path
                if full_path.exists():
                    output_path = None
                    if output_dir:
                        output_path = output_dir / file_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        results[file_path] = self.optimize_file(full_path, output_path)
                    except Exception as e:
                        if self.verbose:
                            print(f"   ❌ Error optimizing {file_path}: {e}")
                        results[file_path] = OptimizationResult(
                            success=False,
                            original_code="",
                            optimized_code="",
                            original_time_ms=0,
                            optimized_time_ms=0,
                            speedup=1.0,
                            techniques_applied=[],
                            explanation="",
                            profile_data={},
                            errors=[str(e)],
                        )
            
            return results
    
    def scan_and_optimize(
        self,
        directory: Path,
        threshold: float = 1.1,
        pattern: str = "baseline_*.py",
    ) -> Dict[str, OptimizationResult]:
        """
        Scan directory for underperforming benchmarks and optimize them.
        
        Args:
            directory: Directory to scan
            threshold: Only optimize if current speedup < threshold
            pattern: File pattern to match
            
        Returns:
            Dict mapping benchmark names to optimization results
        """
        directory = Path(directory)
        results = {}
        
        # Find baseline files
        baseline_files = list(directory.rglob(pattern))
        
        if self.verbose:
            print(f"🔍 Found {len(baseline_files)} benchmark files")
        
        for baseline_path in baseline_files:
            # Find corresponding optimized file
            optimized_path = baseline_path.parent / baseline_path.name.replace("baseline_", "optimized_")
            
            if not optimized_path.exists():
                continue
            
            benchmark_name = f"{baseline_path.parent.name}:{baseline_path.stem.replace('baseline_', '')}"
            
            # Check current speedup
            try:
                baseline_time = self._quick_benchmark(baseline_path)
                optimized_time = self._quick_benchmark(optimized_path)
                current_speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
                
                if current_speedup < threshold:
                    if self.verbose:
                        print(f"\n📊 {benchmark_name}: current speedup {current_speedup:.2f}x < {threshold}x")
                    
                    # Optimize the optimized variant
                    result = self.optimize_file(
                        optimized_path,
                        output_path=optimized_path,  # Overwrite
                    )
                    results[benchmark_name] = result
                else:
                    if self.verbose:
                        print(f"   ✓ {benchmark_name}: {current_speedup:.2f}x (OK)")
                        
            except Exception as e:
                if self.verbose:
                    print(f"   ❌ {benchmark_name}: {e}")
        
        return results
    
    def _profile_code(
        self,
        code: str,
        benchmark_fn: Optional[str] = None,
        setup_code: Optional[str] = None,
    ) -> ProfileResult:
        """Profile GPU code and return timing/memory data."""
        
        # Create a temporary module to run the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)
        
        try:
            # Detect benchmark function if not specified
            if benchmark_fn is None:
                benchmark_fn = self._detect_benchmark_fn(code)
            
            # Profile with torch.profiler
            profile_result = self._run_profiler(temp_path, benchmark_fn, setup_code)
            return profile_result
            
        finally:
            temp_path.unlink()
    
    def _detect_benchmark_fn(self, code: str) -> str:
        """Auto-detect the main benchmark function in code."""
        tree = ast.parse(code)
        
        # Look for common patterns
        candidates = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                name = node.name.lower()
                if any(kw in name for kw in ['benchmark', 'forward', 'run', 'main', 'compute']):
                    candidates.append(node.name)
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name in ['benchmark_fn', 'forward', 'run', '__call__']:
                            return f"{node.name}.{item.name}"
        
        return candidates[0] if candidates else "main"
    
    def _run_profiler(
        self,
        file_path: Path,
        benchmark_fn: str,
        setup_code: Optional[str],
    ) -> ProfileResult:
        """Run torch.profiler on the code."""
        
        # Simple timing-based profiling for now
        # Full profiler integration comes in Phase 2
        
        code = file_path.read_text()
        
        # Try to import and run the code
        kernel_times = {}
        bottlenecks = []
        total_time = 0.0
        memory_peak = 0.0
        
        try:
            # Use subprocess to isolate execution
            profile_script = f'''
import torch
import time
import sys
sys.path.insert(0, "{Path.cwd()}")

# Reset memory stats
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

# Load and run the code
exec(open("{file_path}").read())

# Try to find and run benchmark
import types
benchmarks = []
for name, obj in list(globals().items()):
    if isinstance(obj, type) and hasattr(obj, 'benchmark_fn'):
        benchmarks.append((name, obj))

if benchmarks:
    name, cls = benchmarks[0]
    instance = cls()
    if hasattr(instance, 'setup'):
        instance.setup()
    
    # Warmup
    for _ in range(3):
        if hasattr(instance, 'benchmark_fn'):
            instance.benchmark_fn()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Timed run
    start = time.perf_counter()
    for _ in range(10):
        if hasattr(instance, 'benchmark_fn'):
            instance.benchmark_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    total_time = (end - start) / 10 * 1000  # ms
    memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    
    print(f"TIME_MS:{{total_time}}")
    print(f"MEMORY_MB:{{memory_peak}}")
'''
            
            result = subprocess.run(
                ["python3", "-c", profile_script],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(Path.cwd()),
            )
            
            # Parse output
            for line in result.stdout.split('\n'):
                if line.startswith('TIME_MS:'):
                    total_time = float(line.split(':')[1])
                elif line.startswith('MEMORY_MB:'):
                    memory_peak = float(line.split(':')[1])
            
            # Analyze code for bottlenecks
            bottlenecks = self._analyze_bottlenecks(code)
            
        except Exception as e:
            bottlenecks = [f"Profiling error: {e}"]
            total_time = 999999.0
        
        return ProfileResult(
            total_time_ms=total_time,
            gpu_time_ms=total_time * 0.9,  # Estimate
            cpu_time_ms=total_time * 0.1,
            memory_peak_mb=memory_peak,
            memory_allocated_mb=memory_peak * 0.8,
            kernel_times=kernel_times,
            bottlenecks=bottlenecks,
        )
    
    def _analyze_bottlenecks(self, code: str) -> List[str]:
        """Static analysis to identify potential bottlenecks."""
        bottlenecks = []
        
        patterns = [
            ("for ", "Python loop (potential vectorization opportunity)"),
            (".item()", "GPU→CPU sync (.item() call)"),
            (".cpu()", "GPU→CPU transfer"),
            (".numpy()", "GPU→CPU→NumPy conversion"),
            ("torch.cat", "Tensor concatenation (memory allocation)"),
            ("torch.stack", "Tensor stacking (memory allocation)"),
            (".to(", "Device transfer"),
            ("torch.no_grad", "Missing torch.no_grad context"),
            ("backward()", "Gradient computation"),
        ]
        
        for pattern, description in patterns:
            if pattern in code:
                bottlenecks.append(description)
        
        # Check for missing optimizations
        if "torch.compile" not in code and "@torch.compile" not in code:
            bottlenecks.append("torch.compile not used")
        
        if "cuda.amp" not in code and "autocast" not in code and "bfloat16" not in code:
            bottlenecks.append("Mixed precision not used")
        
        return bottlenecks[:5]  # Top 5 bottlenecks
    
    def _analyze_with_llm(
        self,
        code: str,
        profile: ProfileResult,
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze code and suggest optimizations."""
        
        # Build prompt
        prompt = f"""Analyze this GPU code and suggest optimizations.

## Code
```python
{code[:8000]}  # Truncate if too long
```

## Current Performance
- Total time: {profile.total_time_ms:.2f}ms
- Memory peak: {profile.memory_peak_mb:.1f}MB
- Bottlenecks identified: {', '.join(profile.bottlenecks)}

## Task
Suggest specific code optimizations. Focus on:
1. torch.compile for kernel fusion
2. Memory access patterns
3. Parallelization opportunities
4. Mixed precision (bfloat16/float16)
5. CUDA-specific optimizations

Return JSON with:
{{
    "techniques": ["technique1", "technique2"],
    "patches": [
        {{
            "type": "method_replacement",
            "method_name": "method_to_replace",
            "new_code": "def method_to_replace(self, ...):\\n    ..."
        }}
    ],
    "expected_speedup": 1.5,
    "explanation": "Why these changes help..."
}}
"""
        
        try:
            response = self.analyzer._call_llm(prompt)
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            if self.verbose:
                print(f"   LLM analysis error: {e}")
        
        return None
    
    def _apply_patches(
        self,
        code: str,
        patches: List[Dict],
    ) -> Tuple[str, List[Dict]]:
        """Apply patches to code and return results."""
        
        patched_code = code
        results = []
        
        for patch in patches:
            try:
                new_code = self.patch_applier._apply_structured_patch(
                    patched_code, patch, "auto_optimize"
                )
                if new_code and new_code != patched_code:
                    # Validate syntax
                    ast.parse(new_code)
                    patched_code = new_code
                    results.append({"success": True, "patch": patch})
                else:
                    results.append({"success": False, "error": "No changes made", "patch": patch})
            except SyntaxError as e:
                results.append({"success": False, "error": f"Syntax error: {e}", "patch": patch})
            except Exception as e:
                results.append({"success": False, "error": str(e), "patch": patch})
        
        return patched_code, results
    
    def _generate_explanation(
        self,
        original: str,
        optimized: str,
        techniques: List[str],
        speedup: float,
    ) -> str:
        """Generate human-readable explanation of optimizations."""
        
        if speedup <= 1.0:
            return "No significant optimizations were applied."
        
        explanation = f"## Optimization Summary\n\n"
        explanation += f"**Speedup achieved: {speedup:.2f}x**\n\n"
        
        if techniques:
            explanation += "### Techniques Applied\n"
            for tech in techniques:
                explanation += f"- {tech}\n"
        
        explanation += "\n### What Changed\n"
        
        # Simple diff summary
        orig_lines = set(original.split('\n'))
        opt_lines = set(optimized.split('\n'))
        
        added = len(opt_lines - orig_lines)
        removed = len(orig_lines - opt_lines)
        
        explanation += f"- {added} lines added\n"
        explanation += f"- {removed} lines removed/modified\n"
        
        return explanation
    
    def _find_gpu_files(self, directory: Path) -> List[str]:
        """Find Python files that likely contain GPU code."""
        gpu_files = []
        
        gpu_indicators = [
            'torch.cuda', 'cuda()', '.to(device)',
            'torch.compile', 'triton', 'cuda.amp',
            'bfloat16', 'float16', 'tensor_cores',
        ]
        
        for py_file in directory.rglob('*.py'):
            if '__pycache__' in str(py_file):
                continue
            try:
                content = py_file.read_text()
                if any(ind in content for ind in gpu_indicators):
                    gpu_files.append(str(py_file.relative_to(directory)))
            except:
                pass
        
        return gpu_files
    
    def _quick_benchmark(self, file_path: Path) -> float:
        """Quick benchmark of a file, returns time in ms."""
        profile = self._profile_code(file_path.read_text())
        return profile.total_time_ms


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-optimize GPU code")
    parser.add_argument("input", nargs="?", help="File path or repo URL")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--target", "-t", action="append", help="Target files in repo")
    parser.add_argument("--scan", action="store_true", help="Scan directory for benchmarks")
    parser.add_argument("--threshold", type=float, default=1.1, help="Speedup threshold for --scan")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--model", help="LLM model to use")
    parser.add_argument("--iterations", type=int, default=3, help="Max optimization iterations")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    parser.add_argument("--quiet", "-q", action="store_true")
    
    args = parser.parse_args()
    
    optimizer = AutoOptimizer(
        llm_provider=args.provider,
        model=args.model,
        verbose=not args.quiet,
        max_iterations=args.iterations,
    )
    
    if args.scan:
        directory = Path(args.input) if args.input else Path.cwd()
        results = optimizer.scan_and_optimize(directory, threshold=args.threshold)
        
        print(f"\n{'='*60}")
        print(f"Optimization Summary: {len(results)} benchmarks processed")
        for name, result in results.items():
            status = "✅" if result.success else "❌"
            print(f"  {status} {name}: {result.speedup:.2f}x")
    
    elif args.input and args.input.startswith(("http://", "https://", "git@")):
        results = optimizer.optimize_repo(
            args.input,
            target_files=args.target,
            output_dir=Path(args.output) if args.output else None,
        )
        
        print(f"\n{'='*60}")
        print(f"Repo Optimization Summary: {len(results)} files processed")
        for path, result in results.items():
            status = "✅" if result.success else "❌"
            print(f"  {status} {path}: {result.speedup:.2f}x")
    
    elif args.input:
        result = optimizer.optimize_file(
            Path(args.input),
            output_path=Path(args.output) if args.output else None,
        )
        
        print(f"\n{'='*60}")
        print(f"Result: {'✅ Success' if result.success else '❌ No improvement'}")
        print(f"Speedup: {result.speedup:.2f}x")
        print(f"Time: {result.original_time_ms:.2f}ms → {result.optimized_time_ms:.2f}ms")
        if result.techniques_applied:
            print(f"Techniques: {', '.join(result.techniques_applied)}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

