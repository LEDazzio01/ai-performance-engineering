"""Compare accuracy across different precision modes (FP32, FP16, BF16, FP8)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List


def run_perplexity_eval(
    dataset: Path,
    dtype: str,
    fp8_mode: str = "none",
    seq_len: int = 512,
    stride: int = 256
) -> Dict[str, float]:
    """Run perplexity evaluation with specified precision.
    
    Returns metrics dict with perplexity, avg_loss, etc.
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_json = Path(f.name)
    
    try:
        cmd = [
            sys.executable,
            "ch16/perplexity_eval.py",
            str(dataset),
            "--seq-len", str(seq_len),
            "--stride", str(stride),
            "--dtype", dtype,
            "--output-json", str(output_json)
        ]
        
        print(f"  Running evaluation: dtype={dtype}, fp8_mode={fp8_mode}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            print(f"    Warning: Evaluation failed")
            print(f"    stdout: {result.stdout}")
            print(f"    stderr: {result.stderr}")
            return {}
        
        if not output_json.exists():
            print(f"    Warning: Output file not created")
            return {}
        
        metrics = json.loads(output_json.read_text())
        return metrics
    
    except subprocess.TimeoutExpired:
        print(f"    Warning: Evaluation timed out")
        return {}
    except Exception as e:
        print(f"    Warning: Evaluation error: {e}")
        return {}
    finally:
        if output_json.exists():
            output_json.unlink()


def compare_precisions(
    dataset: Path,
    precisions: List[str],
    seq_len: int = 512,
    stride: int = 256
) -> Dict[str, Dict[str, float]]:
    """Compare perplexity across different precision modes."""
    
    results = {}
    
    for precision in precisions:
        print(f"\nEvaluating {precision}...")
        
        # Map precision to dtype and fp8_mode
        if precision == "fp32":
            dtype = "float16"  # Will use float32 in model
            fp8_mode = "none"
        elif precision == "fp16":
            dtype = "float16"
            fp8_mode = "none"
        elif precision == "bf16":
            dtype = "bfloat16"
            fp8_mode = "none"
        elif precision == "fp8":
            dtype = "float16"
            fp8_mode = "transformer-engine"
        else:
            print(f"  Unknown precision: {precision}, skipping")
            continue
        
        metrics = run_perplexity_eval(dataset, dtype, fp8_mode, seq_len, stride)
        
        if metrics:
            results[precision] = metrics
            print(f"  Perplexity: {metrics.get('perplexity', 'N/A'):.3f}")
            print(f"  Avg Loss: {metrics.get('avg_loss', 'N/A'):.4f}")
    
    return results


def generate_comparison_report(
    results: Dict[str, Dict[str, float]],
    dataset_name: str,
    output_path: Path
) -> None:
    """Generate a markdown report comparing precision modes."""
    
    if not results:
        print("No results to generate report")
        return
    
    lines = []
    lines.append("# Precision Accuracy Comparison\n")
    lines.append(f"\n**Dataset**: `{dataset_name}`\n")
    lines.append(f"\n## Results\n")
    lines.append("\n| Precision | Perplexity | Avg Loss | Tokens Evaluated |\n")
    lines.append("|-----------|------------|----------|------------------|\n")
    
    # Sort by precision (fp32 first as baseline)
    precision_order = ["fp32", "fp16", "bf16", "fp8"]
    sorted_precisions = [p for p in precision_order if p in results]
    
    baseline_perplexity = None
    if "fp16" in results:
        baseline_perplexity = results["fp16"].get("perplexity")
    
    for precision in sorted_precisions:
        metrics = results[precision]
        perplexity = metrics.get("perplexity", float('nan'))
        avg_loss = metrics.get("avg_loss", float('nan'))
        tokens = metrics.get("tokens_evaluated", 0)
        
        lines.append(f"| {precision.upper()} | {perplexity:.3f} | {avg_loss:.4f} | {tokens:,} |\n")
    
    # Add analysis section
    lines.append("\n## Analysis\n")
    
    if "fp16" in results and "fp8" in results:
        fp16_ppl = results["fp16"].get("perplexity", float('nan'))
        fp8_ppl = results["fp8"].get("perplexity", float('nan'))
        
        if fp16_ppl > 0 and fp8_ppl > 0:
            delta = ((fp8_ppl - fp16_ppl) / fp16_ppl) * 100
            lines.append(f"\n### FP16 vs FP8\n")
            lines.append(f"- FP16 perplexity: {fp16_ppl:.3f}\n")
            lines.append(f"- FP8 perplexity: {fp8_ppl:.3f}\n")
            lines.append(f"- Delta: {delta:+.2f}%\n")
            
            if abs(delta) < 1:
                lines.append(f"- **Status**: ✅ Negligible accuracy impact (<1%)\n")
            elif abs(delta) < 5:
                lines.append(f"- **Status**: ⚠️ Small accuracy impact (<5%)\n")
            elif abs(delta) < 10:
                lines.append(f"- **Status**: ⚠️ Moderate accuracy impact (<10%)\n")
            else:
                lines.append(f"- **Status**: 🚨 Significant accuracy impact (>10%)\n")
    
    if "fp16" in results and "bf16" in results:
        fp16_ppl = results["fp16"].get("perplexity", float('nan'))
        bf16_ppl = results["bf16"].get("perplexity", float('nan'))
        
        if fp16_ppl > 0 and bf16_ppl > 0:
            delta = ((bf16_ppl - fp16_ppl) / fp16_ppl) * 100
            lines.append(f"\n### FP16 vs BF16\n")
            lines.append(f"- FP16 perplexity: {fp16_ppl:.3f}\n")
            lines.append(f"- BF16 perplexity: {bf16_ppl:.3f}\n")
            lines.append(f"- Delta: {delta:+.2f}%\n")
    
    lines.append("\n## Recommendations\n")
    lines.append("\n")
    lines.append("- **FP16**: Standard baseline for inference\n")
    lines.append("- **BF16**: Better for training, similar inference accuracy\n")
    lines.append("- **FP8**: Use when accuracy delta is acceptable and performance gain is significant\n")
    lines.append("\n")
    lines.append("Always validate accuracy on your specific workload before deploying FP8.\n")
    
    report = "".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\n✅ Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare accuracy across precision modes"
    )
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Path to tokenized evaluation dataset")
    parser.add_argument("--precisions", nargs='+',
                        default=["fp16", "bf16", "fp8"],
                        choices=["fp32", "fp16", "bf16", "fp8"],
                        help="Precision modes to compare")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for evaluation")
    parser.add_argument("--stride", type=int, default=256,
                        help="Stride between evaluation windows")
    parser.add_argument("--output", type=Path,
                        default=Path("precision_comparison.md"),
                        help="Output file for comparison report")
    parser.add_argument("--output-json", type=Path,
                        help="Optional JSON output for metrics")
    args = parser.parse_args()
    
    if not args.dataset.exists():
        print(f"Error: Dataset not found: {args.dataset}")
        return 1
    
    print("=" * 80)
    print("Precision Accuracy Comparison")
    print("=" * 80)
    print(f"\nDataset: {args.dataset}")
    print(f"Precisions: {', '.join(args.precisions)}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Stride: {args.stride}")
    
    results = compare_precisions(
        args.dataset,
        args.precisions,
        args.seq_len,
        args.stride
    )
    
    if not results:
        print("\n❌ No results collected")
        return 1
    
    # Generate report
    generate_comparison_report(
        results,
        args.dataset.name,
        args.output
    )
    
    # Optionally save JSON
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        print(f"✅ JSON metrics saved to: {args.output_json}")
    
    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


