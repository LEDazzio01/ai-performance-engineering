"""
CLI entry point for profiling tools.

Usage:
    python -m tools.profiling profile script.py --output profile.json
    python -m tools.profiling memory script.py --output memory.json
    python -m tools.profiling flame trace.json --output flame.html
    python -m tools.profiling hta trace.json --output hta_report.html
    python -m tools.profiling compile model.py --output compile_report.json
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="GPU Profiling Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Profile a Python script
    python -m tools.profiling profile script.py -o profile.json
    
    # Generate flame graph from trace
    python -m tools.profiling flame trace.json -o flame.html
    
    # Analyze torch.compile behavior
    python -m tools.profiling compile model.py -o report.json
    
    # Run HTA analysis on trace
    python -m tools.profiling hta trace.json -o hta_report.html
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Profiling command")
    
    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Profile GPU code")
    profile_parser.add_argument("script", help="Python script to profile")
    profile_parser.add_argument("-o", "--output", default="profile.json", help="Output file")
    profile_parser.add_argument("--iterations", type=int, default=10, help="Profile iterations")
    profile_parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    profile_parser.add_argument("--html", action="store_true", help="Generate HTML report")
    
    # Memory command
    memory_parser = subparsers.add_parser("memory", help="Profile memory usage")
    memory_parser.add_argument("script", help="Python script to profile")
    memory_parser.add_argument("-o", "--output", default="memory.json", help="Output file")
    
    # Flame graph command
    flame_parser = subparsers.add_parser("flame", help="Generate flame graph")
    flame_parser.add_argument("trace", help="Chrome trace JSON file")
    flame_parser.add_argument("-o", "--output", default="flame.html", help="Output file")
    flame_parser.add_argument("--json", action="store_true", help="Output JSON instead of HTML")
    
    # Timeline command
    timeline_parser = subparsers.add_parser("timeline", help="Generate CPU/GPU timeline")
    timeline_parser.add_argument("trace", help="Chrome trace JSON file")
    timeline_parser.add_argument("-o", "--output", default="timeline.html", help="Output file")
    
    # HTA command
    hta_parser = subparsers.add_parser("hta", help="Run HTA analysis")
    hta_parser.add_argument("trace", help="Chrome trace JSON file")
    hta_parser.add_argument("-o", "--output", default="hta_report.html", help="Output file")
    hta_parser.add_argument("--json", action="store_true", help="Output JSON instead of HTML")
    
    # Compile analysis command
    compile_parser = subparsers.add_parser("compile", help="Analyze torch.compile")
    compile_parser.add_argument("script", help="Python script with model")
    compile_parser.add_argument("-o", "--output", default="compile_report.json", help="Output file")
    compile_parser.add_argument("--mode", default="default", help="Compile mode")
    compile_parser.add_argument("--compare", action="store_true", help="Compare all modes")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "flame":
        from .flame_graph import FlameGraphGenerator
        
        generator = FlameGraphGenerator()
        data = generator.from_chrome_trace(Path(args.trace))
        
        output_format = "json" if args.json else "html"
        generator.export(data, Path(args.output), format=output_format)
        print(f"✅ Flame graph saved to {args.output}")
    
    elif args.command == "timeline":
        from .timeline import TimelineGenerator
        
        generator = TimelineGenerator()
        timeline = generator.from_chrome_trace(Path(args.trace))
        generator.generate_html_viewer(timeline, Path(args.output))
        print(f"✅ Timeline saved to {args.output}")
    
    elif args.command == "hta":
        from .hta_integration import HTAAnalyzer
        
        analyzer = HTAAnalyzer()
        report = analyzer.analyze_trace(Path(args.trace))
        
        output_format = "json" if args.json else "html"
        analyzer.export_report(report, Path(args.output), format=output_format)
        print(f"✅ HTA report saved to {args.output}")
    
    elif args.command == "profile":
        print(f"Profiling {args.script}...")
        print("Note: For script profiling, use torch.profiler directly or the benchmark harness.")
        
    elif args.command == "compile":
        print(f"Analyzing torch.compile for {args.script}...")
        print("Note: For full analysis, import TorchCompileAnalyzer and use with your model.")
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

