"""
Report Generation Module

Generate polished PDF and HTML reports for:
- Benchmark results
- Profiling analysis
- Optimization summaries
- LLM insights

Usage:
    from tools.reporting import ReportGenerator
    
    generator = ReportGenerator()
    generator.generate_pdf(benchmark_data, "report.pdf")
"""

from .generator import ReportGenerator, ReportConfig, BenchmarkData, generate_report
from .templates import BenchmarkReport, ProfileReport, OptimizationReport

__all__ = [
    'ReportGenerator',
    'ReportConfig',
    'BenchmarkData',
    'generate_report',
    'BenchmarkReport',
    'ProfileReport',
    'OptimizationReport',
]

