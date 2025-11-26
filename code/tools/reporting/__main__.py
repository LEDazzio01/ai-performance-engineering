"""
CLI entry point for report generation.

Usage:
    python -m tools.reporting benchmark_results.json -o report.pdf
    python -m tools.reporting http://localhost:8080 -o report.pdf --from-api
    python -m tools.reporting --list-formats
"""

from .generator import main

if __name__ == "__main__":
    main()

