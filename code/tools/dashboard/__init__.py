"""
GPU Performance Lab Dashboard

A sleek web dashboard for viewing benchmark results, LLM analysis,
and optimization insights.

Usage:
    python -m tools.dashboard.server [--port 6970] [--data results.json]
"""

from .server import serve_dashboard

__all__ = ['serve_dashboard']



