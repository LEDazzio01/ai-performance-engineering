"""
Lab: Matching cuBLAS on Blackwell
=================================

A progressive optimization exercise showing how to approach
cuBLAS-level performance with custom tensor core kernels.

Run with:
    python -m labs.matching_cublas.run_lab
    
Or directly:
    python labs/matching_cublas/run_lab.py
"""

from pathlib import Path

LAB_DIR = Path(__file__).parent
README = LAB_DIR / "README.md"

