"""
CLI entry point for auto-optimizer.

Usage:
    python -m tools.optimize code.py --output optimized.py
    python -m tools.optimize https://github.com/user/repo --target model.py
    python -m tools.optimize --scan . --threshold 1.1
"""

from .optimizer import main

if __name__ == "__main__":
    main()

