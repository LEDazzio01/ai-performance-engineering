"""Book-aligned wrapper for after_ddp.py"""
from pathlib import Path
import sys

CHAPTER_DIR = Path(__file__).parent
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from optimized_dataparallel import get_benchmark as _get_benchmark


def get_benchmark():
    bench = _get_benchmark()
    bench.name = "dataparallel_basic_optimized"
    return bench
