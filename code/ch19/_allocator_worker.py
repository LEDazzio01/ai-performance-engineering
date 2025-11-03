#!/usr/bin/env python3
"""Subprocess helper for dynamic_memory_allocator."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path


def main() -> int:
    factory_path = sys.argv[1]
    request_file = Path(sys.argv[2])

    import dynamic_memory_allocator as dma  # local module import

    factory = dma._resolve_factory(factory_path)
    request = pickle.loads(request_file.read_bytes())
    model = factory()
    result = model.generate(request)
    sys.stdout.buffer.write(pickle.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

