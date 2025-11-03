#!/usr/bin/env python3
"""Allocator retry helper from Chapter 19."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


def _resolve_factory(factory_path: str) -> Callable[[], Any]:
    module_name, func_name = factory_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _spawn_with_allocator(factory_path: str, request_blob: bytes, allocator_conf: str) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(request_blob)
        req_path = tmp.name

    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = allocator_conf

    helper = Path(__file__).with_name("_allocator_worker.py")
    cmd = [sys.executable, str(helper), factory_path, req_path]
    completed = subprocess.run(cmd, check=True, env=env, capture_output=True)
    os.unlink(req_path)
    return completed.stdout


def generate_with_allocator_retry(
    factory_path: str,
    request_object: Any,
    *,
    allocator_conf: str = "backend:cudaMallocAsync",
) -> Any:
    import torch  # Import lazily to avoid polluting subprocess.

    model_factory = _resolve_factory(factory_path)
    model = model_factory()
    try:
        return model.generate(request_object)
    except torch.cuda.OutOfMemoryError:
        request_blob = pickle.dumps(request_object)
        response_blob = _spawn_with_allocator(factory_path, request_blob, allocator_conf)
        return pickle.loads(response_blob)


if __name__ == "__main__":
    print("dynamic_memory_allocator helper module; see book for usage.")
