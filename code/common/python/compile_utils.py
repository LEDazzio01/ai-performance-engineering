"""Utilities for safely applying torch.compile and precision defaults."""

from __future__ import annotations

import importlib
import logging
import threading
import warnings
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, cast

import torch

logger = logging.getLogger(__name__)

_LEGACY_TF32_PATCHED = False
_COMPILE_WARNED = False


def _patch_legacy_tf32_attributes() -> None:
    """
    Override legacy TF32 accessors so reads/writes no-op instead of calling the
    deprecated CUDA context APIs (which cause mixing errors once the new API is used).
    """
    global _LEGACY_TF32_PATCHED
    if _LEGACY_TF32_PATCHED:
        return

    try:
        cu_blas_cls = torch.backends.cuda.cuBLASModule
    except AttributeError:
        cu_blas_cls = None

    if cu_blas_cls is not None:
        original_getattr = cu_blas_cls.__getattr__
        original_setattr = cu_blas_cls.__setattr__

        def _patched_getattr(self, name):
            if name == "allow_tf32":
                return getattr(self, "_legacy_allow_tf32", False)
            return original_getattr(self, name)

        def _patched_setattr(self, name, value):
            if name == "allow_tf32":
                object.__setattr__(self, "_legacy_allow_tf32", bool(value))
                return value
            return original_setattr(self, name, value)

        cu_blas_cls.__getattr__ = _patched_getattr  # type: ignore[assignment]
        cu_blas_cls.__setattr__ = _patched_setattr  # type: ignore[assignment]

    try:
        cudnn_cls = type(torch.backends.cudnn)
    except AttributeError:
        cudnn_cls = None

    if cudnn_cls is not None:
        def _cudnn_allow_tf32_get(self):
            return getattr(self, "_legacy_allow_tf32", False)

        def _cudnn_allow_tf32_set(self, value):
            object.__setattr__(self, "_legacy_allow_tf32", bool(value))

        cudnn_cls.allow_tf32 = property(  # type: ignore[assignment]
            _cudnn_allow_tf32_get,
            _cudnn_allow_tf32_set,
        )

    _LEGACY_TF32_PATCHED = True


def _mirror_legacy_tf32_flags(enable_matmul: bool, enable_cudnn: bool) -> None:
    """
    Store the TF32 state on the patched legacy accessors so callers that still read
    torch.backends.*.allow_tf32 observe the right boolean without touching the old API.
    """
    try:
        torch.backends.cuda.matmul.allow_tf32 = enable_matmul  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = enable_cudnn  # type: ignore[attr-defined]
    except Exception:
        pass


def get_optimal_compile_mode(preferred_mode: str = "max-autotune", sm_threshold: int = 68) -> str:
    """
    Get the optimal torch.compile mode based on GPU SM count.
    
    max-autotune requires >= 68 SMs (Streaming Multiprocessors) for GEMM operations.
    GPUs with fewer SMs will fall back to reduce-overhead to avoid warnings.
    
    Parameters
    ----------
    preferred_mode:
        The preferred compile mode (default: "max-autotune")
    sm_threshold:
        Minimum number of SMs required for max-autotune (default: 68)
    
    Returns
    -------
    str
        The compile mode to use: "max-autotune" if GPU has enough SMs,
        otherwise "reduce-overhead"
    """
    if preferred_mode != "max-autotune":
        return preferred_mode
    
    if not torch.cuda.is_available():
        return "reduce-overhead"
    
    try:
        device_index = torch.cuda.current_device()
        num_sms = torch.cuda.get_device_properties(device_index).multi_processor_count
        
        if num_sms >= sm_threshold:
            return "max-autotune"
        else:
            return "reduce-overhead"
    except Exception:
        # If we can't determine SM count, use reduce-overhead to be safe
        return "reduce-overhead"


def compile_model(module: torch.nn.Module, **_: Any) -> torch.nn.Module:
    """
    Compile a module with torch.compile when available.

    Parameters follow the historical signature used throughout the chapters.
    Unknown keyword arguments are ignored intentionally to preserve backwards
    compatibility with chapter-specific wrappers.
    """
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return module

    if getattr(module, "_is_compiled_benchmark_module", False):
        return module

    mode = _.get("mode", "max-autotune")
    fullgraph = _.get("fullgraph", False)
    dynamic = _.get("dynamic", False)
    options = _.get("options")

    chosen_mode = get_optimal_compile_mode(mode)
    try:
        compiled = compile_fn(
            module,
            mode=chosen_mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            options=options,
        )
        setattr(compiled, "_is_compiled_benchmark_module", True)
        return compiled
    except Exception as exc:
        global _COMPILE_WARNED
        if not _COMPILE_WARNED:
            logger.warning("torch.compile failed (falling back to eager): %s", exc)
            _COMPILE_WARNED = True
        return module


def enable_tf32(
    *,
    matmul_precision: str = "tf32",
    cudnn_precision: str = "tf32",
    set_global_precision: bool = True,
) -> None:
    """
    Configure TF32 execution using the new PyTorch 2.9 APIs only.

    Parameters
    ----------
    matmul_precision:
        Precision setting forwarded to ``torch.backends.cuda.matmul.fp32_precision``.
    cudnn_precision:
        Precision setting forwarded to ``torch.backends.cudnn.conv.fp32_precision``.
    set_global_precision:
        When True, call ``torch.set_float32_matmul_precision('high')`` to
        ensure matmul kernels fall back to TF32-capable tensor cores.
    """
    # Suppress TF32 API deprecation warnings
    # PyTorch internally uses deprecated APIs when set_float32_matmul_precision is called
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32.*", category=UserWarning)

        _patch_legacy_tf32_attributes()
        
        matmul_backend = getattr(torch.backends.cuda, "matmul", None)
        if matmul_backend is not None and hasattr(matmul_backend, "fp32_precision"):
            try:
                backend_mut = cast(Any, matmul_backend)
                backend_mut.fp32_precision = matmul_precision
            except RuntimeError:
                pass

        cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            try:
                cudnn_mut = cast(Any, cudnn_conv)
                cudnn_mut.fp32_precision = cudnn_precision
            except RuntimeError:
                pass

        # Mirror new TF32 settings onto legacy attributes without invoking the old API.
        # This prevents PyTorch internals (e.g., torch.compile) from reading the legacy
        # allow_tf32 accessors, which would otherwise raise a mixing error once the new
        # API has been used.
    _mirror_legacy_tf32_flags(
        matmul_precision == "tf32",
        cudnn_precision == "tf32",
    )


@lru_cache()
def _get_compiled_architectures() -> Tuple[str, ...]:
    """Return the set of architectures baked into the current PyTorch build."""
    try:
        get_arch_list = getattr(torch.cuda, "get_arch_list", None)
        if get_arch_list is None:
            return tuple()
        return tuple(get_arch_list())
    except Exception:
        return tuple()


def _format_arch(major: int, minor: int) -> str:
    return f"sm_{major}{minor}"


def _supported_arch_aliases(major: int, minor: int) -> Tuple[str, ...]:
    """Return possible arch strings (sm_X, sm_Xa, compute_X) for a device."""
    digits = f"{major}{minor}"
    aliases = []
    for suffix in ("", "a", "b"):
        aliases.append(f"sm_{digits}{suffix}")
        aliases.append(f"compute_{digits}{suffix}")
    return tuple(aliases)


def is_torch_compile_supported_on_device(device_index: Optional[int] = None) -> Tuple[bool, Optional[str]]:
    """
    Determine if torch.compile has kernel support for the active CUDA architecture.
    
    Returns (True, None) when the architecture is baked into the current PyTorch build,
    otherwise (False, reason) describing the capability gap.
    """
    if not torch.cuda.is_available():
        return True, None
    
    try:
        if device_index is None:
            if getattr(torch.cuda, "_initialized", False):
                device_index = torch.cuda.current_device()
            else:
                device_index = 0
        major, minor = torch.cuda.get_device_capability(device_index)
    except Exception as exc:  # pragma: no cover - depends on runtime
        return False, f"torch.compile: unable to query CUDA capability ({exc})."
    
    supported_arches = _get_compiled_architectures()
    aliases = _supported_arch_aliases(major, minor)
    for alias in aliases:
        if alias in supported_arches:
            return True, None
    
    arch_display = _format_arch(major, minor)
    supported_display = ", ".join(supported_arches) if supported_arches else "unknown"
    reason = (
        f"GPU {arch_display} (compute capability {major}.{minor}) is missing from the "
        f"PyTorch {torch.__version__} SASS list [{supported_display}]."
    )
    return False, reason


_WARNED_MESSAGES: set[str] = set()


def _log_once(message: Optional[str]) -> None:
    if not message:
        return
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    logger.warning(message)


_CUDAGRAPH_TLS_PATCHED = False


def _make_tls_default(attr_name: str) -> Optional[Any]:
    if attr_name == "tree_manager_containers":
        return {}
    if attr_name == "tree_manager_locks":
        return defaultdict(threading.Lock)
    return None


def _patch_cudagraph_tls_bug() -> None:
    """
    Work around a PyTorch TLS regression on newly added architectures (e.g. SM12.1).
    
    When CUDA graph trees try to read thread-local state before the Inductor module
    has stashed the default dictionaries, an assertion is raised. We defensively
    recreate the missing structures and stash them so later reads succeed.
    """
    global _CUDAGRAPH_TLS_PATCHED
    if _CUDAGRAPH_TLS_PATCHED:
        return

    try:
        trees = importlib.import_module("torch._inductor.cudagraph_trees")
    except Exception as exc:  # pragma: no cover - safety path
        logger.debug("Unable to import torch._inductor.cudagraph_trees: %s", exc)
        return

    original_get_obj = getattr(trees, "get_obj", None)
    if original_get_obj is None:
        return

    def patched_get_obj(local: Any, attr_name: str) -> Any:
        if hasattr(local, attr_name):
            return getattr(local, attr_name)

        if torch._C._is_key_in_tls(attr_name):
            return torch._C._get_obj_in_tls(attr_name)

        fallback = _make_tls_default(attr_name)
        if fallback is None:
            # Mirror the upstream assertion so failures are loud
            raise AssertionError(f"Missing TLS object for {attr_name}")

        setattr(local, attr_name, fallback)
        try:
            torch._C._stash_obj_in_tls(attr_name, fallback)
        except Exception:
            logger.debug("Unable to stash TLS object for %s", attr_name, exc_info=True)
        _log_once(f"Rebuilt missing CUDA graph TLS bucket '{attr_name}' to avoid torch.compile crashes.")
        return fallback

    trees.get_obj = patched_get_obj  # type: ignore[assignment]
    _CUDAGRAPH_TLS_PATCHED = True


_patch_cudagraph_tls_bug()
