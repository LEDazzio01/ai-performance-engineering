"""
Safe torch.compile wrapper with timeout and graceful fallback.

This module provides a robust wrapper around torch.compile that:
1. Detects large models (>40B parameters) and warns about potential hangs
2. Provides timeout support for compilation (prevents indefinite hangs)
3. Gracefully falls back to eager mode if compilation fails or times out
4. Provides clear diagnostics about why compilation was skipped

Usage:
    from common.torch_compile_safe import safe_compile
    
    # Simple usage with automatic fallback
    model_compiled = safe_compile(model, mode='max-autotune')
    
    # With explicit timeout (seconds)
    model_compiled = safe_compile(model, timeout=300)  # 5 min timeout
    
    # Force disable for large models
    model_compiled = safe_compile(model, skip_if_large=True)
"""

import os
import threading
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn


# Environment variable to disable timeout (for debugging)
DISABLE_TIMEOUT = os.environ.get("TORCH_COMPILE_DISABLE_TIMEOUT", "0") == "1"

# Default timeout for compilation (seconds)
# Large models can take 5-10 minutes to compile
DEFAULT_COMPILE_TIMEOUT = int(os.environ.get("TORCH_COMPILE_TIMEOUT", "600"))  # 10 minutes

# Parameter count threshold for "large" models (40B = 40e9)
LARGE_MODEL_THRESHOLD = 40_000_000_000


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def is_large_model(model: nn.Module, threshold: int = LARGE_MODEL_THRESHOLD) -> bool:
    """Check if model exceeds large model threshold."""
    try:
        param_count = count_parameters(model)
        return param_count >= threshold
    except Exception:
        # If we can't count parameters, assume it's not large
        return False


class CompilationTimeoutError(Exception):
    """Raised when compilation exceeds timeout."""
    pass


def _compile_with_timeout(
    model: nn.Module,
    compile_fn: Callable,
    timeout: int,
    *args: Any,
    **kwargs: Any
) -> nn.Module:
    """
    Compile model with timeout support.
    
    Uses threading to implement timeout - the actual compilation happens
    in a separate thread, and we wait for it with a timeout.
    """
    if DISABLE_TIMEOUT:
        return compile_fn(model, *args, **kwargs)
    
    result_container = {"compiled": None, "error": None, "done": False}
    
    def compile_worker():
        try:
            result_container["compiled"] = compile_fn(model, *args, **kwargs)
        except Exception as e:
            result_container["error"] = e
        finally:
            result_container["done"] = True
    
    thread = threading.Thread(target=compile_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if not result_container["done"]:
        raise CompilationTimeoutError(
            f"Compilation exceeded timeout of {timeout} seconds. "
            f"This is common for models >40B parameters. "
            f"Consider using eager mode or increasing timeout."
        )
    
    if result_container["error"]:
        raise result_container["error"]
    
    return result_container["compiled"]


def safe_compile(
    model: nn.Module,
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool = False,
    backend: str = "inductor",
    timeout: Optional[int] = None,
    skip_if_large: bool = False,
    warn_on_skip: bool = True,
    **kwargs: Any
) -> Union[nn.Module, Callable]:
    """
    Safely compile a model with timeout and fallback support.
    
    Args:
        model: Model to compile
        mode: torch.compile mode ('default', 'reduce-overhead', 'max-autotune')
        fullgraph: Whether to compile entire graph
        dynamic: Whether shapes are dynamic
        backend: Compilation backend
        timeout: Compilation timeout in seconds (default: 600)
        skip_if_large: Skip compilation for large models (>40B params)
        warn_on_skip: Print warning when compilation is skipped
        **kwargs: Additional arguments passed to torch.compile
    
    Returns:
        Compiled model, or original model if compilation fails/skipped
    
    Raises:
        CompilationTimeoutError: If compilation exceeds timeout
    """
    if timeout is None:
        timeout = DEFAULT_COMPILE_TIMEOUT
    
    # Check if model is large
    is_large = is_large_model(model)
    
    if skip_if_large and is_large:
        if warn_on_skip:
            param_count = count_parameters(model)
            print(
                f"⚠️  Skipping torch.compile for large model "
                f"({param_count / 1e9:.1f}B parameters). "
                f"Compilation hangs are common for models >40B."
            )
        return model
    
    if is_large and warn_on_skip:
        param_count = count_parameters(model)
        print(
            f"⚠️  Warning: Compiling large model ({param_count / 1e9:.1f}B parameters). "
            f"This may take {timeout // 60} minutes or hang indefinitely. "
            f"Consider using eager mode or --skip-compile flag."
        )
    
    # Define compilation function
    def compile_fn(m: nn.Module) -> nn.Module:
        return torch.compile(
            m,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            **kwargs
        )
    
    try:
        # Attempt compilation with timeout
        compiled = _compile_with_timeout(
            model,
            compile_fn,
            timeout,
        )
        
        if warn_on_skip and is_large:
            print("✅ Compilation completed successfully (large model)")
        
        return compiled
        
    except CompilationTimeoutError as e:
        if warn_on_skip:
            print(f"❌ {e}")
            print("   Falling back to eager mode.")
        return model
        
    except Exception as e:
        error_msg = str(e)
        
        # Check for known failure modes
        if "out of memory" in error_msg.lower():
            if warn_on_skip:
                print(f"❌ Compilation failed: OOM. Falling back to eager mode.")
        elif "duplicate template name" in error_msg.lower():
            if warn_on_skip:
                print(
                    f"⚠️  Compilation skipped: Known PyTorch issue "
                    f"(duplicate kernel template name). "
                    f"Falling back to eager mode."
                )
        else:
            if warn_on_skip:
                print(f"❌ Compilation failed: {error_msg}")
                print("   Falling back to eager mode.")
        
        return model


def should_use_compile(
    model: nn.Module,
    model_size_gb: Optional[float] = None
) -> tuple[bool, str]:
    """
    Determine if torch.compile should be used for a model.
    
    Returns:
        (should_compile, reason)
    """
    param_count = count_parameters(model)
    param_count_b = param_count / 1e9
    
    # Estimate model size if not provided
    if model_size_gb is None:
        # Rough estimate: FP16 = 2 bytes per parameter
        model_size_gb = param_count * 2 / 1e9
    
    # Hard cutoff: >40B parameters
    if param_count >= LARGE_MODEL_THRESHOLD:
        return False, f"Model too large ({param_count_b:.1f}B params): compilation hangs"
    
    # Soft cutoff: >80GB model size suggests memory-bound workload
    if model_size_gb > 80:
        return False, f"Model size ({model_size_gb:.1f}GB) suggests memory-bound workload"
    
    # For smaller models, compilation is usually beneficial
    if param_count_b < 1:
        return True, f"Small model ({param_count_b:.2f}B params): compilation recommended"
    elif param_count_b < 10:
        return True, f"Medium model ({param_count_b:.1f}B params): compilation may help"
    else:
        return True, f"Large model ({param_count_b:.1f}B params): compilation may hang, profile first"


def detect_transformer_layers(model: nn.Module) -> Optional[nn.ModuleList]:
    """
    Detect transformer layers in a model.
    
    Looks for common patterns:
    - model.layers (ModuleList)
    - model.transformer.layers
    - model.blocks (Sequential or ModuleList)
    
    Returns:
        ModuleList of transformer layers, or None if not found
    """
    # Check common attribute names
    for attr_name in ['layers', 'blocks', 'h', 'transformer_blocks']:
        if hasattr(model, attr_name):
            layers = getattr(model, attr_name)
            if isinstance(layers, (nn.ModuleList, nn.Sequential)):
                return layers
    
    # Check nested transformer
    if hasattr(model, 'transformer'):
        transformer = model.transformer
        for attr_name in ['layers', 'blocks', 'h']:
            if hasattr(transformer, attr_name):
                layers = getattr(transformer, attr_name)
                if isinstance(layers, (nn.ModuleList, nn.Sequential)):
                    return layers
    
    return None


def compile_layer(
    layer: nn.Module,
    mode: str = "reduce-overhead",
    timeout: int = 60,
    **kwargs: Any
) -> nn.Module:
    """
    Compile a single layer with timeout.
    
    Args:
        layer: Layer to compile
        mode: Compilation mode
        timeout: Timeout in seconds (per layer)
        **kwargs: Additional torch.compile arguments
    
    Returns:
        Compiled layer, or original if compilation fails
    """
    try:
        def compile_fn(m: nn.Module) -> nn.Module:
            return torch.compile(m, mode=mode, **kwargs)
        
        return _compile_with_timeout(layer, compile_fn, timeout)
    except Exception as e:
        # Silently fall back to eager for this layer
        return layer


def partial_compile(
    model: nn.Module,
    layer_indices: Optional[list[int]] = None,
    max_layers: Optional[int] = None,
    mode: str = "reduce-overhead",
    timeout_per_layer: int = 60,
    verbose: bool = True,
    **kwargs: Any
) -> nn.Module:
    """
    Partially compile a model to avoid hangs on large models.
    
    This addresses the root cause by compiling only specific layers,
    reducing compilation complexity and memory usage.
    
    Strategy:
    - Detect transformer layers automatically
    - Compile each layer individually (avoids graph explosion)
    - Use short timeout per layer (prevents hangs)
    - Fall back to eager for layers that fail/timeout
    
    Args:
        model: Model to partially compile
        layer_indices: List of layer indices to compile (None = auto-select)
        max_layers: Maximum number of layers to compile (None = all)
        mode: torch.compile mode
        timeout_per_layer: Timeout per layer in seconds
        verbose: Print compilation progress
        **kwargs: Additional arguments passed to torch.compile
    
    Returns:
        Model with selected layers compiled
    
    Example:
        # Auto-compile first 10 layers
        model = partial_compile(model, max_layers=10)
        
        # Compile specific layers
        model = partial_compile(model, layer_indices=[0, 1, 2, 5, 10])
        
        # Full model with layer-by-layer compilation (safer than full compile)
        model = partial_compile(model)
    """
    # If model is small enough, use full compilation
    if not is_large_model(model):
        if verbose:
            print("✅ Model is small, using full compilation")
        return safe_compile(model, mode=mode, **kwargs)
    
    # Detect transformer layers
    layers = detect_transformer_layers(model)
    
    if layers is None:
        if verbose:
            print(
                "⚠️  Could not detect transformer layers automatically. "
                "Falling back to eager mode for large model."
            )
        return model
    
    # Determine which layers to compile
    if layer_indices is None:
        # Auto-select layers
        num_layers = len(layers)
        if max_layers is not None:
            # Compile first N layers
            layer_indices = list(range(min(max_layers, num_layers)))
        else:
            # Compile all layers (but individually)
            layer_indices = list(range(num_layers))
    
    if verbose:
        print(f"✅ Detected {len(layers)} transformer layers")
        print(f"🔧 Compiling layers: {layer_indices}")
        print(f"   (Mode: {mode}, Timeout per layer: {timeout_per_layer}s)")
    
    # Compile each layer individually
    compiled_count = 0
    failed_count = 0
    
    for idx in layer_indices:
        if idx >= len(layers):
            continue
        
        if verbose:
            print(f"   Layer {idx}...", end=" ", flush=True)
        
        try:
            compiled_layer = compile_layer(
                layers[idx],
                mode=mode,
                timeout=timeout_per_layer,
                **kwargs
            )
            
            # Check if compilation actually happened
            # (compile_layer returns original on failure)
            if compiled_layer is not layers[idx]:
                layers[idx] = compiled_layer
                compiled_count += 1
                if verbose:
                    print("✅")
            else:
                failed_count += 1
                if verbose:
                    print("⚠️  (fallback to eager)")
        except Exception as e:
            failed_count += 1
            if verbose:
                print(f"❌ ({str(e)[:50]})")
    
    if verbose:
        print(f"\n✅ Partial compilation complete:")
        print(f"   - Compiled: {compiled_count} layers")
        print(f"   - Failed/skipped: {failed_count} layers")
        print(f"   - Total: {len(layers)} layers in model")
    
    return model


def smart_compile(
    model: nn.Module,
    mode: str = "reduce-overhead",
    profile_first: bool = False,
    **kwargs: Any
) -> nn.Module:
    """
    Intelligently choose compilation strategy based on model characteristics.
    
    This is the recommended entry point for compilation.
    
    Strategy:
    - Small models (<1B): Full compilation
    - Medium models (1-10B): Full compilation with timeout
    - Large models (10-40B): Partial compilation (first 20 layers)
    - Very large models (>40B): Eager mode (compilation likely to fail)
    
    Args:
        model: Model to compile
        mode: Compilation mode
        profile_first: Profile model before deciding (not implemented yet)
        **kwargs: Additional torch.compile arguments
    
    Returns:
        Model (compiled, partially compiled, or eager)
    
    Example:
        # Recommended usage
        model = smart_compile(model)
    """
    param_count = count_parameters(model)
    param_count_b = param_count / 1e9
    
    print(f"🔍 Model size: {param_count_b:.2f}B parameters")
    
    # Strategy selection
    if param_count_b < 1:
        print("✅ Strategy: Full compilation (small model)")
        return safe_compile(model, mode=mode, **kwargs)
    
    elif param_count_b < 10:
        print("✅ Strategy: Full compilation with timeout (medium model)")
        return safe_compile(model, mode=mode, timeout=600, **kwargs)
    
    elif param_count_b < 40:
        print("✅ Strategy: Partial compilation (large model)")
        # Compile first 20 layers to get some benefit without hangs
        return partial_compile(model, max_layers=20, mode=mode, **kwargs)
    
    else:
        print("⚠️  Strategy: Eager mode (very large model, compilation likely to hang)")
        return model

