"""Benchmark contract definition and validation.

Defines the required interface that all benchmarks must implement,
and provides utilities for validating benchmark compliance.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class BenchmarkContract:
    """Defines the contract that all benchmarks must follow.
    
    The contract includes required methods that must be implemented by all benchmarks.
    
    Verification Enforcement:
    - get_input_signature(): Returns workload description for equivalence checking
    - validate_result(): Returns error message or None for output validation
    - get_verify_output(): Returns output tensor(s) for correctness comparison
    
    See core.benchmark.verification for enforcement phase configuration.
    """
    
    # Required methods that must be implemented
    # These are enforced based on the current EnforcementPhase (DETECT, QUARANTINE, GATE)
    REQUIRED_METHODS: Set[str] = {
        "setup",
        "benchmark_fn",
        "teardown",
    }
    
    # Verification methods - required for correctness validation
    # Missing these methods will quarantine the benchmark in QUARANTINE/GATE phases
    VERIFICATION_REQUIRED_METHODS: Set[str] = {
        "get_input_signature",  # Returns workload description dict
        "validate_result",  # Returns error message or None
        "get_verify_output",  # Returns output tensor(s) for comparison
        "get_output_tolerance",  # Returns ToleranceSpec for custom tolerances
    }
    
    # Optional methods that enhance verification but have sensible defaults
    RECOMMENDED_METHODS: Set[str] = {
        "get_config",
        "get_equivalence_fn",  # Returns custom comparator function
        "get_workload_metadata",  # Returns WorkloadMetadata for invariant checking
        "get_verify_inputs",  # Returns input tensor(s) for aliasing detection
    }
    
    # Methods for verification skip control (use sparingly with justification)
    VERIFICATION_SKIP_METHODS: Set[str] = {
        "skip_input_verification",
        "skip_output_verification",
    }
    
    # Required attributes (if using BaseBenchmark)
    REQUIRED_ATTRIBUTES: Set[str] = {
        "device",  # Set by BaseBenchmark.__init__
    }
    
    @staticmethod
    def validate_benchmark_class_ast(class_node: ast.ClassDef) -> Tuple[List[str], List[str]]:
        """Validate benchmark class using AST (side-effect free).
        
        Args:
            class_node: AST ClassDef node
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Get method names
        method_names = {
            item.name for item in class_node.body
            if isinstance(item, ast.FunctionDef)
        }
        
        # Check required methods
        for method_name in BenchmarkContract.REQUIRED_METHODS:
            if method_name not in method_names:
                errors.append(f"Missing required method: {method_name}()")
        
        # Check verification required methods (warn in AST mode since we don't know enforcement phase)
        for method_name in BenchmarkContract.VERIFICATION_REQUIRED_METHODS:
            if method_name not in method_names:
                errors.append(f"Missing verification method: {method_name}()")
        
        # Check recommended methods
        for method_name in BenchmarkContract.RECOMMENDED_METHODS:
            if method_name not in method_names:
                warnings.append(f"Missing recommended method: {method_name}()")
        
        # Check method signatures
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                method_name = item.name
                if method_name in BenchmarkContract.REQUIRED_METHODS:
                    # Check signature - should only take self
                    args = item.args.args
                    if len(args) > 1:
                        # More than just self
                        # Allow *args and **kwargs
                        has_var_args = any(arg.arg == '*' for arg in args) or any(arg.arg == '**' for arg in args)
                        if not has_var_args:
                            errors.append(f"{method_name}() should take no arguments (except self)")
        
        # Check for docstring
        if not ast.get_docstring(class_node):
            warnings.append("Class should have a docstring")
        
        # Check benchmark_fn docstring
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "benchmark_fn":
                if not ast.get_docstring(item):
                    warnings.append("benchmark_fn() should have a docstring describing what it benchmarks")
                break
        
        return errors, warnings
    
    @staticmethod
    def validate_benchmark_class(cls: type) -> Tuple[bool, List[str]]:
        """Validate that a benchmark class follows the contract.
        
        Args:
            cls: Benchmark class to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required methods
        for method_name in BenchmarkContract.REQUIRED_METHODS:
            if not hasattr(cls, method_name):
                errors.append(f"Missing required method: {method_name}()")
                continue
            
            method = getattr(cls, method_name)
            if not callable(method):
                errors.append(f"{method_name} is not callable")
                continue
            
            # Check method signature (should take no args except self)
            sig = inspect.signature(method)
            params = list(sig.parameters.values())
            if len(params) > 1:  # More than just 'self'
                # Allow *args and **kwargs
                has_var_args = any(p.kind == p.VAR_POSITIONAL for p in params)
                has_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)
                if not (has_var_args or has_var_kwargs):
                    errors.append(f"{method_name}() should take no arguments (except self)")
        
        # Check verification-required methods
        for method_name in BenchmarkContract.VERIFICATION_REQUIRED_METHODS:
            if not hasattr(cls, method_name):
                errors.append(f"Missing verification method: {method_name}()")
                continue
            method = getattr(cls, method_name)
            if not callable(method):
                errors.append(f"{method_name} is not callable")
                continue
        
        # Check recommended methods (warn, don't fail)
        warnings = []
        for method_name in BenchmarkContract.RECOMMENDED_METHODS:
            if not hasattr(cls, method_name):
                warnings.append(f"Missing recommended method: {method_name}()")
        
        # Check for docstring (recommended)
        if not cls.__doc__:
            warnings.append("Class should have a docstring describing the benchmark")
        
        # Check benchmark_fn docstring (recommended)
        if hasattr(cls, "benchmark_fn"):
            benchmark_fn = getattr(cls, "benchmark_fn")
            if callable(benchmark_fn) and not benchmark_fn.__doc__:
                warnings.append("benchmark_fn() should have a docstring describing what it benchmarks")
        
        return len(errors) == 0, errors + warnings
    
    @staticmethod
    def validate_benchmark_instance(benchmark: Any, run_setup: bool = False) -> Tuple[bool, List[str]]:
        """Validate that a benchmark instance follows the contract.
        
        Args:
            benchmark: Benchmark instance to validate
            run_setup: If True, actually call setup() to test execution (default: False).
                       WARNING: This will allocate GPU memory and run code - only use
                       when explicitly needed, not in pre-commit hooks or CI linting.
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required methods exist and are callable
        for method_name in BenchmarkContract.REQUIRED_METHODS:
            if not hasattr(benchmark, method_name):
                errors.append(f"Missing required method: {method_name}()")
                continue
            
            method = getattr(benchmark, method_name)
            if not callable(method):
                errors.append(f"{method_name} is not callable")
        
        # Verification-required methods must also be present
        for method_name in BenchmarkContract.VERIFICATION_REQUIRED_METHODS:
            if not hasattr(benchmark, method_name):
                errors.append(f"Missing verification method: {method_name}()")
                continue
            method = getattr(benchmark, method_name)
            if not callable(method):
                errors.append(f"{method_name} is not callable")
        
        # Only call setup() if explicitly requested (structural validation only by default)
        if run_setup and hasattr(benchmark, "setup") and callable(benchmark.setup):
            try:
                benchmark.setup()
            except Exception as e:
                errors.append(f"setup() raised exception: {type(e).__name__}: {e}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def check_verification_compliance(benchmark: Any) -> Tuple[bool, List[str], List[str]]:
        """Check if a benchmark is compliant with verification enforcement requirements.
        
        This is separate from validate_benchmark_instance as verification requirements
        are enforced according to the enforcement phase (DETECT, QUARANTINE, GATE).
        
        Args:
            benchmark: Benchmark instance to check
            
        Returns:
            Tuple of (is_compliant, errors, warnings)
            - In DETECT phase: all issues are warnings
            - In QUARANTINE/GATE phase: missing methods are errors
        """
        from core.benchmark.verification import get_enforcement_phase, EnforcementPhase
        
        errors: List[str] = []
        warnings: List[str] = []
        phase = get_enforcement_phase()
        
        # Check verification required methods
        for method_name in BenchmarkContract.VERIFICATION_REQUIRED_METHODS:
            if not hasattr(benchmark, method_name):
                msg = f"Missing verification method: {method_name}()"
                if phase == EnforcementPhase.DETECT:
                    warnings.append(msg)
                else:
                    errors.append(msg)
            else:
                method = getattr(benchmark, method_name)
                if not callable(method):
                    msg = f"Verification method {method_name} is not callable"
                    if phase == EnforcementPhase.DETECT:
                        warnings.append(msg)
                    else:
                        errors.append(msg)
        
        # Check for skip flags (always warn - these should have justification)
        for skip_method in BenchmarkContract.VERIFICATION_SKIP_METHODS:
            if hasattr(benchmark, skip_method):
                try:
                    method = getattr(benchmark, skip_method)
                    if callable(method) and method():
                        # Check for justification
                        justification = getattr(benchmark, f"{skip_method}_reason", None)
                        if not justification:
                            warnings.append(
                                f"{skip_method}() returns True without justification attribute"
                            )
                except Exception:
                    pass
        
        # Check for legacy skip flags (deprecated)
        for legacy_flag in ["skip_output_check", "skip_input_check", "skip_verification"]:
            if hasattr(benchmark, legacy_flag):
                value = getattr(benchmark, legacy_flag)
                if value:
                    msg = f"Legacy skip flag '{legacy_flag}' is deprecated; use skip_*_verification() methods"
                    if phase == EnforcementPhase.DETECT:
                        warnings.append(msg)
                    else:
                        errors.append(msg)
        
        return len(errors) == 0, errors, warnings


def get_benchmark_class_from_module(module_path: Path) -> Optional[type]:
    """Extract benchmark class from a Python module file.
    
    Args:
        module_path: Path to Python module file
        
    Returns:
        Benchmark class if found, None otherwise
    """
    try:
        # Read and parse the file
        source = module_path.read_text()
        tree = ast.parse(source, filename=str(module_path))
        
        # Find classes that have benchmark_fn method
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if class has benchmark_fn method
                has_benchmark_fn = any(
                    isinstance(item, ast.FunctionDef) and item.name == "benchmark_fn"
                    for item in node.body
                )
                if has_benchmark_fn:
                    # Try to import and return the class
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        return getattr(module, node.name, None)
        
        return None
    except Exception:
        return None


def check_benchmark_file_ast(file_path: Path) -> Tuple[bool, List[str], List[str]]:
    """Check benchmark file using AST parsing (side-effect free).
    
    Args:
        file_path: Path to benchmark Python file
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    if not file_path.exists():
        return False, [f"File does not exist: {file_path}"], []
    
    if not file_path.suffix == ".py":
        return False, [f"Not a Python file: {file_path}"], []
    
    try:
        source = file_path.read_text()
        tree = ast.parse(source, filename=str(file_path))
        
        # Check for get_benchmark function
        has_get_benchmark = False
        benchmark_classes = []
        
        # Walk tree to find get_benchmark function and benchmark classes
        for node in ast.walk(tree):
            # Check for module-level get_benchmark function
            if isinstance(node, ast.FunctionDef) and node.name == "get_benchmark":
                # Check if it's at module level (not inside a class)
                # We'll check this by seeing if it's directly in tree.body
                is_module_level = node in tree.body
                if is_module_level:
                    has_get_benchmark = True
                    # Check signature - should take no args
                    if len(node.args.args) > 0:
                        errors.append("get_benchmark() should take no arguments")
            
            # Find classes with benchmark_fn method
            if isinstance(node, ast.ClassDef):
                has_benchmark_fn = any(
                    isinstance(item, ast.FunctionDef) and item.name == "benchmark_fn"
                    for item in node.body
                )
                if has_benchmark_fn:
                    benchmark_classes.append(node.name)
                    # Validate class structure
                    class_errors, class_warnings = BenchmarkContract.validate_benchmark_class_ast(node)
                    errors.extend(class_errors)
                    warnings.extend(class_warnings)
        
        if not has_get_benchmark and not benchmark_classes:
            errors.append("No get_benchmark() function or benchmark class found")
        
        # Check for docstring (recommended)
        module_docstring = ast.get_docstring(tree)
        if not module_docstring:
            warnings.append("Module should have a docstring")
        
    except SyntaxError as e:
        errors.append(f"Syntax error: {e}")
    except Exception as e:
        errors.append(f"Failed to parse file: {type(e).__name__}: {e}")
    
    return len(errors) == 0, errors, warnings


def check_benchmark_file(file_path: Path, run_setup: bool = False) -> Tuple[bool, List[str], List[str]]:
    """Check if a benchmark file follows the contract.
    
    By default, uses AST parsing for side-effect free validation.
    Only imports and instantiates if run_setup=True.
    
    Args:
        file_path: Path to benchmark Python file
        run_setup: If True, actually import and instantiate benchmark (default: False).
                   WARNING: This will execute module-level code and constructors,
                   which may require CUDA. Use only when explicitly needed.
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    # By default, use AST parsing (side-effect free)
    if not run_setup:
        return check_benchmark_file_ast(file_path)
    
    # If run_setup=True, do full validation with instantiation
    errors = []
    warnings = []
    
    if not file_path.exists():
        return False, [f"File does not exist: {file_path}"], []
    
    if not file_path.suffix == ".py":
        return False, [f"Not a Python file: {file_path}"], []
    
    # Try to find get_benchmark function or benchmark class
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for get_benchmark function
            if hasattr(module, "get_benchmark"):
                get_benchmark = getattr(module, "get_benchmark")
                if callable(get_benchmark):
                    try:
                        benchmark = get_benchmark()
                        is_valid, issues = BenchmarkContract.validate_benchmark_instance(benchmark, run_setup=run_setup)
                        if not is_valid:
                            errors.extend(issues)
                        else:
                            # Separate warnings from errors
                            for issue in issues:
                                if issue.startswith("Missing recommended") or "should have" in issue:
                                    warnings.append(issue)
                                else:
                                    errors.append(issue)
                    except Exception as e:
                        errors.append(f"get_benchmark() raised exception: {type(e).__name__}: {e}")
                else:
                    errors.append("get_benchmark is not callable")
            else:
                # Try to find benchmark class
                benchmark_class = get_benchmark_class_from_module(file_path)
                if benchmark_class:
                    is_valid, issues = BenchmarkContract.validate_benchmark_class(benchmark_class)
                    if not is_valid:
                        errors.extend(issues)
                    else:
                        # Separate warnings from errors
                        for issue in issues:
                            if issue.startswith("Missing recommended") or "should have" in issue:
                                warnings.append(issue)
                            else:
                                errors.append(issue)
                else:
                    errors.append("No get_benchmark() function or benchmark class found")
    except Exception as e:
        errors.append(f"Failed to load module: {type(e).__name__}: {e}")
    
    return len(errors) == 0, errors, warnings
