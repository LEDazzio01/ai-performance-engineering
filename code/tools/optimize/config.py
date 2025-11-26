"""
Configuration file support for the auto-optimizer.

Supports YAML and JSON configuration files.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = "anthropic"
    model: Optional[str] = None
    max_tokens: int = 16384
    temperature: float = 0.1
    api_key: Optional[str] = None
    
    def __post_init__(self):
        # Get API key from environment if not provided
        if self.api_key is None:
            if self.provider == "anthropic":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")


@dataclass
class OptimizationConfig:
    """Optimization settings."""
    max_iterations: int = 3
    target_speedup: float = 1.2
    techniques: List[str] = field(default_factory=lambda: [
        "torch_compile",
        "mixed_precision",
        "cuda_graphs",
        "kernel_fusion",
        "memory_optimization",
    ])
    fail_fast: bool = True


@dataclass
class ProfilingConfig:
    """Profiling settings."""
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    enable_memory_tracking: bool = True
    enable_trace: bool = True
    enable_flame_graph: bool = True


@dataclass
class TorchCompileConfig:
    """torch.compile specific settings."""
    mode: str = "reduce-overhead"
    backend: str = "inductor"
    fullgraph: bool = False
    dynamic: bool = False


@dataclass
class MixedPrecisionConfig:
    """Mixed precision settings."""
    dtype: str = "bfloat16"
    enabled_ops: List[str] = field(default_factory=lambda: [
        "linear",
        "matmul",
        "conv2d",
        "attention",
    ])


@dataclass
class OutputConfig:
    """Output settings."""
    save_intermediate: bool = True
    generate_report: bool = True
    output_format: str = "markdown"
    output_dir: Optional[str] = None


@dataclass
class OptimizerConfig:
    """Complete optimizer configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    torch_compile: TorchCompileConfig = field(default_factory=TorchCompileConfig)
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "OptimizerConfig":
        """Load configuration from YAML or JSON file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        content = path.read_text()
        
        if path.suffix in [".yaml", ".yml"]:
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                raise ImportError("PyYAML is required to load YAML config files. Install with: pip install pyyaml")
        elif path.suffix == ".json":
            data = json.loads(content)
        else:
            # Try JSON first, then YAML
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                try:
                    import yaml
                    data = yaml.safe_load(content)
                except ImportError:
                    raise ValueError(f"Unknown config format: {path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        if "llm" in data:
            config.llm = LLMConfig(**data["llm"])
        
        if "optimization" in data:
            config.optimization = OptimizationConfig(**data["optimization"])
        
        if "profiling" in data:
            config.profiling = ProfilingConfig(**data["profiling"])
        
        if "torch_compile" in data:
            config.torch_compile = TorchCompileConfig(**data["torch_compile"])
        
        if "mixed_precision" in data:
            config.mixed_precision = MixedPrecisionConfig(**data["mixed_precision"])
        
        if "output" in data:
            config.output = OutputConfig(**data["output"])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "max_tokens": self.llm.max_tokens,
                "temperature": self.llm.temperature,
            },
            "optimization": {
                "max_iterations": self.optimization.max_iterations,
                "target_speedup": self.optimization.target_speedup,
                "techniques": self.optimization.techniques,
                "fail_fast": self.optimization.fail_fast,
            },
            "profiling": {
                "warmup_iterations": self.profiling.warmup_iterations,
                "benchmark_iterations": self.profiling.benchmark_iterations,
                "enable_memory_tracking": self.profiling.enable_memory_tracking,
                "enable_trace": self.profiling.enable_trace,
                "enable_flame_graph": self.profiling.enable_flame_graph,
            },
            "torch_compile": {
                "mode": self.torch_compile.mode,
                "backend": self.torch_compile.backend,
                "fullgraph": self.torch_compile.fullgraph,
                "dynamic": self.torch_compile.dynamic,
            },
            "mixed_precision": {
                "dtype": self.mixed_precision.dtype,
                "enabled_ops": self.mixed_precision.enabled_ops,
            },
            "output": {
                "save_intermediate": self.output.save_intermediate,
                "generate_report": self.output.generate_report,
                "output_format": self.output.output_format,
                "output_dir": self.output.output_dir,
            },
        }
    
    def save(self, path: Union[str, Path], format: str = "yaml"):
        """Save configuration to file."""
        path = Path(path)
        data = self.to_dict()
        
        if format == "yaml":
            try:
                import yaml
                content = yaml.dump(data, default_flow_style=False, sort_keys=False)
            except ImportError:
                # Fall back to JSON
                content = json.dumps(data, indent=2)
                path = path.with_suffix(".json")
        else:
            content = json.dumps(data, indent=2)
        
        path.write_text(content)
        return path


def get_default_config() -> OptimizerConfig:
    """Get default configuration."""
    return OptimizerConfig()


def find_config_file(start_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find configuration file in current or parent directories.
    
    Looks for:
    - optimize_config.yaml
    - optimize_config.yml
    - optimize_config.json
    - .optimize.yaml
    - .optimize.json
    """
    if start_dir is None:
        start_dir = Path.cwd()
    
    config_names = [
        "optimize_config.yaml",
        "optimize_config.yml",
        "optimize_config.json",
        ".optimize.yaml",
        ".optimize.json",
    ]
    
    current = start_dir
    while current != current.parent:
        for name in config_names:
            config_path = current / name
            if config_path.exists():
                return config_path
        current = current.parent
    
    return None


def load_config(path: Optional[Union[str, Path]] = None) -> OptimizerConfig:
    """
    Load configuration from file or use defaults.
    
    Args:
        path: Path to config file (optional, will search if not provided)
        
    Returns:
        OptimizerConfig
    """
    if path is not None:
        return OptimizerConfig.from_file(path)
    
    # Try to find config file
    found = find_config_file()
    if found is not None:
        return OptimizerConfig.from_file(found)
    
    # Return defaults
    return get_default_config()


# Default configuration template
DEFAULT_CONFIG_TEMPLATE = """# Auto-Optimizer Configuration
# Save as optimize_config.yaml in your project root

llm:
  provider: anthropic  # or 'openai'
  model: null  # Uses default model for provider
  max_tokens: 16384
  temperature: 0.1
  # api_key: null  # Set via ANTHROPIC_API_KEY or OPENAI_API_KEY env var

optimization:
  max_iterations: 3
  target_speedup: 1.2
  techniques:
    - torch_compile
    - mixed_precision
    - cuda_graphs
    - kernel_fusion
    - memory_optimization
  fail_fast: true

profiling:
  warmup_iterations: 3
  benchmark_iterations: 10
  enable_memory_tracking: true
  enable_trace: true
  enable_flame_graph: true

torch_compile:
  mode: reduce-overhead  # 'default', 'reduce-overhead', 'max-autotune'
  backend: inductor
  fullgraph: false
  dynamic: false

mixed_precision:
  dtype: bfloat16  # or 'float16'
  enabled_ops:
    - linear
    - matmul
    - conv2d
    - attention

output:
  save_intermediate: true
  generate_report: true
  output_format: markdown  # or 'json', 'html'
  output_dir: null  # Uses input directory if null
"""


def generate_config_template(path: Union[str, Path] = "optimize_config.yaml"):
    """Generate a configuration template file."""
    path = Path(path)
    path.write_text(DEFAULT_CONFIG_TEMPLATE)
    return path

