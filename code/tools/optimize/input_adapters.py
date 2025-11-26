"""
Input adapters for different code sources.

Supports:
- Local files
- Git repositories  
- Existing benchmark pairs
- Code from stdin
"""

import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


@dataclass
class CodeSource:
    """Represents a code file to optimize."""
    path: Path
    content: str
    name: str
    is_temporary: bool = False
    original_url: Optional[str] = None
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.is_temporary and self.path.exists():
            self.path.unlink()


class InputAdapter(ABC):
    """Base class for input adapters."""
    
    @abstractmethod
    def get_sources(self) -> Iterator[CodeSource]:
        """Yield code sources to optimize."""
        pass
    
    @abstractmethod
    def write_output(self, source: CodeSource, optimized_code: str) -> Path:
        """Write optimized code back."""
        pass


class FileAdapter(InputAdapter):
    """Adapter for local Python files."""
    
    def __init__(
        self,
        paths: List[Path],
        output_dir: Optional[Path] = None,
        suffix: str = "_optimized",
    ):
        self.paths = [Path(p) for p in paths]
        self.output_dir = Path(output_dir) if output_dir else None
        self.suffix = suffix
    
    def get_sources(self) -> Iterator[CodeSource]:
        for path in self.paths:
            if path.exists() and path.suffix == '.py':
                yield CodeSource(
                    path=path,
                    content=path.read_text(),
                    name=path.stem,
                )
    
    def write_output(self, source: CodeSource, optimized_code: str) -> Path:
        if self.output_dir:
            output_path = self.output_dir / f"{source.name}{self.suffix}.py"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_path = source.path.parent / f"{source.name}{self.suffix}.py"
        
        output_path.write_text(optimized_code)
        return output_path


class RepoAdapter(InputAdapter):
    """Adapter for Git repositories."""
    
    def __init__(
        self,
        repo_url: str,
        target_files: Optional[List[str]] = None,
        branch: str = "main",
        output_dir: Optional[Path] = None,
        gpu_patterns: Optional[List[str]] = None,
    ):
        self.repo_url = repo_url
        self.target_files = target_files
        self.branch = branch
        self.output_dir = Path(output_dir) if output_dir else None
        self.gpu_patterns = gpu_patterns or [
            'torch.cuda', 'cuda()', '.to(device)', 'torch.compile',
            'triton', 'cuda.amp', 'bfloat16', 'float16',
        ]
        self._temp_dir: Optional[Path] = None
        self._clone_dir: Optional[Path] = None
    
    def _clone_repo(self) -> Path:
        """Clone the repository to a temp directory."""
        if self._clone_dir is not None:
            return self._clone_dir
        
        self._temp_dir = Path(tempfile.mkdtemp(prefix="auto_optimize_"))
        self._clone_dir = self._temp_dir / "repo"
        
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", self.branch, 
             self.repo_url, str(self._clone_dir)],
            check=True,
            capture_output=True,
        )
        
        return self._clone_dir
    
    def _find_gpu_files(self, directory: Path) -> List[Path]:
        """Find Python files that contain GPU code."""
        gpu_files = []
        
        for py_file in directory.rglob('*.py'):
            if '__pycache__' in str(py_file):
                continue
            try:
                content = py_file.read_text()
                if any(pattern in content for pattern in self.gpu_patterns):
                    gpu_files.append(py_file)
            except:
                pass
        
        return gpu_files
    
    def get_sources(self) -> Iterator[CodeSource]:
        clone_dir = self._clone_repo()
        
        if self.target_files:
            files = [clone_dir / f for f in self.target_files]
        else:
            files = self._find_gpu_files(clone_dir)
        
        for path in files:
            if path.exists() and path.suffix == '.py':
                yield CodeSource(
                    path=path,
                    content=path.read_text(),
                    name=str(path.relative_to(clone_dir)),
                    original_url=self.repo_url,
                )
    
    def write_output(self, source: CodeSource, optimized_code: str) -> Path:
        if self.output_dir:
            output_path = self.output_dir / source.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(optimized_code)
            return output_path
        else:
            # Write back to cloned repo
            source.path.write_text(optimized_code)
            return source.path
    
    def cleanup(self):
        """Clean up cloned repository."""
        if self._temp_dir and self._temp_dir.exists():
            import shutil
            shutil.rmtree(self._temp_dir)


class BenchmarkAdapter(InputAdapter):
    """Adapter for existing benchmark pairs (baseline_ / optimized_)."""
    
    def __init__(
        self,
        directory: Path,
        threshold: float = 1.1,
        pattern: str = "optimized_*.py",
    ):
        self.directory = Path(directory)
        self.threshold = threshold
        self.pattern = pattern
        self._baseline_times: dict = {}
    
    def _get_baseline_time(self, optimized_path: Path) -> Optional[float]:
        """Get baseline timing for comparison."""
        baseline_name = optimized_path.name.replace("optimized_", "baseline_")
        baseline_path = optimized_path.parent / baseline_name
        
        if not baseline_path.exists():
            return None
        
        # Quick benchmark (simplified)
        # In production, use the actual benchmark harness
        return self._baseline_times.get(str(baseline_path))
    
    def get_sources(self) -> Iterator[CodeSource]:
        for path in self.directory.rglob(self.pattern):
            if '__pycache__' in str(path):
                continue
            
            # Get the benchmark name
            rel_path = path.relative_to(self.directory)
            chapter = rel_path.parts[0] if len(rel_path.parts) > 1 else ""
            name = path.stem.replace("optimized_", "")
            
            yield CodeSource(
                path=path,
                content=path.read_text(),
                name=f"{chapter}:{name}" if chapter else name,
            )
    
    def write_output(self, source: CodeSource, optimized_code: str) -> Path:
        source.path.write_text(optimized_code)
        return source.path


class StdinAdapter(InputAdapter):
    """Adapter for code from stdin."""
    
    def __init__(self, output_path: Optional[Path] = None):
        self.output_path = Path(output_path) if output_path else None
        self._temp_file: Optional[Path] = None
    
    def get_sources(self) -> Iterator[CodeSource]:
        import sys
        
        if sys.stdin.isatty():
            return
        
        content = sys.stdin.read()
        if not content.strip():
            return
        
        # Create temp file
        fd, path = tempfile.mkstemp(suffix='.py', prefix='stdin_optimize_')
        self._temp_file = Path(path)
        self._temp_file.write_text(content)
        
        yield CodeSource(
            path=self._temp_file,
            content=content,
            name="stdin",
            is_temporary=True,
        )
    
    def write_output(self, source: CodeSource, optimized_code: str) -> Path:
        if self.output_path:
            self.output_path.write_text(optimized_code)
            return self.output_path
        else:
            # Print to stdout
            print(optimized_code)
            return source.path
    
    def cleanup(self):
        if self._temp_file and self._temp_file.exists():
            self._temp_file.unlink()


def detect_input_type(input_str: str) -> Tuple[str, InputAdapter]:
    """
    Auto-detect input type and return appropriate adapter.
    
    Returns:
        Tuple of (input_type, adapter)
    """
    if not input_str or input_str == "-":
        return "stdin", StdinAdapter()
    
    if input_str.startswith(("http://", "https://", "git@")):
        return "repo", RepoAdapter(input_str)
    
    path = Path(input_str)
    
    if path.is_file():
        return "file", FileAdapter([path])
    
    if path.is_dir():
        # Check if it's a benchmark directory
        has_baselines = list(path.rglob("baseline_*.py"))
        has_optimized = list(path.rglob("optimized_*.py"))
        
        if has_baselines and has_optimized:
            return "benchmark", BenchmarkAdapter(path)
        else:
            # Treat as directory of files
            py_files = list(path.rglob("*.py"))
            return "directory", FileAdapter(py_files)
    
    raise ValueError(f"Unknown input type: {input_str}")

