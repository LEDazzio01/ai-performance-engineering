"""Shared helpers for async input pipeline benchmarks and sweeps."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


@dataclass
class PipelineConfig:
    """Configuration knobs for the async input pipeline."""

    batch_size: int = 16
    feature_shape: Tuple[int, int, int] = (3, 64, 64)
    dataset_size: int = 64
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    pin_memory: bool = False
    non_blocking: bool = False
    use_copy_stream: bool = False


class _SyntheticImageDataset(Dataset):
    """Pre-generated CPU dataset to avoid per-sample allocation overhead."""

    def __init__(self, length: int, feature_shape: Tuple[int, int, int]):
        self.length = length
        self.data = torch.randn((length, *feature_shape), dtype=torch.float32)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _build_dataloader(cfg: PipelineConfig) -> Iterable[torch.Tensor]:
    """Construct a DataLoader with the requested overlap knobs."""

    dataset = _SyntheticImageDataset(cfg.dataset_size, cfg.feature_shape)
    kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "drop_last": True,
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.prefetch_factor is not None and cfg.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.prefetch_factor

    return DataLoader(dataset, **kwargs)


class AsyncInputPipelineBenchmark(BaseBenchmark):
    """Benchmark that measures H2D overlap for a simple vision pipeline."""

    def __init__(self, cfg: Optional[PipelineConfig] = None, label: str = "async_input_pipeline"):
        super().__init__()
        self.cfg = cfg or PipelineConfig()
        self.label = label

        self.loader_iter: Optional[Iterable[torch.Tensor]] = None
        self.loader: Optional[DataLoader] = None
        self.model: Optional[nn.Module] = None
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.compute_stream: Optional[torch.cuda.Stream] = None

    def setup(self) -> None:
        torch.manual_seed(2025)
        torch.backends.cudnn.benchmark = False

        self.loader = _build_dataloader(self.cfg)
        self.loader_iter = iter(self.loader)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.cfg.feature_shape[0] * self.cfg.feature_shape[1] * self.cfg.feature_shape[2], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
        ).to(self.device)

        self.copy_stream = torch.cuda.Stream() if self.cfg.use_copy_stream else None
        self.compute_stream = torch.cuda.current_stream()

        self.register_workload_metadata(samples_per_iteration=self.cfg.batch_size)
        # Lightweight pre-warm to amortize first-call overhead.
        warm = torch.randn((1, *self.cfg.feature_shape), device=self.device)
        with torch.no_grad():
            _ = self.model(warm)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        assert self.loader_iter is not None, "Loader not initialized"
        assert self.model is not None, "Model not initialized"

        try:
            batch_cpu = next(self.loader_iter)
        except StopIteration:
            assert self.loader is not None
            self.loader_iter = iter(self.loader)
            batch_cpu = next(self.loader_iter)
        with self._nvtx_range(self.label):
            if self.copy_stream is not None:
                with torch.cuda.stream(self.copy_stream):
                    batch_gpu = batch_cpu.to(self.device, non_blocking=self.cfg.non_blocking)
                self.compute_stream.wait_stream(self.copy_stream)
            else:
                batch_gpu = batch_cpu.to(self.device, non_blocking=self.cfg.non_blocking)

            with torch.no_grad():
                _ = self.model(batch_gpu)

    def teardown(self) -> None:
        self.loader_iter = None
        self.loader = None
        self.model = None
        self.copy_stream = None
        self.compute_stream = None
        super().teardown()

    def get_config(self) -> Optional[BenchmarkConfig]:
        # Keep iterations modest to make sweeps quick.
        return BenchmarkConfig(
            iterations=1,
            warmup=5,
            timeout_seconds=120,
            measurement_timeout_seconds=120,
            use_subprocess=False,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return async input pipeline configuration metrics."""
        bytes_per_sample = 4 * self.cfg.feature_shape[0] * self.cfg.feature_shape[1] * self.cfg.feature_shape[2]
        bytes_per_batch = bytes_per_sample * self.cfg.batch_size
        return {
            f"{self.label}.batch_size": float(self.cfg.batch_size),
            f"{self.label}.num_workers": float(self.cfg.num_workers),
            f"{self.label}.pin_memory": float(self.cfg.pin_memory),
            f"{self.label}.non_blocking": float(self.cfg.non_blocking),
            f"{self.label}.use_copy_stream": float(self.cfg.use_copy_stream),
            f"{self.label}.bytes_per_batch": float(bytes_per_batch),
        }
