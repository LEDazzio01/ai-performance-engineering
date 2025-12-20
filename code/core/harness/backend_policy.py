"""Backend policy application for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch

from core.utils.compile_utils import configure_tf32, restore_tf32


class BackendPolicyName(str, Enum):
    PERFORMANCE = "performance"
    FP32_STRICT = "fp32_strict"


@dataclass(frozen=True)
class BackendPolicy:
    matmul_precision: str
    cudnn_precision: str


@dataclass
class BackendState:
    tf32_state: Tuple[Optional[str], Optional[str]]
    matmul_precision: Optional[str]
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    deterministic_algorithms: Optional[bool]


_POLICIES = {
    BackendPolicyName.PERFORMANCE: BackendPolicy(
        matmul_precision="high",
        cudnn_precision="high",
    ),
    BackendPolicyName.FP32_STRICT: BackendPolicy(
        matmul_precision="highest",
        cudnn_precision="highest",
    ),
}


def normalize_backend_policy(value: Optional[object]) -> BackendPolicyName:
    if value is None:
        return BackendPolicyName.PERFORMANCE
    if isinstance(value, BackendPolicyName):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for policy in BackendPolicyName:
            if policy.value == normalized:
                return policy
    raise ValueError(
        f"Unknown backend_policy '{value}'. "
        f"Expected one of: {', '.join(p.value for p in BackendPolicyName)}."
    )


def apply_backend_policy(policy_name: BackendPolicyName, deterministic: bool) -> BackendState:
    policy = _POLICIES[policy_name]
    prev_matmul_precision: Optional[str] = None
    if hasattr(torch, "get_float32_matmul_precision"):
        prev_matmul_precision = torch.get_float32_matmul_precision()

    tf32_state = configure_tf32(
        matmul_precision=policy.matmul_precision,
        cudnn_precision=policy.cudnn_precision,
    )

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(policy.matmul_precision)

    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    prev_cudnn_deterministic = torch.backends.cudnn.deterministic
    prev_deterministic: Optional[bool] = None
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        prev_deterministic = torch.are_deterministic_algorithms_enabled()

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

    return BackendState(
        tf32_state=tf32_state,
        matmul_precision=prev_matmul_precision,
        cudnn_benchmark=prev_cudnn_benchmark,
        cudnn_deterministic=prev_cudnn_deterministic,
        deterministic_algorithms=prev_deterministic,
    )


def restore_backend_policy(state: BackendState) -> None:
    restore_tf32(state.tf32_state)
    if state.matmul_precision is not None and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(state.matmul_precision)

    torch.backends.cudnn.benchmark = state.cudnn_benchmark
    torch.backends.cudnn.deterministic = state.cudnn_deterministic
    if state.deterministic_algorithms is not None:
        torch.use_deterministic_algorithms(state.deterministic_algorithms, warn_only=True)
