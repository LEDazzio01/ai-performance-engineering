import json
import os
import sys
from pathlib import Path

import pytest

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_microbench_disk_and_loopback():
    from tools import microbench

    disk = microbench.disk_io_test(file_size_mb=1, block_size_kb=64)
    assert "read_gbps" in disk and disk["read_gbps"] is not None

    net = microbench.network_loopback_test(size_mb=1, port=50507)
    assert "throughput_gbps" in net and net["throughput_gbps"] is not None


@pytest.mark.cuda
def test_microbench_pcie_mem_tensor_sfu_cuda():
    """These assume CUDA + torch available on this hardware."""
    import torch
    from tools import microbench

    assert torch.cuda.is_available(), "CUDA expected to be available"

    pcie = microbench.pcie_bandwidth_test(size_mb=1, iters=1)
    assert pcie.get("h2d_gbps") is not None
    assert pcie.get("d2h_gbps") is not None

    mem = microbench.mem_hierarchy_test(size_mb=1, stride=64)
    assert mem.get("bandwidth_gbps") is not None

    tensor = microbench.tensor_core_bench(size=256, precision="fp16")
    assert tensor.get("tflops") is not None

    sfu = microbench.sfu_bench(size=1_000_000)
    assert sfu.get("gops") is not None


def test_nsys_ncu_available_keys():
    from tools.profiling.nsight_automation import NsightAutomation

    automation = NsightAutomation(Path("artifacts/mcp-profiles"))
    # We only assert keys exist; availability may depend on environment.
    assert "nsys_available" in {"nsys_available": automation.nsys_available}
    assert "ncu_available" in {"ncu_available": automation.ncu_available}


def test_mcp_tools_registration():
    from tools.mcp import server

    required = {
        "aisp_test_disk",
        "aisp_test_pcie",
        "aisp_test_mem_hierarchy",
        "aisp_test_tensor_core",
        "aisp_test_sfu",
        "aisp_test_network_loopback",
        "aisp_profile_nsys",
        "aisp_profile_ncu",
        "aisp_nsys_summary",
        "aisp_compare_nsys",
        "aisp_compare_ncu",
        "aisp_export_csv",
        "aisp_export_pdf",
        "aisp_export_html",
        "aisp_system_capabilities",
        "aisp_full_system_analysis",
        "aisp_available_benchmarks",
        "aisp_benchmark_targets",
        "aisp_nsys_ncu_available",
    }
    assert required.issubset(set(server.TOOLS.keys()))


def test_cli_microbench_disk():
    import subprocess, sys
    cmd = [sys.executable, "tools/cli/aisp.py", "microbench", "disk", "--file_size_mb", "1", "--block_size_kb", "64"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
