import subprocess
import sys


def test_aisp_help_exits_cleanly():
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "aisp" in result.stdout.lower()


def test_mcp_server_import():
    import tools.mcp.server as mcp_server

    assert isinstance(mcp_server.TOOLS, dict)
    # ensure harness tools are registered
    assert "aisp_run_benchmarks" in mcp_server.TOOLS


def test_bench_list_targets_help():
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "bench", "list-targets", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "list-targets" in result.stdout


def test_bench_analyze_help():
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "bench", "analyze", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "analyze" in result.stdout.lower()


def test_bench_whatif_help():
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "bench", "whatif", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "whatif" in result.stdout.lower()
