#!/usr/bin/env python3
"""Best-effort GPU reset utility without sudo access.

The script does three things:
1. Uses ``nvidia-smi`` to enumerate active compute processes.
2. Sends SIGTERM/SIGKILL to any remaining processes we don't own.
3. Invokes ``nvidia-smi --gpu-reset`` as a best effort (ignoring failures).

This mirrors the behavior we had previously when the harness invoked the
reset helper after a runaway benchmark. Having this utility in-tree keeps
the harness happy and prevents zombie CUDA contexts from cascading into
later tests.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Iterable, List


def run_command(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def list_gpu_pids() -> List[int]:
    """Return a list of compute PIDs reported by nvidia-smi."""
    smi = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ]
    )
    if smi.returncode != 0:
        print(
            "[reset_gpu.py] WARNING: nvidia-smi unavailable; "
            f"stdout={smi.stdout.strip()} stderr={smi.stderr.strip()}",
            file=sys.stderr,
        )
        return []
    pids: List[int] = []
    for line in smi.stdout.strip().splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid > 0:
            pids.append(pid)
    return pids


def kill_process(pid: int, timeout: float = 2.0) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        print(f"[reset_gpu.py] WARNING: no permission to terminate PID {pid}", file=sys.stderr)
        return

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process_alive(pid):
            return
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        print(f"[reset_gpu.py] WARNING: no permission to SIGKILL PID {pid}", file=sys.stderr)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we lack permission; treat as alive
        return True


def maybe_reset_gpu(index: int | None) -> None:
    base_cmd = ["nvidia-smi", "--gpu-reset"]
    if index is not None:
        base_cmd.extend(["-i", str(index)])
    result = run_command(base_cmd)
    if result.returncode != 0:
        print(
            "[reset_gpu.py] INFO: GPU reset command failed "
            f"(this is expected without admin privileges): {result.stderr.strip()}",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best-effort GPU reset helper.")
    parser.add_argument("--reason", default="unspecified", help="Reason reported by the harness.")
    parser.add_argument("--device", type=int, default=None, help="GPU index to reset (optional).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[reset_gpu.py] Reset requested (reason: {args.reason})")

    my_pid = os.getpid()
    pids = [pid for pid in list_gpu_pids() if pid != my_pid]
    if pids:
        print(f"[reset_gpu.py] Terminating GPU processes: {pids}")
        for pid in pids:
            kill_process(pid)
    else:
        print("[reset_gpu.py] No foreign GPU processes found.")

    maybe_reset_gpu(args.device)
    print("[reset_gpu.py] Best-effort reset complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
