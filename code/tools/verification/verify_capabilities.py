#!/usr/bin/env python3
"""Run all verification scripts in sequence."""

import os
import subprocess
import sys

TOOLS = [
    "verify_tma.py",
    "verify_thread_block_clusters.py",
    "verify_hbm.py",
    "verify_dynamic_parallelism.py",
    "verify_cooperative_launch.py",
    "verify_graph_launch.py",
    "verify_managed_memory.py",
    "verify_memory_pools.py",
    "verify_peer_access.py",
    "verify_power_state.py",
    "verify_ecc_mig.py",
    "verify_sm_resources.py",
]

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tools")

FAILED = False
for script in TOOLS:
    path = os.path.join(SCRIPT_DIR, script)
    if not os.path.exists(path):
        print(f"[verify_capabilities] Missing script: {script}")
        FAILED = True
        continue
    print(f"\n=== running {script} ===")
    try:
        subprocess.check_call([sys.executable, path], cwd=SCRIPT_DIR)
    except subprocess.CalledProcessError as exc:
        print(f"[verify_capabilities] {script} exited with status {exc.returncode}")
        FAILED = True

if FAILED:
    sys.exit(1)

print("\n[verify_capabilities] All capability checks passed.")
