# AI Systems Performance Engineering

Production playbook for standing up, validating, and tuning PyTorch LLM workloads on 8x NVIDIA B200 systems.

---

## Overview
**Target hardware:** 8x NVIDIA B200 (180 GB HBM3e, 148 SMs)  
**Reference stack:** Ubuntu 22.04, CUDA 13.0, PyTorch 2.9

The repository packages everything needed to:
- Provision a reproducible software stack (`setup.sh`) for new lab machines.
- Exercise and benchmark the platform end-to-end before deploying workloads.
- Dive into architecture, profiling, and troubleshooting guides when deeper work is required.

If you are exploring the broader curriculum, see the chapter index in `docs/README.md`.

## Quick Start

### Prerequisites
- Root access to the host (the setup script installs NVIDIA driver 580+, CUDA 13.0, and dependencies)
- Python 3.10+ on the path (the setup script installs required packages in-place)
- Network access to fetch Python wheels and Nsight tooling

### Setup
1. Clone and enter the repository:
   ```bash
   git clone <repo-url> && cd ai-performance-engineering/code
   ```
2. Run the automated bootstrap (installs drivers, CUDA, Python deps, and validation tooling):
   ```bash
   sudo ./setup.sh
   ```
3. If the script upgrades the driver, reboot and rerun `sudo ./setup.sh` to finish verification.
4. Optional extras—Docker images, multi-node orchestration, or single-GPU serving—documented in:
   - `docs/guides/READY_TO_RUN_GUIDE.md` (benchmark presets and profiles)
   - `docs/guides/single_gpu_serving_guide.md` (production serving playbook)
   - `docs/playbooks/nvlink_pcie_playbook.md` (topology tuning)

### Configuration
- Environment knobs for specific benchmarks: see `docs/guides/READY_TO_RUN_GUIDE.md`
- Architecture-specific tuning (tensor parallelism, FlexAttention settings, etc.): see `docs/guides/architecture_guides.md` and `docs/guides/OPTIMIZATION_QUICK_START.md`

## Verification
Run the quick smoke tests after installation:
1. Confirm the hardware and driver:
   ```bash
   nvidia-smi
   ```
   Expect eight B200 GPUs and driver 580+.
2. Execute the automated test suite (covers CUDA samples and PyTorch checks):
   ```bash
   ./run_all_tests.sh
   ```
3. Verify TMA support on Grace-Blackwell GB10 (if applicable):
   ```bash
   ./verify_tma_sm121.py
   ```
4. Capture a baseline performance snapshot:
   ```bash
   python3 benchmark_peak.py
   ```
   Compare results with `docs/reference/performance_baseline.md`. For a full validation matrix, follow `docs/planning/llm_validation_checklist.md`.

## Deep Dives & Troubleshooting
- **Architecture & deployment:** `docs/guides/architecture_guides.md`, `docs/guides/migration_to_sm100.md`, `docs/playbooks/moe_deployment_playbook.md`
- **Performance & profiling:** `docs/guides/OPTIMIZATION_QUICK_START.md`, `docs/reference/nsight_fp8_flexattention.md`, `docs/playbooks/long_context_playbook.md`
- **Common issues:** `docs/planning/common_issues_faq.md`, `docs/playbooks/torch_compile_troubleshooting.md`, `docs/playbooks/nvlink_pcie_playbook.md`
- **Tooling quick reference:** `docs/reference/TOOLS_QUICK_REFERENCE.md`, `docs/guides/READY_TO_RUN_GUIDE.md`

## More Resources
- **Validation status:** `docs/planning/llm_validation_checklist.md`, `docs/reference/performance_baseline.md`, `docs/reference/power_efficiency_baselines.md`
- **Playbooks & walkthroughs:** `docs/README.md` (chapter index), `docs/guides/8xb200_load_testing_guide.md`, `docs/guides/single_gpu_serving_guide.md`
- **Active work & enhancements:** `docs/planning/TODO.md`, `docs/planning/future_optimizations.md`
- **Reference materials:** `docs/reference/MODEL_SIZE_RECOMMENDATIONS.md`, `docs/reference/B200_CUDA13_AUDIT.md`

## Repository Layout
```text
code/
├── setup.sh                # End-to-end system bootstrap
├── docs/                   # Architecture, optimization, troubleshooting guides
├── ch1...ch20/             # Chapter walkthroughs with focused READMEs
├── scripts/                # Capture and profiling helpers
├── tools/                  # Verification utilities
└── tests/                  # Automated checks invoked by run_all_tests.sh
```

## Next Steps
- Track open work in `docs/planning/TODO.md` and planned enhancements in `docs/planning/future_optimizations.md`
- Record measured metrics or new findings in `docs/reference/performance_baseline.md` and related docs under `docs/`
- For questions or new issues, start with `docs/planning/common_issues_faq.md` then escalate via the team's issue tracker
