# LLM Validation Checklist (Hardware Runs)

Use this checklist when executing the outstanding validation work on the
multi-GPU cluster. Capture the artifacts listed for each item and drop them in
`artifacts/` with a timestamp.

## 1. Multi-GPU Inference Benchmark (NVLink Mesh)
- [ ] Export NCCL env (`docs/nvlink_pcie_playbook.md`).
- [ ] Run: `torchrun --nproc_per_node=8 ch16/inference_server_load_test.py --duration 300 --target-qps 28 --attention-backend flex --fp8-mode transformer-engine --output-json artifacts/inference/full_mesh_$(date +%Y%m%d).json`
- [ ] Capture Nsight Systems trace via `scripts/capture_fp8_flexattention_traces.sh`.
- [ ] Update throughput/latency expectations in `docs/performance_baseline.md` if they drift >5%.

## 2. torch.compile Hang Investigation (40B+)
- [ ] Reproduce with and without `--compile-mode smart` on the 40B checkpoint.
- [ ] Use `python tools/torch_compile_repro.py --model llama40b --sequence-length 8192 --batch-size 4 --tensor-parallel-gpus 8 --output-dir artifacts/torch_compile_repro_$(date +%Y%m%d)`.
- [ ] Collect stdout/stderr logs and attach to the repro bundle.
- [ ] File/Update PyTorch bug report, link in `docs/TODO.md`.

## 3. Power Efficiency Sweep
- [ ] Run: `python tools/precision_power_sweep.py --sequence-length 4096 --model-layers 32 48 --model-d-model 5120 6144 --model-heads 40 48 --model-d-ff 20480 24576 --batch-size 2 3 4 --modes fp16 bf16 fp8_te --skip-compile --attention-backend sdpa --output-dir power_results/$(date +%Y%m%d)`.
- [ ] Summarise with `python tools/power_efficiency_analyzer.py power_results/<date>/*.json --output artifacts/power/summary_<date>.json`.
- [ ] Update `docs/power_efficiency_baselines.md` with new tables.

## 4. Long-Context Validation (32K+, FP8)
- [ ] Run: `python tools/long_context_validation.py --sequence-lengths 32768 65536 --tensor-parallel-gpus 8 --output-json artifacts/long_context/results_$(date +%Y%m%d).json`.
- [ ] Note peak memory, add findings to `docs/long_context_playbook.md` (create if absent).
- [ ] Update `docs/TODO.md` once 32K/64K runs pass without OOM.

Keep this checklist in version control. When an item is fully completed, tick it
off and reference the artifact path in `docs/TODO.md`.
