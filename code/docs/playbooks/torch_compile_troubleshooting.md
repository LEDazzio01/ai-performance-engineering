# torch.compile Troubleshooting Guide

This playbook helps you decide when and how to use `torch.compile` with the
Blackwell LLM workloads in this repository. The focus is on large language
models (10B–120B parameters) and production inference paths.

## 1. Know the Risk Profile

| Model Size | Recommendation | Rationale |
|------------|----------------|-----------|
| <10B       | Safe to compile (`--compile-mode reduce-overhead`) | Fits within compiler memory budget, warmup <2 minutes |
| 10B–40B    | Use `common.torch_compile_safe.safe_compile()` or `--compile-mode smart` | Partial compilation avoids graph explosion |
| >40B       | Default to eager (`--skip-compile`) | Inductor frequently hangs or runs slower than eager |

For quick triage, check the parameter count emitted by
`common.torch_compile_safe.count_parameters()` or the warmup log emitted by the
benchmark scripts. Anything above 40B should default to eager mode unless a
compelling regression fix lands upstream.

## 2. Use the Safe Wrapper

Prefer the safe wrapper over raw `torch.compile`:

```python
from common.torch_compile_safe import safe_compile

model = load_model(...)
model = safe_compile(
    model,
    mode="reduce-overhead",
    timeout=600,
    skip_if_large=True,   # auto-skip for >40B params
)
```

Benefits:
- Detects large models and skips compilation automatically
- Adds a configurable timeout (default 10 minutes)
- Falls back to eager mode on failure while logging a warning

CLI equivalent in our benchmarks:

```bash
python ch16/test_gpt_large_optimized.py \
    --compile-mode smart \
    --tensor-parallel-gpus 4 \
    --sequence-length 8192
```

`--compile-mode smart` maps to the safe wrapper with size-based strategy
selection.

## 3. When Compilation Hangs

Symptoms:
- Process stalls during warmup
- `torch._dynamo` debug logs stop progressing
- GPUs idle but CPU thread pinned at 100%

Immediate actions:
1. Rerun with the safe wrapper (enables timeout + fallback).
2. Set `TORCH_COMPILE_TIMEOUT=300` to shorten the wait.
3. Use `--skip-compile` to gather eager-mode throughput as a baseline.

Example fallback command:

```bash
python ch16/inference_server_load_test.py \
  --skip-compile \
  --attention-backend flex \
  --fp8-mode transformer-engine \
  --output-json artifacts/load_test_eager.json
```

## 4. Performance Validation Checklist

Before enabling `torch.compile` in production, run:

1. **Eager baseline**  
   `python ... --skip-compile --output-json artifacts/eager_baseline.json`
2. **Compiled run**  
   `python ... --compile-mode reduce-overhead --output-json artifacts/compiled.json`
3. **Compare**  
   `python tools/compare_benchmarks.py artifacts/eager_baseline.json artifacts/compiled.json`

If speedup <5% or latency regresses, prefer eager mode.

## 5. Filing Upstream Bugs

Capture a minimal reproduction pack:

```bash
python tools/torch_compile_repro.py \
  --model llama40b \
  --sequence-length 8192 \
  --batch-size 4 \
  --tensor-parallel-gpus 8 \
  --output-dir artifacts/torch_compile_repro
```

Attach the generated tarball when opening issues with PyTorch or Triton. Include:
- `torch` and `torchvision` versions
- GPU model and driver version
- Exact command and flags

## 6. Decision Tree

```
Is model >40B params?
 ├─ Yes → Use eager mode (`--skip-compile`), run safe wrapper only for per-layer tests
 └─ No
     └─ Did benchmarking show ≥5% gain?
         ├─ Yes → Keep `--compile-mode reduce-overhead` and monitor latency drift
         └─ No  → Stay on eager, revisit after PyTorch release notes mention fixes
```

## 7. Reference Material

- `common/torch_compile_safe.py` – Safe wrapper implementation
- `docs/ROOT_CAUSE_ANALYSIS.md` – Deep dive on the 40B+ hang
- `docs/TODO.md` – Track upstream bug filing status
- `docs/READY_TO_RUN_GUIDE.md` – Deployment flag matrix

Keep this guide open whenever enabling `torch.compile` so you can fall back to
the proven eager path quickly if regressions appear.
