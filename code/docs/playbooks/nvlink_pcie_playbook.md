# NVLink vs PCIe Topology Playbook (LLM Inference)

Use this playbook when the 8×B200 cluster is suspected of running in PCIe mode
or when throughput falls below the validated NVLink baseline.

## 1. Detect the Active Topology

```bash
nvidia-smi topo -m
```

- **Expected**: Majority of GPU pairs report `NV18` or `NV4`, CPU entries show
  `PXB`. This indicates full NVLink mesh.
- **Problematic**: Entries show `PHB` or `SYS`, indicating PCIe fallback.

Record the result in `artifacts/topology/topo_$(date +%Y%m%d).txt` for audits.

## 2. Validate NCCL Configuration

Export the known-good environment before launching any tensor-parallel job:

```bash
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
```

To enforce the configuration in systemd or Kubernetes manifests, place the
variables in the unit spec (`Environment=` lines) or container env section.

## 3. Run the Sanity Benchmark

```bash
python tools/nsys_profile_workload.py \
  --workload ch16/inference_server_load_test.py \
  --duration 30 \
  --extra-args "--attention-backend flex --fp8-mode transformer-engine"
```

Use `nvidia-smi nvlink --status` during the run to confirm link counters are
increasing.

## 4. When NVLink Is Unavailable

1. Switch to PCIe-safe launch flags:
   ```bash
   export NCCL_P2P_DISABLE=1
   export NCCL_P2P_LEVEL=PCI
   python ch16/inference_server_load_test.py --skip-compile --max-new-tokens 256
   ```
2. Reduce tensor-parallel degree (e.g., 2-way instead of 8-way).
3. Increase batch aggregation windows to offset bandwidth loss.
4. Update SLO dashboards with the degraded throughput (typically 2–3× slower).

Document the incident in `docs/incident_response.md` and plan the return-to-NVLink
remediation.

## 5. Back to NVLink

After maintenance or hardware replacement, re-enable the standard config:

```bash
unset NCCL_P2P_LEVEL
unset NCCL_P2P_DISABLE
./verify_nvlink.py && ./verify_pytorch.py
```

Finish with a full multi-GPU benchmark:

```bash
torchrun --nproc_per_node=8 \
  ch16/inference_server_load_test.py \
  --duration 120 \
  --target-qps 28 \
  --attention-backend flex \
  --fp8-mode transformer-engine \
  --output-json artifacts/nvlink/full_mesh_validation.json
```

Archive the JSON artifact with the topology snapshot for traceability.
