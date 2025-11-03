# Long-Context Playbook (32K–64K Tokens)

Use this document to track configuration changes required for ultra-long
sequence validation. Populate the tables after running the checklist in
`docs/llm_validation_checklist.md`.

## Data Capture Template

| Date | Sequence Length | Batch Size | Peak Memory (GB) | Max Tokens/Sec | Notes |
|------|-----------------|------------|------------------|----------------|-------|
| YYYY-MM-DD | 32768 | 2 | TBD | TBD | Pending run |
| YYYY-MM-DD | 65536 | 1 | TBD | TBD | Pending run |

## Recommended Mitigations
- Enable activation checkpointing on decoder blocks (`--enable-ckpt`).
- Reduce KV cache precision to FP8 when memory exceeds 110 GB per node.
- Use `--max-new-tokens 256` for smoke tests before moving to 512+ tokens.
- Monitor `torch.cuda.max_memory_reserved()` per rank and log to JSON.

Update this playbook after each sweep so the next operator can pick up the
latest headroom numbers without re-running the entire experiment.
