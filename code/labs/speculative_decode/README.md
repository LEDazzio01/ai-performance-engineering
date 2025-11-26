# Speculative Decoding Lab

**Goal**: Accelerate autoregressive generation by drafting multiple tokens in parallel.

## Overview

Speculative decoding uses a small "draft" model to propose multiple tokens, which are then verified in parallel by the large "target" model. When acceptance rates are high, this provides significant speedups.

## Key Concepts

| Concept | Description |
|---------|-------------|
| Draft model | Small, fast model (e.g., 1B params) |
| Target model | Large, accurate model (e.g., 70B params) |
| Draft length | Tokens proposed per round (typically 4-8) |
| Acceptance rate | % of draft tokens accepted |
| Speedup | Proportional to `acceptance_rate × draft_length` |

## Theoretical Speedup

```
Speedup ≈ 1 / (1 - acceptance_rate + acceptance_rate/draft_length)
```

| Acceptance Rate | Draft Length | Speedup |
|-----------------|--------------|---------|
| 70% | 4 | 2.1× |
| 80% | 4 | 2.5× |
| 70% | 8 | 2.5× |
| 80% | 8 | 3.3× |

## Files

Currently a placeholder lab. Implementation coming soon.

Planned files:
- `baseline_autoregressive.py` - Standard token-by-token generation
- `optimized_speculative.py` - Draft-verify speculative decoding
- `draft_models.py` - Small draft model implementations

## Algorithm

```python
def speculative_decode(draft_model, target_model, prompt, max_tokens):
    tokens = prompt
    while len(tokens) < max_tokens:
        # 1. Draft: generate k tokens with small model
        draft_tokens = draft_model.generate(tokens, k=4)
        
        # 2. Verify: score all drafts in parallel with target model
        target_logits = target_model(tokens + draft_tokens)
        
        # 3. Accept/reject using rejection sampling
        accepted = 0
        for i, (draft_tok, target_logit) in enumerate(zip(draft_tokens, target_logits)):
            p_target = softmax(target_logit)[draft_tok]
            p_draft = draft_probs[i][draft_tok]
            
            if random() < min(1, p_target / p_draft):
                accepted += 1
            else:
                # Sample correction token from adjusted distribution
                tokens.append(sample_correction(target_logit, draft_probs[i]))
                break
        
        tokens.extend(draft_tokens[:accepted])
    
    return tokens
```

## When to Use

✅ **Good for**:
- Similar draft/target model families (high acceptance)
- Long generation tasks (amortizes overhead)
- Batch size = 1 (memory-bound scenarios)

❌ **Not ideal for**:
- Very different model architectures
- Short outputs (< 50 tokens)
- Large batch sizes (already compute-bound)

## Related Chapters

- **Ch18**: Speculative decoding deep dive
- **Ch18**: FlashMLA and advanced decode optimizations
- **Ch15**: MoE inference (can use speculative decoding)

## References

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2022)
- [SpecInfer: Accelerating LLM Serving](https://arxiv.org/abs/2305.09781) (Miao et al., 2023)

