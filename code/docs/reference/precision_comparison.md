# Precision Accuracy Comparison

**Dataset**: `structured_tokens.txt`

## Results

| Precision | Perplexity | Avg Loss | Tokens Evaluated |
|-----------|------------|----------|------------------|
| FP16 | 11542.545 | 9.3538 | 9,856 |
| BF16 | 11742.161 | 9.3709 | 9,856 |

## Analysis

### FP16 vs BF16
- FP16 perplexity: 11542.545
- BF16 perplexity: 11742.161
- Delta: +1.73%

## Recommendations

- **FP16**: Standard baseline for inference
- **BF16**: Better for training, similar inference accuracy
- **FP8**: Use when accuracy delta is acceptable and performance gain is significant

Always validate accuracy on your specific workload before deploying FP8.
