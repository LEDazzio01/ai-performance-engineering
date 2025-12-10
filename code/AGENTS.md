# Coding Styles

## Chapter Consistency
- Make sure all code in the chapter (chXX/ examples are consistent with the content in the equivalent chapter (XX) of the AI Systems Performance Engineering book (book/chXX.md)

## FAIL FAST - NO FALLBACKS, NO AUTO-INFERENCE

**CRITICAL**: This project follows a STRICT fail-fast policy. DO NOT implement fallbacks, auto-detection, or auto-inference.

### What This Means

1. **NO Auto-Inference**: Never write code that guesses or infers values from attributes
   - BAD: `if hasattr(self, 'batch_size'): return self.batch_size`
   - GOOD: Require explicit implementation, raise `NotImplementedError` if missing

2. **NO Fallbacks**: Never provide default values when explicit implementation is required
   - BAD: `return sig if sig else None` or `return sig if sig else {}`
   - GOOD: `raise NotImplementedError("Benchmark must implement this method")`

3. **NO Silent Failures**: Never swallow errors or return empty/None when something is wrong
   - BAD: `try: ... except: return None`
   - GOOD: Let exceptions propagate with clear error messages

### Benchmark Verification Interface

Every benchmark MUST explicitly implement these methods (NO auto-detection):

```python
def get_verify_output(self) -> torch.Tensor:
    """MANDATORY: Return output tensor for verification."""
    raise NotImplementedError("Must implement explicitly")

def get_input_signature(self) -> dict:
    """MANDATORY: Return workload parameters for matching."""
    raise NotImplementedError("Must implement explicitly")
    
def get_output_tolerance(self) -> tuple:
    """MANDATORY: Return (rtol, atol) for numerical comparison."""
    raise NotImplementedError("Must implement explicitly")
```

### When You Find Code With Fallbacks

If you encounter code with auto-inference or fallbacks:
1. **DO NOT** add more fallbacks to fix the symptom
2. **DO** fix the underlying benchmarks to implement required methods
3. **DO** remove the fallback logic and make it fail-fast

### Audit Compliance

Use `aisp bench audit --all` to check verification compliance:
- All benchmark files must have 100% compliance
- Compliance means explicit implementations, not auto-detected ones

