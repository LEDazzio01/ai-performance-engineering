"""Kernel correctness verification utilities.

This module provides tools for verifying GPU kernel correctness:

1. **ManualKernelVerifier**: Traditional testing approach with random inputs,
   edge cases, and boundary conditions. Limited coverage.

2. **FormalKernelVerifier**: ProofWright-style verification with memory safety,
   thread safety, and semantic correctness proofs. Complete coverage.

Usage:
    from tools.verification.kernel_correctness import (
        ManualKernelVerifier,
        FormalKernelVerifier,
        VerificationResult,
    )
    
    # Manual testing
    verifier = ManualKernelVerifier(device="cuda")
    result = verifier.verify(kernel_fn, reference_fn, shape=(1024, 1024))
    
    # Formal verification
    formal = FormalKernelVerifier(device="cuda")
    proofs = formal.verify(kernel_source, specifications)

The key insight is that these tools demonstrate two fundamentally different
approaches to verification:

- Manual testing: "We ran N tests and found no bugs"
- Formal proofs: "We mathematically proved no bugs exist"
"""

from .manual_verifier import ManualKernelVerifier, get_dtype_tolerances
from .formal_verifier import FormalKernelVerifier, VerificationProof, VerificationStatus

__all__ = [
    "ManualKernelVerifier",
    "FormalKernelVerifier",
    "VerificationProof",
    "VerificationStatus",
    "get_dtype_tolerances",
]

