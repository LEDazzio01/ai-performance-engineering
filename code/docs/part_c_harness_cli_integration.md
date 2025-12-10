# Part C: Harness & CLI Integration

## Summary

This document describes the integration of the benchmark verification system into the harness and CLI, ensuring that verification is **mandatory** and **automatic** for all benchmark runs.

## Design Principles

1. **Verification First**: Always verify before measuring performance
2. **Fail Fast**: Block performance measurement if verification fails
3. **Transparent**: Clear reporting of verification status
4. **Configurable Phases**: Support DETECT → QUARANTINE → GATE rollout

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        aisp bench run                           │
├─────────────────────────────────────────────────────────────────┤
│  1. Load benchmark pair (baseline + optimized)                  │
│  2. Check contract compliance                                    │
│  3. Run VERIFY MODE                                             │
│     ├─ Input signature matching                                 │
│     ├─ Output comparison (baseline → golden)                    │
│     ├─ Anti-cheat checks (jitter, fresh-input, workload)       │
│     └─ Quarantine if failed                                     │
│  4. If verification passed → Run PERF MODE                      │
│  5. Generate report with verify + perf results                  │
└─────────────────────────────────────────────────────────────────┘
```

## Enforcement Phases

| Phase | Behavior | CI Impact |
|-------|----------|-----------|
| **DETECT** | Report issues, don't fail | Warning only |
| **QUARANTINE** | Exclude non-compliant from perf reports | No perf data |
| **GATE** | Fail CI on any non-compliance | Build fails |

Set phase via environment variable:
```bash
export VERIFY_ENFORCEMENT_PHASE=gate  # or detect, quarantine
```

## CLI Integration

### New CLI Options

```bash
# Run with verification (default)
aisp bench run ch11

# Skip verification (NOT RECOMMENDED)
aisp bench run ch11 --skip-verify

# Skip only input verification
aisp bench run ch11 --skip-input-verify

# Skip only output verification
aisp bench run ch11 --skip-output-verify

# Set enforcement phase
aisp bench run ch11 --verify-phase gate

# Run verification only (no performance measurement)
aisp bench verify ch11

# Audit compliance
aisp bench audit
```

### Implementation

**File**: `core/benchmark/bench_commands.py`

```python
@click.command()
@click.argument('targets', nargs=-1)
@click.option('--skip-verify', is_flag=True, help='Skip verification (NOT RECOMMENDED)')
@click.option('--skip-input-verify', is_flag=True, help='Skip input signature verification')
@click.option('--skip-output-verify', is_flag=True, help='Skip output comparison')
@click.option('--verify-phase', type=click.Choice(['detect', 'quarantine', 'gate']),
              default=None, help='Override enforcement phase')
@click.option('--profile', type=click.Choice(['none', 'minimal', 'deep_dive', 'roofline']),
              default='minimal', help='Profiling level')
def run(targets, skip_verify, skip_input_verify, skip_output_verify, verify_phase, profile):
    """Run benchmarks with verification."""
    
    # Determine enforcement phase
    phase = verify_phase or os.environ.get('VERIFY_ENFORCEMENT_PHASE', 'detect')
    
    # Run benchmarks
    for target in targets:
        result = _execute_benchmark(
            target,
            verify_input=not skip_input_verify and not skip_verify,
            verify_output=not skip_output_verify and not skip_verify,
            enforcement_phase=phase,
            profile=profile
        )
        
        # Handle phase-specific behavior
        if not result.verification_passed:
            if phase == 'gate':
                sys.exit(1)  # Fail CI
            elif phase == 'quarantine':
                result.exclude_from_perf_report = True
            # detect: just report
```

## Harness Integration

### BenchmarkHarness Changes

**File**: `core/harness/benchmark_harness.py`

```python
class BenchmarkHarness:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.verify_runner = VerifyRunner(
            cache_dir=Path("artifacts/verify_cache"),
            phase=get_enforcement_phase()
        )
        self.quarantine_manager = QuarantineManager()
    
    def run_benchmark_pair(
        self,
        baseline: BaseBenchmark,
        optimized: BaseBenchmark,
        verify: bool = True
    ) -> BenchmarkResult:
        """Run a benchmark pair with verification."""
        
        result = BenchmarkResult()
        
        # 1. Check contract compliance
        baseline_compliance = BenchmarkContract.check_compliance(baseline)
        optimized_compliance = BenchmarkContract.check_compliance(optimized)
        
        if not baseline_compliance.compliant or not optimized_compliance.compliant:
            self._handle_non_compliance(baseline, optimized, result)
            return result
        
        # 2. Run verification
        if verify:
            verify_result = self.verify_runner.verify_pair(baseline, optimized)
            result.verification = verify_result
            
            if not verify_result.passed:
                self._handle_verification_failure(baseline, optimized, verify_result, result)
                return result
        
        # 3. Run performance measurement (only if verification passed)
        perf_result = self._run_performance(baseline, optimized)
        result.performance = perf_result
        
        return result
    
    def _handle_verification_failure(self, baseline, optimized, verify_result, result):
        """Handle verification failure based on enforcement phase."""
        phase = get_enforcement_phase()
        
        if phase == EnforcementPhase.GATE:
            raise VerificationError(f"Verification failed: {verify_result.reason}")
        
        elif phase == EnforcementPhase.QUARANTINE:
            self.quarantine_manager.quarantine(
                benchmark_path=baseline.__module__,
                reason=verify_result.reason,
                details=verify_result.details
            )
            result.quarantined = True
            result.perf_blocked = True
        
        else:  # DETECT
            result.verification_warning = verify_result.reason
            # Still run perf in detect mode
```

### New Harness Methods

```python
class BenchmarkHarness:
    def verify(self, baseline: BaseBenchmark, optimized: BaseBenchmark) -> VerifyResult:
        """Run verification suite without performance measurement."""
        return self.verify_runner.verify_pair(baseline, optimized)
    
    def gate_perf(self, benchmark_path: str) -> bool:
        """Check if performance measurement is allowed."""
        if self.quarantine_manager.is_quarantined(benchmark_path):
            return False
        
        phase = get_enforcement_phase()
        if phase == EnforcementPhase.GATE:
            # In GATE mode, must have passed verification
            return self._has_passed_verification(benchmark_path)
        
        return True
    
    def get_verification_status(self, benchmark_path: str) -> Dict:
        """Get current verification status for a benchmark."""
        return {
            "quarantined": self.quarantine_manager.is_quarantined(benchmark_path),
            "quarantine_reason": self.quarantine_manager.get_reason(benchmark_path),
            "last_verify_result": self._get_cached_verify_result(benchmark_path),
            "enforcement_phase": get_enforcement_phase().value,
        }
```

## Run Flow Integration

**File**: `core/harness/run_benchmarks.py`

```python
def _test_chapter_impl(
    chapter: str,
    examples: List[str],
    verify_input: bool = True,
    verify_output: bool = True,
    **kwargs
) -> ChapterResult:
    """Run benchmarks for a chapter with verification."""
    
    harness = BenchmarkHarness(BenchmarkConfig(**kwargs))
    results = []
    
    for example in examples:
        baseline = load_benchmark(chapter, f"baseline_{example}")
        optimized = load_benchmark(chapter, f"optimized_{example}")
        
        # Run full verification suite
        if verify_input or verify_output:
            verify_result = _run_full_verification_suite(
                harness, baseline, optimized,
                verify_input=verify_input,
                verify_output=verify_output
            )
            
            if not verify_result.passed:
                phase = get_enforcement_phase()
                
                if phase == EnforcementPhase.GATE:
                    raise VerificationError(
                        f"GATE mode: {example} failed verification: {verify_result.reason}"
                    )
                
                elif phase == EnforcementPhase.QUARANTINE:
                    logger.warning(f"QUARANTINE: {example} excluded from perf")
                    continue  # Skip perf measurement
        
        # Run performance measurement
        perf_result = harness.run_benchmark_pair(baseline, optimized, verify=False)
        results.append(perf_result)
    
    return ChapterResult(results)


def _run_full_verification_suite(
    harness: BenchmarkHarness,
    baseline: BaseBenchmark,
    optimized: BaseBenchmark,
    verify_input: bool = True,
    verify_output: bool = True
) -> VerifyResult:
    """Run the complete verification suite."""
    
    verify_runner = harness.verify_runner
    
    # 1. Contract compliance check
    for bench, name in [(baseline, "baseline"), (optimized, "optimized")]:
        compliance = BenchmarkContract.check_verification_compliance(bench)
        if not compliance.compliant:
            return VerifyResult.fail(
                f"{name}_non_compliant",
                compliance.missing_methods
            )
    
    # 2. Input signature verification
    if verify_input:
        baseline_sig = verify_runner._extract_signature(baseline)
        optimized_sig = verify_runner._extract_signature(optimized)
        
        if not baseline_sig.is_valid():
            return VerifyResult.fail("baseline_invalid_signature", baseline_sig.validation_errors)
        
        if baseline_sig != optimized_sig:
            return VerifyResult.fail("signature_mismatch", {
                "baseline": baseline_sig,
                "optimized": optimized_sig
            })
    
    # 3. Output verification
    if verify_output:
        # Run baseline in verify mode, cache golden output
        baseline_result = verify_runner.verify_baseline(baseline)
        if not baseline_result.passed:
            return baseline_result
        
        # Run optimized in verify mode, compare to golden
        pair_result = verify_runner.verify_pair(baseline, optimized)
        if not pair_result.passed:
            return pair_result
    
    return VerifyResult.success()
```

## Manifest Integration

**File**: `core/benchmark/run_manifest.py`

```python
@dataclass
class VerifyManifestEntry:
    """Verification results in run manifest."""
    verify_status: Literal["passed", "failed", "skipped", "quarantined"]
    baseline_checksum: Optional[str]
    optimized_checksum: Optional[str]
    comparison_result: Optional[ComparisonDetails]
    timestamp: datetime
    signature_hash: str
    workload_metrics: Optional[Dict]
    workload_delta: Optional[Dict[str, float]]
    quarantine_reason: Optional[str]
    tolerance_used: Optional[Dict]
    jitter_check_passed: Optional[bool]
    fresh_input_check_passed: Optional[bool]
    seed_info: Dict[str, int]
    enforcement_phase: str


@dataclass
class RunManifest:
    """Extended manifest with verification."""
    # ... existing fields ...
    
    # NEW: Verification results
    verification: Optional[VerifyManifestEntry] = None
    
    def to_json(self) -> Dict:
        """Export manifest to JSON."""
        data = asdict(self)
        if self.verification:
            data['verification'] = asdict(self.verification)
        return data
```

## Reporting Integration

### Verification Status in Reports

```python
def generate_benchmark_report(results: List[BenchmarkResult]) -> Report:
    """Generate report with verification status."""
    
    report = Report()
    
    # Summary section
    report.add_section("Verification Summary", {
        "total_benchmarks": len(results),
        "verified_passed": sum(1 for r in results if r.verification and r.verification.passed),
        "verified_failed": sum(1 for r in results if r.verification and not r.verification.passed),
        "quarantined": sum(1 for r in results if r.quarantined),
        "enforcement_phase": get_enforcement_phase().value,
    })
    
    # Quarantined benchmarks
    quarantined = [r for r in results if r.quarantined]
    if quarantined:
        report.add_section("Quarantined Benchmarks", [
            {
                "benchmark": r.name,
                "reason": r.quarantine_reason,
                "details": r.quarantine_details,
            }
            for r in quarantined
        ])
    
    # Performance results (only verified benchmarks)
    verified_results = [r for r in results if r.verification and r.verification.passed]
    report.add_section("Performance Results", verified_results)
    
    return report
```

## Implementation Status

| Component | Status | Files |
|-----------|--------|-------|
| CLI options | ✅ Implemented | `bench_commands.py` |
| Harness verify() | ✅ Implemented | `benchmark_harness.py` |
| Harness gate_perf() | ✅ Implemented | `benchmark_harness.py` |
| run_benchmarks integration | ✅ Implemented | `run_benchmarks.py` |
| Manifest fields | ✅ Implemented | `run_manifest.py` |
| Report integration | 🔲 Planned | TBD |
| CI job configuration | 🔲 Planned | `.github/workflows/` |

## Usage Examples

### Run with Full Verification (Default)
```bash
aisp bench run ch11
```

### Check Compliance
```bash
aisp bench audit ch11
```

### Verify Only (No Perf)
```bash
aisp bench verify ch11
```

### CI Pipeline
```yaml
jobs:
  verify:
    runs-on: gpu-runner
    env:
      VERIFY_ENFORCEMENT_PHASE: gate
    steps:
      - run: aisp bench run --verify-phase gate ch01 ch02 ch03
```

## Testing

Tests are located in:
- `tests/test_verification.py` - Unit tests for VerifyRunner
- `tests/test_verification_e2e.py` - End-to-end tests with real benchmarks
- `tests/test_harness_integration.py` - Harness integration tests (planned)

## Migration Path

1. **Week 1**: Deploy in DETECT mode, monitor warnings
2. **Week 2**: Fix high-priority issues, move to QUARANTINE
3. **Week 3**: Fix remaining issues, move to GATE
4. **Week 4**: Full enforcement, CI fails on any non-compliance


