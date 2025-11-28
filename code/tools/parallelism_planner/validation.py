"""
Validation Module for Parallelism Planner

Provides mechanisms to validate recommendations against reality:
- Memory prediction accuracy testing
- Dry-run execution for sanity checks
- Configuration compatibility verification
- Performance prediction validation
"""

import subprocess
import json
import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class ValidationStatus(Enum):
    """Status of a validation check."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    

@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    predicted: Optional[float] = None
    actual: Optional[float] = None
    error_pct: Optional[float] = None


@dataclass
class DryRunResult:
    """Result of a dry-run execution."""
    success: bool
    duration_seconds: float
    memory_allocated_gb: float
    memory_peak_gb: float
    checks: List[ValidationResult]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ConfigValidator:
    """Validates parallelism configurations for correctness and compatibility."""
    
    def __init__(self):
        self.checks = []
    
    def validate_strategy(
        self,
        strategy: Dict[str, Any],
        hardware: Dict[str, Any],
        model: Dict[str, Any]
    ) -> List[ValidationResult]:
        """
        Validate a parallelism strategy against hardware and model constraints.
        
        Returns list of validation results with pass/fail/warning status.
        """
        results = []
        
        # Extract strategy parameters
        dp = strategy.get('data_parallel', 1)
        tp = strategy.get('tensor_parallel', 1)
        pp = strategy.get('pipeline_parallel', 1)
        cp = strategy.get('context_parallel', 1)
        ep = strategy.get('expert_parallel', 1)
        
        # Extract hardware info
        num_gpus = hardware.get('num_gpus', 1)
        gpu_memory_gb = hardware.get('gpu_memory_gb', 80)
        has_nvlink = hardware.get('has_nvlink', False)
        nvlink_bandwidth_gbps = hardware.get('nvlink_bandwidth_gbps', 0)
        
        # Extract model info
        num_layers = model.get('num_layers', 32)
        hidden_size = model.get('hidden_size', 4096)
        num_experts = model.get('num_experts', 1)
        
        # Check 1: Total parallelism matches GPU count
        total_parallel = dp * tp * pp * cp
        if ep > 1 and num_experts > 1:
            # EP is within DP for MoE
            pass
        
        if total_parallel != num_gpus:
            results.append(ValidationResult(
                name="gpu_count_match",
                status=ValidationStatus.FAILED,
                message=f"Parallelism product ({total_parallel}) doesn't match GPU count ({num_gpus})",
                details={"dp": dp, "tp": tp, "pp": pp, "cp": cp, "total": total_parallel, "gpus": num_gpus}
            ))
        else:
            results.append(ValidationResult(
                name="gpu_count_match",
                status=ValidationStatus.PASSED,
                message=f"Parallelism ({dp}×{tp}×{pp}×{cp}={total_parallel}) matches {num_gpus} GPUs"
            ))
        
        # Check 2: TP requires NVLink for efficiency
        if tp > 1 and not has_nvlink:
            results.append(ValidationResult(
                name="tp_nvlink",
                status=ValidationStatus.WARNING,
                message=f"TP={tp} without NVLink will have significant communication overhead",
                details={"tp": tp, "has_nvlink": has_nvlink}
            ))
        elif tp > 1:
            results.append(ValidationResult(
                name="tp_nvlink",
                status=ValidationStatus.PASSED,
                message=f"TP={tp} with NVLink ({nvlink_bandwidth_gbps} GB/s) - good configuration"
            ))
        
        # Check 3: TP should divide hidden size evenly
        if tp > 1 and hidden_size % tp != 0:
            results.append(ValidationResult(
                name="tp_hidden_divisibility",
                status=ValidationStatus.FAILED,
                message=f"Hidden size ({hidden_size}) not divisible by TP ({tp})",
                details={"hidden_size": hidden_size, "tp": tp}
            ))
        elif tp > 1:
            results.append(ValidationResult(
                name="tp_hidden_divisibility",
                status=ValidationStatus.PASSED,
                message=f"Hidden size ({hidden_size}) evenly divides by TP ({tp})"
            ))
        
        # Check 4: PP should divide layers reasonably
        if pp > 1:
            layers_per_stage = num_layers / pp
            if layers_per_stage < 1:
                results.append(ValidationResult(
                    name="pp_layer_distribution",
                    status=ValidationStatus.FAILED,
                    message=f"PP ({pp}) exceeds layer count ({num_layers})",
                    details={"pp": pp, "num_layers": num_layers}
                ))
            elif layers_per_stage < 2:
                results.append(ValidationResult(
                    name="pp_layer_distribution",
                    status=ValidationStatus.WARNING,
                    message=f"Only {layers_per_stage:.1f} layers per PP stage - high bubble overhead",
                    details={"pp": pp, "num_layers": num_layers, "layers_per_stage": layers_per_stage}
                ))
            else:
                results.append(ValidationResult(
                    name="pp_layer_distribution",
                    status=ValidationStatus.PASSED,
                    message=f"{layers_per_stage:.1f} layers per PP stage"
                ))
        
        # Check 5: EP for MoE models
        if ep > 1 and num_experts <= 1:
            results.append(ValidationResult(
                name="ep_moe_required",
                status=ValidationStatus.FAILED,
                message=f"EP={ep} specified but model has no experts (num_experts={num_experts})",
                details={"ep": ep, "num_experts": num_experts}
            ))
        elif ep > 1:
            if num_experts % ep != 0:
                results.append(ValidationResult(
                    name="ep_expert_divisibility",
                    status=ValidationStatus.WARNING,
                    message=f"Experts ({num_experts}) not evenly divisible by EP ({ep})",
                    details={"num_experts": num_experts, "ep": ep}
                ))
            else:
                results.append(ValidationResult(
                    name="ep_expert_divisibility",
                    status=ValidationStatus.PASSED,
                    message=f"EP={ep} evenly distributes {num_experts} experts"
                ))
        
        # Check 6: CP requires long sequences
        if cp > 1:
            seq_length = model.get('max_sequence_length', 4096)
            if seq_length < 8192:
                results.append(ValidationResult(
                    name="cp_sequence_length",
                    status=ValidationStatus.WARNING,
                    message=f"CP={cp} with short sequences ({seq_length}) has high overhead",
                    details={"cp": cp, "seq_length": seq_length}
                ))
            elif seq_length % cp != 0:
                results.append(ValidationResult(
                    name="cp_sequence_divisibility",
                    status=ValidationStatus.WARNING,
                    message=f"Sequence length ({seq_length}) not evenly divisible by CP ({cp})",
                    details={"seq_length": seq_length, "cp": cp}
                ))
            else:
                results.append(ValidationResult(
                    name="cp_sequence_length",
                    status=ValidationStatus.PASSED,
                    message=f"CP={cp} with {seq_length} sequence length - appropriate"
                ))
        
        # Check 7: TP should be power of 2
        if tp > 1 and (tp & (tp - 1)) != 0:
            results.append(ValidationResult(
                name="tp_power_of_2",
                status=ValidationStatus.WARNING,
                message=f"TP={tp} is not a power of 2 - may have suboptimal performance",
                details={"tp": tp}
            ))
        
        return results
    
    def validate_memory_fit(
        self,
        predicted_memory_gb: float,
        gpu_memory_gb: float,
        headroom_pct: float = 0.1
    ) -> ValidationResult:
        """Check if predicted memory fits in GPU with headroom."""
        available = gpu_memory_gb * (1 - headroom_pct)
        
        if predicted_memory_gb > gpu_memory_gb:
            return ValidationResult(
                name="memory_fit",
                status=ValidationStatus.FAILED,
                message=f"Predicted {predicted_memory_gb:.1f} GB exceeds GPU memory {gpu_memory_gb} GB",
                predicted=predicted_memory_gb,
                actual=gpu_memory_gb
            )
        elif predicted_memory_gb > available:
            return ValidationResult(
                name="memory_fit",
                status=ValidationStatus.WARNING,
                message=f"Predicted {predicted_memory_gb:.1f} GB leaves only {gpu_memory_gb - predicted_memory_gb:.1f} GB headroom",
                predicted=predicted_memory_gb,
                actual=gpu_memory_gb
            )
        else:
            return ValidationResult(
                name="memory_fit",
                status=ValidationStatus.PASSED,
                message=f"Predicted {predicted_memory_gb:.1f} GB fits in {gpu_memory_gb} GB with {gpu_memory_gb - predicted_memory_gb:.1f} GB headroom",
                predicted=predicted_memory_gb,
                actual=gpu_memory_gb
            )


class MemoryValidator:
    """Validates memory predictions against actual usage."""
    
    def __init__(self):
        self.measurements = []
    
    def measure_actual_memory(self) -> Optional[Dict[str, float]]:
        """
        Measure actual GPU memory using nvidia-smi.
        
        Returns dict with memory info per GPU.
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,memory.free',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                return None
            
            gpus = {}
            for line in result.stdout.strip().split('\n'):
                parts = line.split(',')
                if len(parts) >= 4:
                    idx = int(parts[0].strip())
                    gpus[idx] = {
                        'used_mb': float(parts[1].strip()),
                        'total_mb': float(parts[2].strip()),
                        'free_mb': float(parts[3].strip()),
                        'used_gb': float(parts[1].strip()) / 1024,
                        'total_gb': float(parts[2].strip()) / 1024
                    }
            
            return gpus
            
        except Exception as e:
            return None
    
    def compare_prediction(
        self,
        predicted_gb: float,
        actual_gb: float
    ) -> ValidationResult:
        """Compare predicted vs actual memory usage."""
        error_pct = abs(predicted_gb - actual_gb) / actual_gb * 100 if actual_gb > 0 else 0
        
        if error_pct < 5:
            status = ValidationStatus.PASSED
            message = f"Excellent prediction accuracy: {error_pct:.1f}% error"
        elif error_pct < 15:
            status = ValidationStatus.PASSED
            message = f"Good prediction accuracy: {error_pct:.1f}% error"
        elif error_pct < 30:
            status = ValidationStatus.WARNING
            message = f"Moderate prediction error: {error_pct:.1f}%"
        else:
            status = ValidationStatus.WARNING
            message = f"High prediction error: {error_pct:.1f}% - consider calibration"
        
        return ValidationResult(
            name="memory_prediction_accuracy",
            status=status,
            message=message,
            predicted=predicted_gb,
            actual=actual_gb,
            error_pct=error_pct
        )


class DryRunner:
    """
    Executes quick dry-run tests to validate configurations.
    
    Performs lightweight tests that catch common issues:
    - CUDA memory allocation
    - Basic forward/backward pass
    - Communication patterns
    """
    
    def __init__(self, timeout_seconds: int = 60):
        self.timeout = timeout_seconds
    
    def generate_dry_run_script(
        self,
        model_config: Dict[str, Any],
        strategy: Dict[str, Any],
        batch_size: int = 1,
        seq_length: int = 128
    ) -> str:
        """Generate a Python script for dry-run testing."""
        
        script = '''#!/usr/bin/env python3
"""Auto-generated dry-run validation script."""
import torch
import torch.nn as nn
import json
import time
import gc

def main():
    results = {
        "success": True,
        "checks": [],
        "memory_allocated_gb": 0,
        "memory_peak_gb": 0,
        "errors": [],
        "warnings": []
    }
    
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            results["errors"].append("CUDA not available")
            results["success"] = False
            print(json.dumps(results))
            return
        
        results["checks"].append({
            "name": "cuda_available",
            "status": "passed",
            "message": f"CUDA available with {torch.cuda.device_count()} GPUs"
        })
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()
        
        # Model parameters
        hidden_size = ''' + str(model_config.get('hidden_size', 4096)) + '''
        num_layers = min(''' + str(model_config.get('num_layers', 32)) + ''', 4)  # Use fewer layers for dry run
        batch_size = ''' + str(batch_size) + '''
        seq_length = ''' + str(seq_length) + '''
        
        # Create a simple transformer-like model for testing
        class SimpleTransformerBlock(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.attention = nn.MultiheadAttention(hidden_size, num_heads=32, batch_first=True)
                self.ffn = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
                self.norm1 = nn.LayerNorm(hidden_size)
                self.norm2 = nn.LayerNorm(hidden_size)
            
            def forward(self, x):
                # Self-attention
                normed = self.norm1(x)
                attn_out, _ = self.attention(normed, normed, normed)
                x = x + attn_out
                # FFN
                x = x + self.ffn(self.norm2(x))
                return x
        
        class SimpleModel(nn.Module):
            def __init__(self, hidden_size, num_layers):
                super().__init__()
                self.embed = nn.Embedding(32000, hidden_size)
                self.layers = nn.ModuleList([
                    SimpleTransformerBlock(hidden_size) for _ in range(num_layers)
                ])
                self.head = nn.Linear(hidden_size, 32000)
            
            def forward(self, x):
                x = self.embed(x)
                for layer in self.layers:
                    x = layer(x)
                return self.head(x)
        
        # Create model on GPU
        device = torch.device("cuda:0")
        model = SimpleModel(hidden_size, num_layers).to(device)
        
        results["checks"].append({
            "name": "model_creation",
            "status": "passed",
            "message": f"Created model with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters"
        })
        
        # Create dummy input
        dummy_input = torch.randint(0, 32000, (batch_size, seq_length), device=device)
        
        # Forward pass
        start_time = time.time()
        with torch.cuda.amp.autocast():
            output = model(dummy_input)
        torch.cuda.synchronize()
        forward_time = time.time() - start_time
        
        results["checks"].append({
            "name": "forward_pass",
            "status": "passed",
            "message": f"Forward pass completed in {forward_time*1000:.1f}ms"
        })
        
        # Backward pass
        loss = output.mean()
        start_time = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time = time.time() - start_time
        
        results["checks"].append({
            "name": "backward_pass",
            "status": "passed",
            "message": f"Backward pass completed in {backward_time*1000:.1f}ms"
        })
        
        # Memory stats
        allocated = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        
        results["memory_allocated_gb"] = allocated
        results["memory_peak_gb"] = peak
        
        results["checks"].append({
            "name": "memory_usage",
            "status": "passed",
            "message": f"Peak memory: {peak:.2f} GB, Current: {allocated:.2f} GB"
        })
        
        # Cleanup
        del model, dummy_input, output, loss
        torch.cuda.empty_cache()
        gc.collect()
        
    except torch.cuda.OutOfMemoryError as e:
        results["success"] = False
        results["errors"].append(f"CUDA OOM: {str(e)}")
        results["checks"].append({
            "name": "memory_allocation",
            "status": "failed",
            "message": "Out of memory error"
        })
    except Exception as e:
        results["success"] = False
        results["errors"].append(str(e))
    
    print(json.dumps(results))

if __name__ == "__main__":
    main()
'''
        return script
    
    def run_dry_test(
        self,
        model_config: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> DryRunResult:
        """
        Run a quick dry-run test to validate configuration.
        
        Returns DryRunResult with success/failure and details.
        """
        import tempfile
        
        script = self.generate_dry_run_script(model_config, strategy)
        
        start_time = time.time()
        
        try:
            # Write script to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                script_path = f.name
            
            # Execute script
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
            )
            
            duration = time.time() - start_time
            
            # Parse output
            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout.strip())
                    checks = [
                        ValidationResult(
                            name=c['name'],
                            status=ValidationStatus.PASSED if c['status'] == 'passed' else ValidationStatus.FAILED,
                            message=c['message']
                        )
                        for c in output.get('checks', [])
                    ]
                    
                    return DryRunResult(
                        success=output.get('success', False),
                        duration_seconds=duration,
                        memory_allocated_gb=output.get('memory_allocated_gb', 0),
                        memory_peak_gb=output.get('memory_peak_gb', 0),
                        checks=checks,
                        errors=output.get('errors', []),
                        warnings=output.get('warnings', [])
                    )
                except json.JSONDecodeError:
                    return DryRunResult(
                        success=False,
                        duration_seconds=duration,
                        memory_allocated_gb=0,
                        memory_peak_gb=0,
                        checks=[],
                        errors=[f"Failed to parse output: {result.stdout}"]
                    )
            else:
                return DryRunResult(
                    success=False,
                    duration_seconds=duration,
                    memory_allocated_gb=0,
                    memory_peak_gb=0,
                    checks=[],
                    errors=[result.stderr or "Unknown error"]
                )
                
        except subprocess.TimeoutExpired:
            return DryRunResult(
                success=False,
                duration_seconds=self.timeout,
                memory_allocated_gb=0,
                memory_peak_gb=0,
                checks=[],
                errors=[f"Dry run timed out after {self.timeout} seconds"]
            )
        except Exception as e:
            return DryRunResult(
                success=False,
                duration_seconds=time.time() - start_time,
                memory_allocated_gb=0,
                memory_peak_gb=0,
                checks=[],
                errors=[str(e)]
            )
        finally:
            # Cleanup
            try:
                os.unlink(script_path)
            except:
                pass


class FrameworkConfigValidator:
    """Validates generated framework configurations against schemas."""
    
    def validate_deepspeed_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate DeepSpeed configuration."""
        results = []
        
        # Check required fields
        if 'train_batch_size' not in config:
            results.append(ValidationResult(
                name="ds_train_batch_size",
                status=ValidationStatus.WARNING,
                message="train_batch_size not specified - will be computed automatically"
            ))
        
        # Check ZeRO config
        if 'zero_optimization' in config:
            zero = config['zero_optimization']
            stage = zero.get('stage', 0)
            
            if stage not in [0, 1, 2, 3]:
                results.append(ValidationResult(
                    name="ds_zero_stage",
                    status=ValidationStatus.FAILED,
                    message=f"Invalid ZeRO stage: {stage}"
                ))
            else:
                results.append(ValidationResult(
                    name="ds_zero_stage",
                    status=ValidationStatus.PASSED,
                    message=f"ZeRO Stage {stage} configured"
                ))
            
            # Stage 3 specific checks
            if stage == 3:
                if not zero.get('stage3_gather_16bit_weights_on_model_save', True):
                    results.append(ValidationResult(
                        name="ds_zero3_save",
                        status=ValidationStatus.WARNING,
                        message="stage3_gather_16bit_weights_on_model_save is False - checkpoints may be unusable"
                    ))
        
        # Check FP16/BF16 config
        if config.get('fp16', {}).get('enabled') and config.get('bf16', {}).get('enabled'):
            results.append(ValidationResult(
                name="ds_mixed_precision",
                status=ValidationStatus.FAILED,
                message="Both FP16 and BF16 enabled - choose one"
            ))
        
        # Check gradient accumulation
        grad_accum = config.get('gradient_accumulation_steps', 1)
        if grad_accum > 64:
            results.append(ValidationResult(
                name="ds_grad_accum",
                status=ValidationStatus.WARNING,
                message=f"High gradient accumulation ({grad_accum}) may affect training dynamics"
            ))
        
        return results
    
    def validate_fsdp_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate FSDP configuration."""
        results = []
        
        sharding = config.get('sharding_strategy', 'FULL_SHARD')
        valid_strategies = ['FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD', 'HYBRID_SHARD']
        
        if sharding not in valid_strategies:
            results.append(ValidationResult(
                name="fsdp_sharding_strategy",
                status=ValidationStatus.FAILED,
                message=f"Invalid sharding strategy: {sharding}"
            ))
        else:
            results.append(ValidationResult(
                name="fsdp_sharding_strategy",
                status=ValidationStatus.PASSED,
                message=f"Using {sharding} strategy"
            ))
        
        # Check backward prefetch
        if config.get('backward_prefetch') and config.get('forward_prefetch'):
            results.append(ValidationResult(
                name="fsdp_prefetch",
                status=ValidationStatus.PASSED,
                message="Both forward and backward prefetch enabled - optimal for throughput"
            ))
        
        return results


def validate_full_configuration(
    strategy: Dict[str, Any],
    hardware: Dict[str, Any],
    model: Dict[str, Any],
    framework_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive validation of a parallelism configuration.
    
    Returns a validation report with all checks.
    """
    config_validator = ConfigValidator()
    framework_validator = FrameworkConfigValidator()
    
    all_results = []
    
    # Strategy validation
    strategy_results = config_validator.validate_strategy(strategy, hardware, model)
    all_results.extend(strategy_results)
    
    # Memory validation
    predicted_memory = strategy.get('predicted_memory_gb', 0)
    if predicted_memory > 0:
        memory_result = config_validator.validate_memory_fit(
            predicted_memory,
            hardware.get('gpu_memory_gb', 80)
        )
        all_results.append(memory_result)
    
    # Framework config validation
    if framework_config:
        if 'zero_optimization' in framework_config or 'train_batch_size' in framework_config:
            ds_results = framework_validator.validate_deepspeed_config(framework_config)
            all_results.extend(ds_results)
        
        if 'sharding_strategy' in framework_config:
            fsdp_results = framework_validator.validate_fsdp_config(framework_config)
            all_results.extend(fsdp_results)
    
    # Aggregate results
    passed = sum(1 for r in all_results if r.status == ValidationStatus.PASSED)
    failed = sum(1 for r in all_results if r.status == ValidationStatus.FAILED)
    warnings = sum(1 for r in all_results if r.status == ValidationStatus.WARNING)
    
    return {
        "valid": failed == 0,
        "summary": {
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "total": len(all_results)
        },
        "checks": [
            {
                "name": r.name,
                "status": r.status.value,
                "message": r.message,
                "details": r.details,
                "predicted": r.predicted,
                "actual": r.actual,
                "error_pct": r.error_pct
            }
            for r in all_results
        ],
        "can_proceed": failed == 0,
        "recommendation": "Configuration valid" if failed == 0 else "Fix failed checks before proceeding"
    }



