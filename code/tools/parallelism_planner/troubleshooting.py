"""
Troubleshooting and Diagnostics Module for Parallelism Planner

Provides comprehensive troubleshooting guidance:
- Common distributed training errors and solutions
- NCCL error diagnosis
- Memory error analysis
- Hanging job detection
- Configuration validation with actionable fixes
- Hardware compatibility checks
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class ErrorCategory(Enum):
    """Categories of distributed training errors."""
    MEMORY = "memory"
    COMMUNICATION = "communication"
    CONFIGURATION = "configuration"
    HARDWARE = "hardware"
    FRAMEWORK = "framework"
    PERFORMANCE = "performance"


@dataclass
class TroubleshootingIssue:
    """A diagnosed issue with solution."""
    category: ErrorCategory
    severity: str  # "critical", "warning", "info"
    title: str
    description: str
    symptoms: List[str]
    root_causes: List[str]
    solutions: List[str]
    code_fix: Optional[str] = None
    env_vars: Optional[Dict[str, str]] = None
    references: List[str] = field(default_factory=list)


class DistributedTrainingTroubleshooter:
    """Diagnoses and provides solutions for distributed training issues."""
    
    # Common error patterns and solutions
    ERROR_DATABASE = {
        "cuda_oom": TroubleshootingIssue(
            category=ErrorCategory.MEMORY,
            severity="critical",
            title="CUDA Out of Memory",
            description="GPU memory exhausted during training",
            symptoms=[
                "RuntimeError: CUDA out of memory",
                "torch.cuda.OutOfMemoryError",
                "Tried to allocate X GiB",
            ],
            root_causes=[
                "Batch size too large",
                "Model too large for single GPU",
                "Activation memory accumulation",
                "Memory fragmentation",
                "Memory leak in training loop",
            ],
            solutions=[
                "1. Reduce micro-batch size",
                "2. Enable gradient checkpointing: model.gradient_checkpointing_enable()",
                "3. Use mixed precision (BF16/FP16)",
                "4. Increase tensor parallelism to shard model",
                "5. Enable ZeRO-3/FSDP for optimizer state sharding",
                "6. Use 8-bit optimizer: pip install bitsandbytes",
                "7. Clear cache periodically: torch.cuda.empty_cache()",
            ],
            code_fix="""
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
""",
            env_vars={
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
        ),
        
        "nccl_timeout": TroubleshootingIssue(
            category=ErrorCategory.COMMUNICATION,
            severity="critical",
            title="NCCL Timeout / Hanging",
            description="Distributed communication timeout or process hanging",
            symptoms=[
                "NCCL timeout",
                "Watchdog caught collective operation timeout",
                "Process hanging at barrier/all_reduce",
                "RuntimeError: NCCL communicator was aborted",
            ],
            root_causes=[
                "Network connectivity issues between nodes",
                "Mismatched tensor sizes across ranks",
                "Deadlock from incorrect collective ordering",
                "Firewall blocking NCCL ports",
                "InfiniBand/RoCE configuration issues",
            ],
            solutions=[
                "1. Increase timeout: set NCCL_TIMEOUT to higher value",
                "2. Check network connectivity: ping between nodes",
                "3. Verify all ranks have same tensor shapes",
                "4. Open required ports (29500 for torch, NCCL ports)",
                "5. Check InfiniBand: ibstat, ibv_devinfo",
                "6. Try TCP fallback: NCCL_IB_DISABLE=1",
                "7. Enable NCCL debug: NCCL_DEBUG=INFO",
            ],
            env_vars={
                "NCCL_TIMEOUT": "1800",
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "ALL",
                "NCCL_IB_TIMEOUT": "23",
                "NCCL_SOCKET_NTHREADS": "4",
            },
        ),
        
        "nccl_init_failed": TroubleshootingIssue(
            category=ErrorCategory.COMMUNICATION,
            severity="critical",
            title="NCCL Initialization Failed",
            description="Failed to initialize NCCL communication",
            symptoms=[
                "NCCL error: unhandled system error",
                "Failed to initialize NCCL",
                "NCCL WARN Connect to X failed",
            ],
            root_causes=[
                "NCCL version mismatch",
                "GPU driver issues",
                "Network interface not found",
                "Incorrect MASTER_ADDR/MASTER_PORT",
            ],
            solutions=[
                "1. Verify NCCL installation: python -c 'import torch; print(torch.cuda.nccl.version())'",
                "2. Check GPU driver: nvidia-smi",
                "3. Set correct network interface: NCCL_SOCKET_IFNAME=eth0",
                "4. Verify MASTER_ADDR is reachable from all nodes",
                "5. Try different MASTER_PORT (avoid conflicts)",
            ],
            env_vars={
                "NCCL_SOCKET_IFNAME": "eth0",
                "NCCL_IB_GID_INDEX": "3",
            },
        ),
        
        "tp_mismatch": TroubleshootingIssue(
            category=ErrorCategory.CONFIGURATION,
            severity="critical",
            title="Tensor Parallel Size Mismatch",
            description="Tensor parallel size incompatible with model",
            symptoms=[
                "Hidden size not divisible by TP",
                "Number of attention heads not divisible by TP",
                "RuntimeError: size mismatch",
            ],
            root_causes=[
                "TP size doesn't divide hidden_size evenly",
                "TP size doesn't divide num_attention_heads evenly",
                "TP size doesn't divide num_key_value_heads for GQA",
            ],
            solutions=[
                "1. Use TP size that divides hidden_size (e.g., 2, 4, 8)",
                "2. For Llama-70B (hidden=8192): use TP=1,2,4,8",
                "3. Check GQA: num_kv_heads must be divisible by TP",
                "4. For Mixtral: num_experts (8) should be divisible by EP",
            ],
        ),
        
        "pp_imbalance": TroubleshootingIssue(
            category=ErrorCategory.PERFORMANCE,
            severity="warning",
            title="Pipeline Parallel Imbalance",
            description="Uneven layer distribution causing load imbalance",
            symptoms=[
                "Some GPUs idle while others compute",
                "Pipeline bubble > 30%",
                "Low GPU utilization with PP",
            ],
            root_causes=[
                "Uneven layer distribution across stages",
                "First/last stage has embedding/head overhead",
                "Too few micro-batches for pipeline efficiency",
            ],
            solutions=[
                "1. Increase num_micro_batches >= 4 * pp_degree",
                "2. Use interleaved scheduling (virtual pipeline stages)",
                "3. Balance layers accounting for embedding layers",
                "4. Consider reducing PP degree if bubble is high",
            ],
        ),
        
        "low_gpu_util": TroubleshootingIssue(
            category=ErrorCategory.PERFORMANCE,
            severity="warning",
            title="Low GPU Utilization",
            description="GPUs not being fully utilized",
            symptoms=[
                "nvidia-smi shows < 80% GPU utilization",
                "Low MFU (Model FLOPS Utilization)",
                "Training slower than expected",
            ],
            root_causes=[
                "Batch size too small",
                "CPU data loading bottleneck",
                "Communication overhead",
                "Kernel launch overhead",
                "Memory bandwidth bound",
            ],
            solutions=[
                "1. Increase batch size (use gradient accumulation if OOM)",
                "2. Increase num_workers in DataLoader",
                "3. Use pin_memory=True in DataLoader",
                "4. Enable torch.compile() for kernel fusion",
                "5. Use Flash Attention for memory-bound attention",
                "6. Profile with torch.profiler or nsys",
            ],
            code_fix="""
# Optimize DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,  # Increase workers
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

# Enable torch.compile
model = torch.compile(model, mode="reduce-overhead")
""",
        ),
        
        "gradient_overflow": TroubleshootingIssue(
            category=ErrorCategory.FRAMEWORK,
            severity="warning",
            title="Gradient Overflow/Underflow",
            description="Numerical instability in gradients",
            symptoms=[
                "Loss becomes NaN or Inf",
                "GradScaler skipping updates frequently",
                "Training diverges",
            ],
            root_causes=[
                "Learning rate too high",
                "FP16 without proper loss scaling",
                "Gradient explosion",
                "Numerical instability in model",
            ],
            solutions=[
                "1. Use BF16 instead of FP16 (no loss scaling needed)",
                "2. Enable gradient clipping: torch.nn.utils.clip_grad_norm_",
                "3. Reduce learning rate",
                "4. Check for NaN in inputs",
                "5. Use dynamic loss scaling with GradScaler",
            ],
            code_fix="""
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Or use BF16 instead of FP16
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    outputs = model(inputs)
""",
        ),
        
        "checkpoint_oom": TroubleshootingIssue(
            category=ErrorCategory.MEMORY,
            severity="warning",
            title="OOM During Checkpointing",
            description="Out of memory when saving checkpoints",
            symptoms=[
                "OOM when calling model.state_dict()",
                "OOM during torch.save()",
            ],
            root_causes=[
                "Full model gathered to single GPU for saving",
                "ZeRO-3/FSDP state dict gathering",
                "Large optimizer states",
            ],
            solutions=[
                "1. Use sharded checkpointing (FSDP/DeepSpeed)",
                "2. Save on CPU: model.cpu() before save",
                "3. Use streaming checkpoint saving",
                "4. Increase checkpoint interval to reduce frequency",
            ],
            code_fix="""
# For FSDP: Use sharded state dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state_dict = model.state_dict()
    torch.save(state_dict, f"checkpoint_rank{rank}.pt")
""",
        ),
    }
    
    # NCCL environment variable recommendations
    NCCL_TUNING = {
        "nvlink": {
            "NCCL_NVLS_ENABLE": "1",
            "NCCL_P2P_LEVEL": "NVL",
            "NCCL_NET_GDR_LEVEL": "5",
        },
        "infiniband": {
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_GID_INDEX": "3",
            "NCCL_IB_TIMEOUT": "23",
            "NCCL_IB_RETRY_CNT": "7",
        },
        "ethernet": {
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_NTHREADS": "8",
            "NCCL_NSOCKS_PERTHREAD": "4",
        },
        "debug": {
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,ENV",
            "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
        },
    }
    
    def diagnose(
        self,
        error_message: Optional[str] = None,
        symptoms: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[TroubleshootingIssue]:
        """
        Diagnose issues based on error message or symptoms.
        
        Returns list of matching issues with solutions.
        """
        matches = []
        
        # Match by error message
        if error_message:
            error_lower = error_message.lower()
            
            if "out of memory" in error_lower or "oom" in error_lower:
                matches.append(self.ERROR_DATABASE["cuda_oom"])
            
            if "nccl" in error_lower:
                if "timeout" in error_lower or "watchdog" in error_lower:
                    matches.append(self.ERROR_DATABASE["nccl_timeout"])
                elif "init" in error_lower or "failed" in error_lower:
                    matches.append(self.ERROR_DATABASE["nccl_init_failed"])
            
            if "size mismatch" in error_lower or "divisible" in error_lower:
                matches.append(self.ERROR_DATABASE["tp_mismatch"])
            
            if "nan" in error_lower or "inf" in error_lower:
                matches.append(self.ERROR_DATABASE["gradient_overflow"])
        
        # Match by symptoms
        if symptoms:
            for symptom in symptoms:
                symptom_lower = symptom.lower()
                for issue in self.ERROR_DATABASE.values():
                    for issue_symptom in issue.symptoms:
                        if symptom_lower in issue_symptom.lower():
                            if issue not in matches:
                                matches.append(issue)
        
        # Configuration-based checks
        if config:
            issues = self._check_config(config)
            matches.extend(issues)
        
        return matches
    
    def _check_config(self, config: Dict[str, Any]) -> List[TroubleshootingIssue]:
        """Check configuration for potential issues."""
        issues = []
        
        tp = config.get("tp", 1)
        pp = config.get("pp", 1)
        hidden_size = config.get("hidden_size", 8192)
        num_heads = config.get("num_attention_heads", 64)
        num_micro_batches = config.get("num_micro_batches", 1)
        
        # TP divisibility check
        if tp > 1:
            if hidden_size % tp != 0:
                issues.append(self.ERROR_DATABASE["tp_mismatch"])
            if num_heads % tp != 0:
                issues.append(self.ERROR_DATABASE["tp_mismatch"])
        
        # PP micro-batch check
        if pp > 1 and num_micro_batches < pp * 4:
            issues.append(self.ERROR_DATABASE["pp_imbalance"])
        
        return issues
    
    def get_nccl_recommendations(
        self,
        interconnect: str = "nvlink",
        debug: bool = False,
    ) -> Dict[str, str]:
        """Get recommended NCCL environment variables."""
        env_vars = {}
        
        if interconnect in self.NCCL_TUNING:
            env_vars.update(self.NCCL_TUNING[interconnect])
        
        if debug:
            env_vars.update(self.NCCL_TUNING["debug"])
        
        return env_vars
    
    def get_all_issues(self) -> List[Dict[str, Any]]:
        """Get all known issues with solutions."""
        return [
            {
                "id": key,
                "category": issue.category.value,
                "severity": issue.severity,
                "title": issue.title,
                "description": issue.description,
                "symptoms": issue.symptoms,
                "solutions": issue.solutions,
            }
            for key, issue in self.ERROR_DATABASE.items()
        ]


@dataclass
class MemoryBreakdown:
    """Detailed memory breakdown analysis."""
    total_memory_gb: float
    
    # Component breakdown
    parameters_gb: float
    gradients_gb: float
    optimizer_states_gb: float
    activations_gb: float
    kv_cache_gb: float
    workspace_gb: float
    fragmentation_gb: float
    
    # Per-component details
    details: Dict[str, Any]
    
    # Recommendations
    savings_opportunities: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_memory_gb": self.total_memory_gb,
            "breakdown": {
                "parameters": self.parameters_gb,
                "gradients": self.gradients_gb,
                "optimizer_states": self.optimizer_states_gb,
                "activations": self.activations_gb,
                "kv_cache": self.kv_cache_gb,
                "workspace": self.workspace_gb,
                "fragmentation": self.fragmentation_gb,
            },
            "details": self.details,
            "savings_opportunities": self.savings_opportunities,
        }


class MemoryAnalyzer:
    """Provides detailed memory breakdown and optimization recommendations."""
    
    def analyze(
        self,
        model_params_b: float,
        batch_size: int,
        seq_length: int,
        hidden_size: int,
        num_layers: int,
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
        precision_bytes: int = 2,
        optimizer: str = "adamw",
        gradient_checkpointing: bool = False,
        is_training: bool = True,
    ) -> MemoryBreakdown:
        """
        Provide detailed memory breakdown.
        """
        
        # Parameters
        params_bytes = model_params_b * 1e9 * precision_bytes
        params_per_gpu = params_bytes / (tp * pp)
        params_gb = params_per_gpu / 1e9
        
        # Gradients (same size as params if training)
        grads_gb = params_gb if is_training else 0
        
        # Optimizer states
        if is_training:
            if optimizer == "adamw":
                # 2 FP32 moments = 8 bytes per param
                optimizer_bytes = model_params_b * 1e9 * 8 / (tp * pp * dp)
            elif optimizer == "adamw_8bit":
                # Quantized moments
                optimizer_bytes = model_params_b * 1e9 * 2 / (tp * pp * dp)
            elif optimizer == "adafactor":
                # Factored moments
                optimizer_bytes = model_params_b * 1e9 * 0.5 / (tp * pp * dp)
            elif optimizer == "sgd":
                # Single momentum
                optimizer_bytes = model_params_b * 1e9 * 4 / (tp * pp * dp)
            else:
                optimizer_bytes = model_params_b * 1e9 * 8 / (tp * pp * dp)
            optimizer_gb = optimizer_bytes / 1e9
        else:
            optimizer_gb = 0
        
        # Activations
        if gradient_checkpointing:
            # Only store activations at checkpoint boundaries (~sqrt(layers))
            checkpoint_layers = max(1, int(num_layers ** 0.5))
            activation_factor = checkpoint_layers / num_layers
        else:
            activation_factor = 1.0
        
        # Activation memory per layer: batch * seq * hidden * 2 (for residual)
        activation_per_layer = (batch_size / dp) * seq_length * hidden_size * precision_bytes * 2
        total_activations = activation_per_layer * num_layers * activation_factor / (tp * pp)
        activations_gb = total_activations / 1e9
        
        # KV cache (inference)
        if not is_training:
            # 2 * num_layers * batch * seq * hidden / num_heads * 2 (K and V)
            kv_cache_bytes = 2 * num_layers * batch_size * seq_length * hidden_size * precision_bytes / tp
            kv_cache_gb = kv_cache_bytes / 1e9
        else:
            kv_cache_gb = 0
        
        # Workspace and temporary buffers (~10% of model)
        workspace_gb = params_gb * 0.1
        
        # Fragmentation estimate (~5-15%)
        subtotal = params_gb + grads_gb + optimizer_gb + activations_gb + kv_cache_gb + workspace_gb
        fragmentation_gb = subtotal * 0.1
        
        total = subtotal + fragmentation_gb
        
        # Savings opportunities
        savings = []
        
        if precision_bytes == 4:
            savings.append({
                "technique": "Mixed Precision (BF16/FP16)",
                "savings_gb": params_gb * 0.5 + activations_gb * 0.5,
                "difficulty": "easy",
                "impact": "50% memory reduction for params and activations",
            })
        
        if not gradient_checkpointing and is_training:
            savings.append({
                "technique": "Gradient Checkpointing",
                "savings_gb": activations_gb * 0.9,
                "difficulty": "easy",
                "impact": "~90% activation memory reduction (33% compute overhead)",
            })
        
        if optimizer == "adamw" and is_training:
            savings.append({
                "technique": "8-bit Optimizer",
                "savings_gb": optimizer_gb * 0.75,
                "difficulty": "easy",
                "impact": "75% optimizer state reduction",
            })
        
        if dp > 1 and is_training:
            current_opt_gb = optimizer_gb
            zero3_opt_gb = optimizer_gb / dp
            savings.append({
                "technique": "ZeRO-3/FSDP",
                "savings_gb": current_opt_gb - zero3_opt_gb + grads_gb * (1 - 1/dp),
                "difficulty": "medium",
                "impact": f"Shard optimizer and gradients across {dp} GPUs",
            })
        
        details = {
            "model_params_billion": model_params_b,
            "precision": "FP32" if precision_bytes == 4 else "BF16/FP16",
            "optimizer": optimizer,
            "gradient_checkpointing": gradient_checkpointing,
            "parallelism": {"tp": tp, "pp": pp, "dp": dp},
        }
        
        return MemoryBreakdown(
            total_memory_gb=total,
            parameters_gb=params_gb,
            gradients_gb=grads_gb,
            optimizer_states_gb=optimizer_gb,
            activations_gb=activations_gb,
            kv_cache_gb=kv_cache_gb,
            workspace_gb=workspace_gb,
            fragmentation_gb=fragmentation_gb,
            details=details,
            savings_opportunities=savings,
        )


def diagnose_error(
    error_message: Optional[str] = None,
    symptoms: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Diagnose distributed training errors.
    """
    troubleshooter = DistributedTrainingTroubleshooter()
    issues = troubleshooter.diagnose(error_message, symptoms, config)
    
    return {
        "issues_found": len(issues),
        "issues": [
            {
                "category": issue.category.value,
                "severity": issue.severity,
                "title": issue.title,
                "description": issue.description,
                "symptoms": issue.symptoms,
                "root_causes": issue.root_causes,
                "solutions": issue.solutions,
                "code_fix": issue.code_fix,
                "env_vars": issue.env_vars,
            }
            for issue in issues
        ],
    }


def get_nccl_tuning(
    interconnect: str = "nvlink",
    debug: bool = False,
) -> Dict[str, Any]:
    """Get NCCL tuning recommendations."""
    troubleshooter = DistributedTrainingTroubleshooter()
    env_vars = troubleshooter.get_nccl_recommendations(interconnect, debug)
    
    return {
        "interconnect": interconnect,
        "debug_enabled": debug,
        "environment_variables": env_vars,
        "export_commands": [f"export {k}={v}" for k, v in env_vars.items()],
    }


def get_memory_breakdown(
    model_config: Dict[str, Any],
    hardware_config: Dict[str, Any],
    parallelism_config: Dict[str, Any],
    training_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get detailed memory breakdown analysis."""
    
    training = training_config or {}
    
    analyzer = MemoryAnalyzer()
    breakdown = analyzer.analyze(
        model_params_b=model_config.get("parameters_billions", 70),
        batch_size=training.get("batch_size", 8),
        seq_length=model_config.get("max_sequence_length", 4096),
        hidden_size=model_config.get("hidden_size", 8192),
        num_layers=model_config.get("num_layers", 80),
        tp=parallelism_config.get("tensor_parallel", 1),
        pp=parallelism_config.get("pipeline_parallel", 1),
        dp=parallelism_config.get("data_parallel", 8),
        precision_bytes=2,  # BF16
        optimizer=training.get("optimizer", "adamw"),
        gradient_checkpointing=training.get("gradient_checkpointing", False),
        is_training=training.get("is_training", True),
    )
    
    return breakdown.to_dict()


def get_all_troubleshooting_topics() -> List[Dict[str, Any]]:
    """Get all troubleshooting topics."""
    troubleshooter = DistributedTrainingTroubleshooter()
    return troubleshooter.get_all_issues()



