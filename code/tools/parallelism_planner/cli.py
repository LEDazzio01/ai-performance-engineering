#!/usr/bin/env python3
"""
Parallelism Strategy Advisor - Unified CLI

Provides CLI access to all parallelism planning features:
- Parallelism recommendations (TP/PP/DP/CP/EP)
- Sharding strategies (ZeRO/FSDP/HSDP)
- Launch command generation (torchrun/deepspeed/accelerate)
- Cost/throughput Pareto analysis
- Calibration from benchmark data

Usage:
    # Parallelism recommendations
    python -m tools.parallelism_planner recommend llama-3.1-70b --training
    
    # Sharding recommendations
    python -m tools.parallelism_planner sharding llama-3.1-70b --dp 8 --memory 80
    
    # Generate launch commands
    python -m tools.parallelism_planner launch --nodes 2 --gpus 8 --tp 4 --pp 2 --dp 2
    
    # Pareto analysis
    python -m tools.parallelism_planner pareto llama-3.1-70b --gpu-cost 4.0
    
    # Calibration
    python -m tools.parallelism_planner calibrate
    
    # List presets
    python -m tools.parallelism_planner presets
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_recommend(args):
    """Handle recommend subcommand."""
    from .advisor import ParallelismAdvisor, create_mock_topology_8xb200, create_mock_topology_8xh100
    
    advisor = ParallelismAdvisor(auto_detect_topology=False)
    
    # Set topology
    if args.mock_topology:
        if args.mock_topology == "8xB200":
            advisor.set_topology(create_mock_topology_8xb200())
        else:
            advisor.set_topology(create_mock_topology_8xh100())
    else:
        try:
            advisor.detect_topology()
        except RuntimeError:
            print("Warning: Could not detect topology, using mock 8xB200...")
            advisor.set_topology(create_mock_topology_8xb200())
    
    if args.json:
        result = advisor.recommend(
            model=args.model,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            goal=args.goal,
            is_training=args.training,
        )
        print(result.to_json())
    else:
        report = advisor.generate_report(
            model=args.model,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            goal=args.goal,
            is_training=args.training,
        )
        print(report)


def cmd_sharding(args):
    """Handle sharding subcommand."""
    from .model_analyzer import ModelAnalyzer
    from .sharding_strategies import ShardingOptimizer
    
    analyzer = ModelAnalyzer()
    model = analyzer.analyze(args.model)
    
    optimizer = ShardingOptimizer()
    recommendations = optimizer.recommend(
        model=model,
        dp_size=args.dp,
        gpu_memory_gb=args.memory,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_nodes=args.nodes,
        gpus_per_node=args.gpus,
    )
    
    if args.json:
        print(json.dumps([r.to_dict() for r in recommendations], indent=2))
    else:
        print(optimizer.format_recommendations(recommendations))
        
        # Show DeepSpeed config for best option if ZeRO
        if recommendations and recommendations[0].deepspeed_config:
            print("\n" + "=" * 60)
            print("DEEPSPEED CONFIG (ds_config.json):")
            print("=" * 60)
            print(json.dumps(recommendations[0].deepspeed_config, indent=2))


def cmd_launch(args):
    """Handle launch subcommand."""
    from .launch_commands import LaunchCommandGenerator, LaunchConfig, ShardingStrategy
    
    # Map sharding string to enum
    sharding_map = {
        "none": ShardingStrategy.NO_SHARD,
        "zero1": ShardingStrategy.ZERO_1,
        "zero2": ShardingStrategy.ZERO_2,
        "zero3": ShardingStrategy.ZERO_3,
        "fsdp": ShardingStrategy.FSDP_FULL,
        "hsdp": ShardingStrategy.HSDP,
    }
    sharding = sharding_map.get(args.sharding.lower(), ShardingStrategy.NO_SHARD)
    
    config = LaunchConfig(
        num_nodes=args.nodes,
        gpus_per_node=args.gpus,
        tp_size=args.tp,
        pp_size=args.pp,
        dp_size=args.dp,
        sharding=sharding,
        micro_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        master_addr=args.master_addr,
        master_port=args.master_port,
    )
    
    gen = LaunchCommandGenerator()
    
    if args.json:
        all_commands = gen.generate_all(config, args.script)
        print(json.dumps(all_commands, indent=2, default=str))
    elif args.framework:
        # Output specific framework only
        all_commands = gen.generate_all(config, args.script)
        fw = args.framework.lower()
        if fw in all_commands:
            if fw == "deepspeed":
                print("# Command:")
                print(all_commands[fw]["command"])
                print("\n# Config (save as ds_config.json):")
                print(json.dumps(all_commands[fw]["config"], indent=2))
            elif fw == "accelerate":
                print("# Command:")
                print(all_commands[fw]["command"])
                print("\n# Config (save as accelerate_config.yaml):")
                import yaml
                try:
                    print(yaml.dump(all_commands[fw]["config"], default_flow_style=False))
                except ImportError:
                    print(json.dumps(all_commands[fw]["config"], indent=2))
            else:
                print(all_commands[fw].get("command", json.dumps(all_commands[fw], indent=2)))
    else:
        print(gen.format_launch_guide(config, args.script))


def cmd_pareto(args):
    """Handle pareto subcommand."""
    from .advisor import ParallelismAdvisor, create_mock_topology_8xb200
    from .pareto_analysis import ParetoAnalyzer, ConfigurationPoint
    
    # Get parallelism recommendations
    advisor = ParallelismAdvisor(auto_detect_topology=False)
    try:
        advisor.detect_topology()
    except RuntimeError:
        advisor.set_topology(create_mock_topology_8xb200())
    
    result = advisor.recommend(
        model=args.model,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        is_training=args.training,
    )
    
    # Convert to ConfigurationPoints
    configs = []
    for rec in result.recommendations:
        s = rec.strategy
        a = rec.analysis
        configs.append(ConfigurationPoint(
            name=f"TP{s.tp}_PP{s.pp}_DP{s.dp}" + (f"_CP{s.cp}" if s.cp > 1 else ""),
            tp=s.tp,
            pp=s.pp,
            dp=s.dp,
            throughput_tps=a.estimated_throughput_tps,
            latency_ms=a.estimated_latency_ms,
            memory_per_gpu_gb=a.memory_per_gpu_gb,
            num_gpus=s.world_size,
        ))
    
    pareto = ParetoAnalyzer(gpu_hourly_cost=args.gpu_cost)
    
    if args.json:
        analysis = pareto.generate_cost_throughput_analysis(configs)
        viz = pareto.generate_visualization_data(configs)
        print(json.dumps({"analysis": analysis, "visualization": viz}, indent=2))
    else:
        print(pareto.format_pareto_report(configs))


def cmd_calibrate(args):
    """Handle calibrate subcommand."""
    from .calibration import CalibrationEngine
    
    engine = CalibrationEngine()
    
    if args.data_dir:
        loaded = engine.load_benchmark_data([Path(args.data_dir)])
    else:
        loaded = engine.load_benchmark_data()
    
    model = engine.calibrate()
    
    if args.json:
        print(json.dumps({
            "data_points_loaded": loaded,
            "calibration_model": model.to_dict(),
        }, indent=2))
    else:
        print(engine.format_calibration_report())


def cmd_presets(args):
    """Handle presets subcommand."""
    from .model_analyzer import ModelAnalyzer, MODEL_PRESETS
    
    analyzer = ModelAnalyzer()
    
    if args.json:
        presets = []
        for name in MODEL_PRESETS.keys():
            arch = analyzer.analyze(name)
            presets.append({
                "name": name,
                "params_b": arch.total_params_billion,
                "type": arch.model_type.value,
                "layers": arch.num_layers,
                "hidden_size": arch.hidden_size,
            })
        print(json.dumps(presets, indent=2))
    else:
        print("=" * 70)
        print("AVAILABLE MODEL PRESETS")
        print("=" * 70)
        print(f"{'Name':<20} {'Params':<12} {'Type':<10} {'Layers':<8} {'Hidden':<10}")
        print("-" * 70)
        for name in sorted(MODEL_PRESETS.keys()):
            arch = analyzer.analyze(name)
            print(f"{name:<20} {arch.total_params_billion:>8.1f}B   {arch.model_type.value:<10} {arch.num_layers:<8} {arch.hidden_size:<10}")
        print("=" * 70)


def cmd_topology(args):
    """Handle topology subcommand."""
    from .topology_detector import TopologyDetector
    from .advisor import create_mock_topology_8xb200, create_mock_topology_8xh100
    
    detector = TopologyDetector()
    
    if args.mock:
        if args.mock == "8xB200":
            topology = create_mock_topology_8xb200()
        else:
            topology = create_mock_topology_8xh100()
    else:
        try:
            topology = detector.detect()
        except RuntimeError as e:
            print(f"Error detecting topology: {e}")
            print("Use --mock 8xB200 or --mock 8xH100 to use a mock topology")
            sys.exit(1)
    
    if args.json:
        print(json.dumps(topology.to_dict(), indent=2))
    else:
        print(detector.format_topology_report(topology))


def cmd_analyze(args):
    """Handle analyze subcommand (model analysis)."""
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    if args.json:
        mem_inference = arch.estimate_memory_gb(batch_size=1, seq_length=args.seq_length)
        mem_training = arch.estimate_memory_gb(batch_size=args.batch_size, seq_length=args.seq_length, include_optimizer=True)
        print(json.dumps({
            "architecture": arch.to_dict(),
            "memory_inference": mem_inference,
            "memory_training": mem_training,
        }, indent=2))
    else:
        print(analyzer.format_architecture_report(arch))


def cmd_estimate(args):
    """Handle estimate subcommand (training time/cost)."""
    from .extras import TrainingEstimator
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    model = analyzer.analyze(args.model)
    
    estimator = TrainingEstimator(gpu_hourly_cost=args.gpu_cost)
    estimate = estimator.estimate(
        total_tokens=args.tokens,
        tokens_per_second=args.throughput,
        num_gpus=args.gpus,
        model_params_billion=model.total_params_billion,
        checkpoint_interval_tokens=args.checkpoint_interval,
    )
    
    if args.json:
        print(json.dumps(estimate.to_dict(), indent=2))
    else:
        print(estimator.format_estimate(estimate))


def cmd_compare(args):
    """Handle compare subcommand (multi-model comparison)."""
    from .extras import ModelComparator
    from .advisor import create_mock_topology_8xb200, create_mock_topology_8xh100
    from .topology_detector import TopologyDetector
    
    # Get topology
    if args.mock_topology:
        if args.mock_topology == "8xB200":
            topology = create_mock_topology_8xb200()
        else:
            topology = create_mock_topology_8xh100()
    else:
        try:
            detector = TopologyDetector()
            topology = detector.detect()
        except RuntimeError:
            topology = create_mock_topology_8xb200()
    
    comparator = ModelComparator()
    results = comparator.compare(
        models=args.models,
        topology=topology,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        is_training=args.training,
    )
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(comparator.format_comparison(results))


def cmd_slurm(args):
    """Handle slurm subcommand (job script generation)."""
    from .extras import JobScriptGenerator
    
    generator = JobScriptGenerator()
    script = generator.generate_slurm(
        job_name=args.job_name,
        num_nodes=args.nodes,
        gpus_per_node=args.gpus,
        time_hours=args.time,
        partition=args.partition,
        account=args.account,
        script=args.script,
        conda_env=args.conda_env,
    )
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(script)
        print(f"Wrote job script to {args.output}")
    else:
        print(script)


def cmd_validate(args):
    """Handle validate subcommand (configuration validation)."""
    from .validation import validate_full_configuration, ConfigValidator
    from .model_analyzer import ModelAnalyzer
    from .advisor import ParallelismAdvisor, create_mock_topology_8xb200
    
    # Get model info
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    # Get topology
    advisor = ParallelismAdvisor(auto_detect_topology=False)
    try:
        advisor.detect_topology()
        topology = advisor.topology
    except RuntimeError:
        topology = create_mock_topology_8xb200()
    
    # Get memory estimate (returns dict with breakdown)
    mem_estimate = arch.estimate_memory_gb(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        include_optimizer=args.training
    )
    # Extract total memory from dict
    predicted_memory = mem_estimate.get('total', sum(mem_estimate.values())) if isinstance(mem_estimate, dict) else mem_estimate
    
    # Build strategy dict
    strategy = {
        "data_parallel": args.dp,
        "tensor_parallel": args.tp,
        "pipeline_parallel": args.pp,
        "context_parallel": args.cp,
        "expert_parallel": args.ep,
        "predicted_memory_gb": predicted_memory,
    }
    
    # Get per-GPU memory from first GPU or divide total
    gpu_memory_gb = topology.gpus[0].memory_gb if topology.gpus else topology.total_memory_gb / max(1, topology.num_gpus)
    
    hardware = {
        "num_gpus": topology.num_gpus,
        "gpu_memory_gb": gpu_memory_gb,
        "has_nvlink": topology.has_nvlink,
        "nvlink_bandwidth_gbps": topology.nvlink_bandwidth_gbps if hasattr(topology, 'nvlink_bandwidth_gbps') else 600,
    }
    
    model = {
        "num_layers": arch.num_layers,
        "hidden_size": arch.hidden_size,
        "num_experts": arch.num_experts if hasattr(arch, 'num_experts') else 1,
        "max_sequence_length": args.seq_length,
    }
    
    result = validate_full_configuration(strategy, hardware, model)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("CONFIGURATION VALIDATION REPORT")
        print("=" * 60)
        print(f"\nModel: {args.model}")
        print(f"Strategy: DP={args.dp}, TP={args.tp}, PP={args.pp}, CP={args.cp}, EP={args.ep}")
        print(f"\nResult: {'✓ VALID' if result['valid'] else '✗ INVALID'}")
        print(f"Passed: {result['summary']['passed']}, Failed: {result['summary']['failed']}, Warnings: {result['summary']['warnings']}")
        print("\n" + "-" * 60)
        for check in result['checks']:
            icon = "✓" if check['status'] == 'passed' else ("⚠" if check['status'] == 'warning' else "✗")
            print(f"  {icon} {check['name']}: {check['message']}")
        print("=" * 60)


def cmd_dryrun(args):
    """Handle dry-run subcommand (quick sanity check)."""
    from .validation import DryRunner
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    model_config = {
        "hidden_size": arch.hidden_size,
        "num_layers": arch.num_layers,
    }
    
    strategy = {
        "data_parallel": args.dp,
        "tensor_parallel": args.tp,
        "pipeline_parallel": args.pp,
    }
    
    runner = DryRunner(timeout_seconds=args.timeout)
    
    print(f"Running dry-run test for {args.model}...")
    print(f"Configuration: DP={args.dp}, TP={args.tp}, PP={args.pp}")
    print(f"Timeout: {args.timeout}s")
    print()
    
    result = runner.run_dry_test(model_config, strategy)
    
    if args.json:
        print(json.dumps({
            "success": result.success,
            "duration_seconds": result.duration_seconds,
            "memory_allocated_gb": result.memory_allocated_gb,
            "memory_peak_gb": result.memory_peak_gb,
            "checks": [{"name": c.name, "status": c.status.value, "message": c.message} for c in result.checks],
            "errors": result.errors,
            "warnings": result.warnings,
        }, indent=2))
    else:
        print("=" * 60)
        print("DRY-RUN RESULTS")
        print("=" * 60)
        print(f"\nResult: {'✓ SUCCESS' if result.success else '✗ FAILED'}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print(f"Memory - Allocated: {result.memory_allocated_gb:.2f} GB, Peak: {result.memory_peak_gb:.2f} GB")
        print("\nChecks:")
        for check in result.checks:
            icon = "✓" if check.status.value == 'passed' else "✗"
            print(f"  {icon} {check.name}: {check.message}")
        if result.errors:
            print("\nErrors:")
            for err in result.errors:
                print(f"  ✗ {err}")
        if result.warnings:
            print("\nWarnings:")
            for warn in result.warnings:
                print(f"  ⚠ {warn}")
        print("=" * 60)


def cmd_optimize(args):
    """Handle optimize subcommand (advanced optimizations)."""
    from .advanced_optimizations import get_advanced_optimization_report
    from .model_analyzer import ModelAnalyzer
    from .advisor import ParallelismAdvisor, create_mock_topology_8xb200
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    # Get topology
    advisor = ParallelismAdvisor(auto_detect_topology=False)
    try:
        advisor.detect_topology()
        topology = advisor.topology
    except RuntimeError:
        topology = create_mock_topology_8xb200()
    
    model_config = {
        "parameters_billions": arch.total_params_billion,
        "num_layers": arch.num_layers,
        "hidden_size": arch.hidden_size,
        "max_sequence_length": args.seq_length,
        "batch_size": args.batch_size,
    }
    
    # Get per-GPU memory from first GPU or divide total
    gpu_memory_gb = topology.gpus[0].memory_gb if topology.gpus else topology.total_memory_gb / max(1, topology.num_gpus)
    gpu_arch = topology.gpus[0].architecture if topology.gpus else 'ampere'
    
    hardware_config = {
        "gpu_arch": gpu_arch.lower() if gpu_arch else 'ampere',
        "gpu_memory_gb": gpu_memory_gb,
        "num_gpus": topology.num_gpus,
        "has_nvlink": topology.has_nvlink,
    }
    
    report = get_advanced_optimization_report(model_config, hardware_config, args.goal)
    
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 70)
        print("ADVANCED OPTIMIZATION REPORT")
        print("=" * 70)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B params)")
        print(f"Optimization Goal: {args.goal.upper()}")
        
        compound = report["compound_optimization"]
        print(f"\n{'─' * 70}")
        print(f"COMPOUND OPTIMIZATION: {compound['name']}")
        print(f"{'─' * 70}")
        print(f"Complexity: {compound['complexity'].upper()}")
        print(f"Memory Savings: {compound['total_memory_savings_gb']:.1f} GB")
        print(f"Throughput Boost: ~{compound['total_throughput_boost_pct']:.0f}%")
        
        print("\nTechniques Applied:")
        for tech in compound['techniques']:
            print(f"  • {tech}")
        
        print("\nImplementation Steps:")
        for i, step in enumerate(compound['implementation_steps'], 1):
            print(f"  {i}. {step}")
        
        if compound['compatibility_notes']:
            print("\nNotes:")
            for note in compound['compatibility_notes']:
                print(f"  ⚠ {note}")
        
        print(f"\n{'─' * 70}")
        print("PRECISION SETTINGS")
        print(f"{'─' * 70}")
        prec = report["precision"]
        print(f"  Mode: {prec['mode']}")
        print(f"  Compute: {prec['compute_dtype']}, Params: {prec['param_dtype']}")
        print(f"  Memory Savings: {prec['memory_savings_pct']}%")
        print(f"  Throughput Boost: {prec['throughput_boost_pct']}%")
        print(f"  Accuracy Impact: {prec['accuracy_impact']}")
        
        print(f"\n{'─' * 70}")
        print("KERNEL OPTIMIZATIONS")
        print(f"{'─' * 70}")
        kernels = report["kernels"]
        fa = kernels.get("flash_attention", {})
        print(f"  Flash Attention: {fa.get('version', 'N/A')}")
        for fk in kernels.get("fused_kernels", [])[:4]:
            print(f"  • {fk['name']}: {fk['benefit']}")
        
        print(f"\n{'─' * 70}")
        print("HARDWARE FEATURES UTILIZED")
        print(f"{'─' * 70}")
        for feat in report.get("hardware_features_used", []):
            print(f"  ✓ {feat}")
        
        print("=" * 70)


def cmd_profile(args):
    """Handle profile subcommand (workload-specific profiles)."""
    from .performance_profiles import get_performance_profile, list_available_profiles
    from .advisor import ParallelismAdvisor, create_mock_topology_8xb200
    from .model_analyzer import ModelAnalyzer
    
    if args.list:
        profiles = list_available_profiles()
        print("=" * 60)
        print("AVAILABLE PERFORMANCE PROFILES")
        print("=" * 60)
        for p in profiles:
            print(f"  {p['name']:<20} {p['description']}")
        print("=" * 60)
        return
    
    if not args.model:
        print("Error: model is required (use --list to see available profiles)")
        sys.exit(1)
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    # Get topology
    advisor = ParallelismAdvisor(auto_detect_topology=False)
    try:
        advisor.detect_topology()
        topology = advisor.topology
    except RuntimeError:
        topology = create_mock_topology_8xb200()
    
    # Get per-GPU memory from first GPU or divide total
    gpu_memory_gb = topology.gpus[0].memory_gb if topology.gpus else topology.total_memory_gb / max(1, topology.num_gpus)
    gpu_arch = topology.gpus[0].architecture if topology.gpus else 'ampere'
    
    hardware_config = {
        "gpu_arch": gpu_arch.lower() if gpu_arch else 'ampere',
        "gpu_memory_gb": gpu_memory_gb,
        "num_gpus": topology.num_gpus,
        "has_nvlink": topology.has_nvlink,
        "has_infiniband": False,
        "num_nodes": 1,
    }
    
    profile = get_performance_profile(
        model_params_b=arch.total_params_billion,
        workload=args.workload,
        hardware_config=hardware_config,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        use_lora=args.lora,
        mode=args.inference_mode,
    )
    
    if args.json:
        print(json.dumps(profile, indent=2))
    else:
        print("=" * 70)
        print(f"PERFORMANCE PROFILE: {profile['name'].upper()}")
        print("=" * 70)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"Workload: {profile['workload_type']}")
        print(f"Priority: {profile['priority']}")
        print(f"Description: {profile['description']}")
        
        print(f"\n{'─' * 70}")
        print("PARALLELISM CONFIGURATION")
        print(f"{'─' * 70}")
        par = profile["parallelism"]
        print(f"  Data Parallel: {par['data_parallel']}")
        print(f"  Tensor Parallel: {par['tensor_parallel']}")
        print(f"  Pipeline Parallel: {par['pipeline_parallel']}")
        print(f"  Context Parallel: {par.get('context_parallel', 1)}")
        if par.get('expert_parallel'):
            print(f"  Expert Parallel: {par['expert_parallel']}")
        if par.get('sequence_parallel'):
            print(f"  Sequence Parallel: ✓")
        
        print(f"\n{'─' * 70}")
        print("PRECISION & MEMORY")
        print(f"{'─' * 70}")
        prec = profile["precision"]
        mem = profile["memory"]
        print(f"  Compute dtype: {prec.get('compute_dtype', 'bf16')}")
        print(f"  Param dtype: {prec.get('param_dtype', 'bf16')}")
        print(f"  Gradient Checkpointing: {'✓' if mem.get('gradient_checkpointing') else '✗'}")
        if mem.get('optimizer'):
            print(f"  Optimizer: {mem['optimizer']}")
        if mem.get('cpu_offload'):
            print(f"  CPU Offload: ✓")
        
        print(f"\n{'─' * 70}")
        print("BATCH SETTINGS")
        print(f"{'─' * 70}")
        batch = profile["batch_settings"]
        print(f"  Global Batch Size: {batch.get('global_batch_size', 'N/A')}")
        print(f"  Micro Batch Size: {batch.get('micro_batch_size', 'N/A')}")
        print(f"  Grad Accumulation: {batch.get('gradient_accumulation_steps', 1)}")
        print(f"  Sequence Length: {batch.get('sequence_length', args.seq_length)}")
        
        if profile.get('expected_mfu'):
            print(f"\n{'─' * 70}")
            print("EXPECTED PERFORMANCE")
            print(f"{'─' * 70}")
            if profile.get('expected_throughput_tps'):
                print(f"  Throughput: ~{profile['expected_throughput_tps']:.0f} tokens/s")
            print(f"  Model FLOPS Utilization: ~{profile['expected_mfu']*100:.0f}%")
        
        if profile.get('warnings'):
            print(f"\n{'─' * 70}")
            print("WARNINGS & NOTES")
            print(f"{'─' * 70}")
            for warn in profile['warnings']:
                print(f"  ⚠ {warn}")
        
        if profile.get('prerequisites'):
            print(f"\n{'─' * 70}")
            print("PREREQUISITES")
            print(f"{'─' * 70}")
            for prereq in profile['prerequisites']:
                print(f"  • {prereq}")
        
        if profile.get('deepspeed_config'):
            print(f"\n{'─' * 70}")
            print("DEEPSPEED CONFIG")
            print(f"{'─' * 70}")
            print(json.dumps(profile['deepspeed_config'], indent=2))
        
        print("=" * 70)


def cmd_bottleneck(args):
    """Handle bottleneck subcommand."""
    from .bottleneck_analysis import analyze_bottlenecks
    from .model_analyzer import ModelAnalyzer
    from .advisor import ParallelismAdvisor, create_mock_topology_8xb200
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    advisor = ParallelismAdvisor(auto_detect_topology=False)
    try:
        advisor.detect_topology()
        topology = advisor.topology
    except RuntimeError:
        topology = create_mock_topology_8xb200()
    
    gpu_memory_gb = topology.gpus[0].memory_gb if topology.gpus else topology.total_memory_gb / max(1, topology.num_gpus)
    
    model_config = {
        "parameters_billions": arch.total_params_billion,
        "batch_size": args.batch_size,
        "max_sequence_length": args.seq_length,
        "hidden_size": arch.hidden_size,
        "num_layers": arch.num_layers,
    }
    
    hardware_config = {
        "gpu_type": "h100",
        "num_gpus": args.tp * args.pp * args.dp,
        "gpu_memory_gb": gpu_memory_gb,
        "has_nvlink": topology.has_nvlink,
    }
    
    parallelism_config = {
        "tensor_parallel": args.tp,
        "pipeline_parallel": args.pp,
        "data_parallel": args.dp,
    }
    
    result = analyze_bottlenecks(
        model_config, hardware_config, parallelism_config, args.throughput
    )
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("BOTTLENECK ANALYSIS")
        print("=" * 60)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"Configuration: TP={args.tp}, PP={args.pp}, DP={args.dp}")
        print(f"\nPrimary Bottleneck: {result['primary_bottleneck'].upper()}")
        if result['secondary_bottleneck']:
            print(f"Secondary Bottleneck: {result['secondary_bottleneck']}")
        print(f"Severity: {result['bottleneck_severity']*100:.0f}%")
        print(f"\nUtilization:")
        print(f"  Compute: {result['utilization']['compute']*100:.1f}%")
        print(f"  Memory: {result['utilization']['memory']*100:.1f}%")
        print(f"  Communication: {result['utilization']['communication']*100:.1f}%")
        print(f"  Memory Bandwidth: {result['utilization']['memory_bandwidth']*100:.1f}%")
        print(f"\nRecommendations:")
        for rec in result['recommendations']:
            print(f"  {rec}")
        print("=" * 60)


def cmd_scaling(args):
    """Handle scaling subcommand."""
    from .bottleneck_analysis import analyze_scaling
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    model_config = {
        "parameters_billions": arch.total_params_billion,
        "batch_size": 8,
        "max_sequence_length": 4096,
    }
    
    hardware_config = {
        "gpu_type": "h100",
        "num_gpus": args.gpus,
        "has_nvlink": True,
    }
    
    result = analyze_scaling(model_config, hardware_config, args.throughput, args.max_gpus)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("SCALING ANALYSIS")
        print("=" * 60)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"Current: {args.gpus} GPUs @ {args.throughput:.0f} tokens/s")
        print(f"\nScaling Efficiency:")
        print(f"  Strong Scaling: {result['strong_scaling_efficiency']*100:.1f}%")
        print(f"  Weak Scaling: {result['weak_scaling_efficiency']*100:.1f}%")
        print(f"\nOptimal Scale: {result['optimal_gpu_count']} GPUs ({result['optimal_efficiency']*100:.0f}% efficiency)")
        print(f"\nProjections:")
        for proj in result['scaling_projections']:
            print(f"  {proj['gpus']:>4} GPUs: {proj['throughput_tps']:>10.0f} tokens/s ({proj['efficiency']*100:.0f}% eff)")
        print(f"\nRecommendations:")
        for rec in result['scaling_recommendations']:
            print(f"  • {rec}")
        print("=" * 60)


def cmd_whatif(args):
    """Handle whatif subcommand."""
    from .bottleneck_analysis import analyze_whatif
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    current_config = {
        "model_params_b": arch.total_params_billion,
        "batch_size": args.current_batch,
        "seq_length": 4096,
        "num_gpus": args.current_tp * args.current_pp * args.current_dp,
        "tp": args.current_tp,
        "pp": args.current_pp,
        "dp": args.current_dp,
        "gpu_type": "h100",
        "gpu_memory_gb": 80,
    }
    
    result = analyze_whatif(current_config)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("WHAT-IF ANALYSIS")
        print("=" * 60)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"Current: TP={args.current_tp}, PP={args.current_pp}, DP={args.current_dp}, Batch={args.current_batch}")
        print(f"\nScenarios:")
        for scenario in result['scenarios']:
            icon = "✓" if scenario['is_feasible'] else "✗"
            print(f"\n  {icon} {scenario['scenario']}")
            print(f"    Throughput: {scenario['change_vs_current']['throughput_pct']:+.1f}%")
            print(f"    Memory: {scenario['projected']['memory_per_gpu']:.1f} GB")
            if scenario['trade_offs']:
                for tradeoff in scenario['trade_offs']:
                    print(f"    → {tradeoff}")
            if not scenario['is_feasible']:
                for issue in scenario['feasibility_issues']:
                    print(f"    ⚠ {issue}")
        print("=" * 60)


def cmd_batchsize(args):
    """Handle batch-size subcommand."""
    from .auto_tuning import find_max_batch_size
    from .model_analyzer import ModelAnalyzer
    from .advisor import ParallelismAdvisor, create_mock_topology_8xb200
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    advisor = ParallelismAdvisor(auto_detect_topology=False)
    try:
        advisor.detect_topology()
        topology = advisor.topology
    except RuntimeError:
        topology = create_mock_topology_8xb200()
    
    gpu_memory_gb = topology.gpus[0].memory_gb if topology.gpus else topology.total_memory_gb / max(1, topology.num_gpus)
    
    model_config = {
        "parameters_billions": arch.total_params_billion,
        "max_sequence_length": args.seq_length,
        "hidden_size": arch.hidden_size,
        "num_layers": arch.num_layers,
    }
    
    hardware_config = {"gpu_memory_gb": gpu_memory_gb}
    parallelism_config = {"tensor_parallel": args.tp, "pipeline_parallel": args.pp, "data_parallel": args.dp}
    
    result = find_max_batch_size(model_config, hardware_config, parallelism_config, args.target_batch)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("BATCH SIZE ANALYSIS")
        print("=" * 60)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
        print(f"\nMaximum Micro-Batch Size: {result['max_batch_size']}")
        print(f"Memory at Max: {result['memory_at_max_gb']:.1f} GB")
        print(f"Headroom: {result['memory_headroom_gb']:.1f} GB")
        print(f"\nRecommended Settings:")
        print(f"  Micro-Batch Size: {result['recommended']['batch_size']}")
        print(f"  Gradient Accumulation: {result['recommended']['gradient_accumulation']}")
        print(f"  Effective Batch Size: {result['recommended']['effective_batch_size']}")
        print(f"\nEstimated Throughput: {result['throughput_estimate_tps']:.0f} tokens/s")
        print("=" * 60)


def cmd_autotune(args):
    """Handle auto-tune subcommand."""
    from .auto_tuning import auto_tune_config
    from .model_analyzer import ModelAnalyzer
    from .advisor import ParallelismAdvisor, create_mock_topology_8xb200
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    advisor = ParallelismAdvisor(auto_detect_topology=False)
    try:
        advisor.detect_topology()
        topology = advisor.topology
    except RuntimeError:
        topology = create_mock_topology_8xb200()
    
    gpu_memory_gb = topology.gpus[0].memory_gb if topology.gpus else topology.total_memory_gb / max(1, topology.num_gpus)
    
    model_config = {
        "parameters_billions": arch.total_params_billion,
        "max_sequence_length": 4096,
        "hidden_size": arch.hidden_size,
        "num_layers": arch.num_layers,
    }
    
    hardware_config = {
        "gpu_memory_gb": gpu_memory_gb,
        "num_gpus": topology.num_gpus,
        "has_nvlink": topology.has_nvlink,
    }
    
    result = auto_tune_config(model_config, hardware_config, args.target_batch, args.goal)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("AUTO-TUNE RESULTS")
        print("=" * 60)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"Goal: {args.goal.upper()}")
        print(f"Configurations Evaluated: {result['search_statistics']['total_iterations']}")
        print(f"\nBest Configuration:")
        cfg = result['best_config']
        print(f"  TP={cfg['tp']}, PP={cfg['pp']}, DP={cfg['dp']}")
        print(f"  Score: {result['best_score']:.2f}")
        print(f"\nRecommendations:")
        for rec in result['recommendations']:
            print(f"  • {rec}")
        print(f"\nTop Alternatives:")
        for alt in result['evaluated_configs'][1:4]:
            c = alt['config']
            print(f"  TP={c['tp']}, PP={c['pp']}, DP={c['dp']} - Score: {alt['score']:.2f}")
        print("=" * 60)


def cmd_inference(args):
    """Handle inference subcommand."""
    from .inference_optimization import get_inference_optimization_report
    from .model_analyzer import ModelAnalyzer
    from .advisor import ParallelismAdvisor, create_mock_topology_8xb200
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    advisor = ParallelismAdvisor(auto_detect_topology=False)
    try:
        advisor.detect_topology()
        topology = advisor.topology
    except RuntimeError:
        topology = create_mock_topology_8xb200()
    
    gpu_memory_gb = topology.gpus[0].memory_gb if topology.gpus else topology.total_memory_gb / max(1, topology.num_gpus)
    gpu_arch = topology.gpus[0].architecture if topology.gpus else 'hopper'
    
    model_config = {
        "name": args.model,
        "parameters_billions": arch.total_params_billion,
        "num_layers": arch.num_layers,
        "hidden_size": arch.hidden_size,
        "num_kv_heads": getattr(arch, 'num_kv_heads', 8),
        "num_attention_heads": getattr(arch, 'num_attention_heads', 64),
        "max_sequence_length": 4096,
    }
    
    hardware_config = {
        "gpu_arch": gpu_arch.lower() if gpu_arch else 'hopper',
        "gpu_memory_gb": gpu_memory_gb,
        "num_gpus": topology.num_gpus,
    }
    
    result = get_inference_optimization_report(model_config, hardware_config, args.goal)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 70)
        print("INFERENCE OPTIMIZATION REPORT")
        print("=" * 70)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"Goal: {args.goal.upper()}")
        
        print(f"\n{'─' * 70}")
        print("QUANTIZATION")
        print(f"{'─' * 70}")
        quant = result['quantization']
        print(f"  Method: {quant['method'].upper()}")
        print(f"  Bits: {quant['bits']}")
        print(f"  Memory Reduction: {quant['memory_reduction_pct']}%")
        print(f"  Throughput Change: {quant['throughput_change_pct']:+}%")
        print(f"  Accuracy Impact: {quant['accuracy_impact']}")
        
        print(f"\n{'─' * 70}")
        print("KV CACHE")
        print(f"{'─' * 70}")
        kv = result['kv_cache']
        print(f"  Paged Attention: {'✓' if kv['paged_attention'] else '✗'}")
        print(f"  KV Cache Dtype: {kv['kv_cache_dtype']}")
        print(f"  Max Sequences: {kv['max_num_seqs']}")
        print(f"  KV Cache Size: {kv['memory']['total_kv_cache_gb']:.1f} GB")
        
        if result.get('speculative_decoding') and result['speculative_decoding']['enabled']:
            print(f"\n{'─' * 70}")
            print("SPECULATIVE DECODING")
            print(f"{'─' * 70}")
            spec = result['speculative_decoding']
            print(f"  Method: {spec['method']}")
            if spec['draft_model']:
                print(f"  Draft Model: {spec['draft_model']}")
            print(f"  Expected Speedup: {spec['expected_speedup']:.1f}x")
        
        print(f"\n{'─' * 70}")
        print("RECOMMENDED ENGINE")
        print(f"{'─' * 70}")
        engine = result['engine']
        print(f"  Engine: {engine['engine']}")
        print(f"  Expected Throughput: {engine['performance']['throughput_tps']:.0f} tokens/s")
        print(f"  Expected Latency: {engine['performance']['latency_ms']:.0f} ms")
        print(f"\n  Launch Command:")
        for line in engine['launch_command'].split('\n'):
            print(f"    {line}")
        
        print(f"\n{'─' * 70}")
        print("SUMMARY")
        print(f"{'─' * 70}")
        summary = result['summary']
        print(f"  Original Memory: {summary['original_memory_gb']:.1f} GB")
        print(f"  Optimized Memory: {summary['optimized_memory_gb']:.1f} GB")
        print(f"  Memory Reduction: {summary['memory_reduction_pct']:.0f}%")
        print(f"  Expected Speedup: {summary['expected_speedup']:.1f}x")
        print("=" * 70)


def cmd_troubleshoot(args):
    """Handle troubleshoot subcommand."""
    from .troubleshooting import (
        diagnose_error, get_nccl_tuning, get_all_troubleshooting_topics
    )
    
    if args.list:
        topics = get_all_troubleshooting_topics()
        if args.json:
            print(json.dumps(topics, indent=2))
        else:
            print("=" * 60)
            print("KNOWN DISTRIBUTED TRAINING ISSUES")
            print("=" * 60)
            for topic in topics:
                icon = "🔴" if topic['severity'] == 'critical' else "🟡"
                print(f"\n{icon} {topic['title']} [{topic['category']}]")
                print(f"   {topic['description']}")
            print("=" * 60)
        return
    
    if args.nccl:
        result = get_nccl_tuning(args.interconnect, debug=True)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("=" * 60)
            print(f"NCCL TUNING FOR {args.interconnect.upper()}")
            print("=" * 60)
            print("\nEnvironment Variables:")
            for cmd in result['export_commands']:
                print(f"  {cmd}")
            print("\n# Add to your job script or ~/.bashrc")
            print("=" * 60)
        return
    
    if args.error:
        result = diagnose_error(error_message=args.error)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("=" * 60)
            print("DIAGNOSIS RESULTS")
            print("=" * 60)
            if result['issues_found'] == 0:
                print("\nNo matching issues found. Try --list to see all known issues.")
            else:
                for issue in result['issues']:
                    print(f"\n🔴 {issue['title']}")
                    print(f"   Category: {issue['category']}")
                    print(f"\n   Root Causes:")
                    for cause in issue['root_causes'][:3]:
                        print(f"     • {cause}")
                    print(f"\n   Solutions:")
                    for sol in issue['solutions'][:5]:
                        print(f"     {sol}")
                    if issue.get('env_vars'):
                        print(f"\n   Environment Variables:")
                        for k, v in issue['env_vars'].items():
                            print(f"     export {k}={v}")
            print("=" * 60)
        return
    
    # Default: show help
    print("Use --error 'message' to diagnose, --list to see all issues, or --nccl for NCCL tuning")


def cmd_memory(args):
    """Handle memory subcommand."""
    from .troubleshooting import get_memory_breakdown
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    model_config = {
        "parameters_billions": arch.total_params_billion,
        "hidden_size": arch.hidden_size,
        "num_layers": arch.num_layers,
        "max_sequence_length": args.seq_length,
    }
    
    hardware_config = {"gpu_memory_gb": 80}
    parallelism_config = {
        "tensor_parallel": args.tp,
        "pipeline_parallel": args.pp,
        "data_parallel": args.dp,
    }
    training_config = {
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "gradient_checkpointing": not args.no_checkpointing,
        "is_training": True,
    }
    
    result = get_memory_breakdown(model_config, hardware_config, parallelism_config, training_config)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("MEMORY BREAKDOWN ANALYSIS")
        print("=" * 60)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"Configuration: TP={args.tp}, PP={args.pp}, DP={args.dp}")
        print(f"Batch Size: {args.batch_size}, Seq Length: {args.seq_length}")
        
        print(f"\n{'─' * 60}")
        print("MEMORY USAGE PER GPU")
        print(f"{'─' * 60}")
        breakdown = result['breakdown']
        total = result['total_memory_gb']
        
        components = [
            ("Parameters", breakdown['parameters']),
            ("Gradients", breakdown['gradients']),
            ("Optimizer States", breakdown['optimizer_states']),
            ("Activations", breakdown['activations']),
            ("Workspace", breakdown['workspace']),
            ("Fragmentation", breakdown['fragmentation']),
        ]
        
        for name, value in components:
            pct = value / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {name:<20} {value:>8.1f} GB  {bar} {pct:>5.1f}%")
        
        print(f"  {'─' * 50}")
        print(f"  {'TOTAL':<20} {total:>8.1f} GB")
        
        if result['savings_opportunities']:
            print(f"\n{'─' * 60}")
            print("OPTIMIZATION OPPORTUNITIES")
            print(f"{'─' * 60}")
            for opp in result['savings_opportunities']:
                print(f"\n  💡 {opp['technique']}")
                print(f"     Potential Savings: {opp['savings_gb']:.1f} GB")
                print(f"     Difficulty: {opp['difficulty']}")
                print(f"     Impact: {opp['impact']}")
        
        print("=" * 60)


def cmd_nccl(args):
    """Handle nccl subcommand."""
    from .distributed_training import NCCLTuningAdvisor
    
    advisor = NCCLTuningAdvisor()
    
    if args.diagnose:
        result = advisor.diagnose_issues()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("=" * 70)
            print("NCCL DIAGNOSTICS")
            print("=" * 70)
            print(f"\nDetected Backend: {result['backend']}")
            if result['issues']:
                print("\n⚠️  Issues Found:")
                for issue in result['issues']:
                    print(f"  • {issue}")
            else:
                print("\n✅ No issues detected")
            if result['recommendations']:
                print("\n📋 Recommendations:")
                for rec in result['recommendations']:
                    print(f"  • {rec}")
            print("=" * 70)
        return
    
    config = advisor.get_optimal_config(
        num_nodes=args.nodes,
        gpus_per_node=args.gpus,
        model_size_b=args.model_size,
        tp_size=args.tp,
        pp_size=args.pp,
    )
    
    if args.json:
        print(json.dumps({
            "env_vars": config.env_vars,
            "description": config.description,
            "expected_bandwidth_gb_s": config.expected_bandwidth_gb_s,
            "warnings": config.warnings,
            "optimizations": config.optimizations,
        }, indent=2))
    else:
        print("=" * 70)
        print("NCCL TUNING RECOMMENDATION")
        print("=" * 70)
        print(config.description)
        print("\nEnvironment Variables:")
        for k, v in config.env_vars.items():
            print(f"  export {k}={v}")
        if config.optimizations:
            print("\n✅ Optimizations Applied:")
            for opt in config.optimizations:
                print(f"  • {opt}")
        if config.warnings:
            print("\n⚠️  Warnings:")
            for warn in config.warnings:
                print(f"  • {warn}")
        print("=" * 70)


def cmd_rlhf(args):
    """Handle rlhf subcommand."""
    from .distributed_training import RLHFMemoryCalculator, RLHFAlgorithm
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    calculator = RLHFMemoryCalculator()
    
    if args.compare:
        result = calculator.get_optimal_config(
            arch.total_params_billion,
            num_gpus=8,
            gpu_memory_gb=args.memory,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("=" * 70)
            print(f"RLHF ALGORITHM COMPARISON - {args.model}")
            print("=" * 70)
            print(f"\nModel: {arch.total_params_billion:.1f}B parameters")
            print(f"\n{'Algorithm':<12} {'Memory (GB)':<15} {'Fits 1 GPU?':<15} {'Rec. TP':<10}")
            print("-" * 52)
            for algo, data in result['algorithms'].items():
                fits = "✅" if data['fits_single_gpu'] else "❌"
                print(f"{algo.upper():<12} {data['total_memory_gb']:<15.1f} {fits:<15} {data['recommended_tp']:<10}")
            print("-" * 52)
            print(f"\n✅ Recommended: {result['recommended'].upper()}")
            print(f"   {result['recommendation_reason']}")
            print("=" * 70)
        return
    
    algo_map = {
        "ppo": RLHFAlgorithm.PPO,
        "dpo": RLHFAlgorithm.DPO,
        "orpo": RLHFAlgorithm.ORPO,
        "kto": RLHFAlgorithm.KTO,
        "grpo": RLHFAlgorithm.GRPO,
    }
    
    estimate = calculator.calculate(
        arch.total_params_billion,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        algorithm=algo_map.get(args.algorithm, RLHFAlgorithm.PPO),
        gpu_memory_gb=args.memory,
    )
    
    if args.json:
        print(json.dumps({
            "algorithm": estimate.algorithm.value,
            "actor_memory_gb": estimate.actor_memory_gb,
            "critic_memory_gb": estimate.critic_memory_gb,
            "reference_memory_gb": estimate.reference_memory_gb,
            "reward_memory_gb": estimate.reward_memory_gb,
            "optimizer_memory_gb": estimate.optimizer_memory_gb,
            "activation_memory_gb": estimate.activation_memory_gb,
            "total_memory_gb": estimate.total_memory_gb,
            "fits_single_gpu": estimate.fits_single_gpu,
            "recommended_tp": estimate.recommended_tp,
            "optimizations": estimate.optimizations,
        }, indent=2))
    else:
        print("=" * 70)
        print(f"RLHF MEMORY ESTIMATE - {args.algorithm.upper()}")
        print("=" * 70)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"Algorithm: {args.algorithm.upper()}")
        print(f"\nMemory Breakdown:")
        print(f"  Actor Model:      {estimate.actor_memory_gb:>8.1f} GB")
        print(f"  Critic Model:     {estimate.critic_memory_gb:>8.1f} GB")
        print(f"  Reference Model:  {estimate.reference_memory_gb:>8.1f} GB")
        print(f"  Reward Model:     {estimate.reward_memory_gb:>8.1f} GB")
        print(f"  Optimizer States: {estimate.optimizer_memory_gb:>8.1f} GB")
        print(f"  Activations:      {estimate.activation_memory_gb:>8.1f} GB")
        print(f"  {'─' * 30}")
        print(f"  TOTAL:            {estimate.total_memory_gb:>8.1f} GB")
        
        fits = "✅ Yes" if estimate.fits_single_gpu else "❌ No"
        print(f"\nFits single {args.memory}GB GPU: {fits}")
        print(f"Recommended TP: {estimate.recommended_tp}")
        
        if estimate.optimizations:
            print("\n📋 Optimization Recommendations:")
            for opt in estimate.optimizations:
                print(f"  • {opt}")
        print("=" * 70)


def cmd_moe(args):
    """Handle moe subcommand."""
    from .distributed_training import MoEOptimizer
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    optimizer = MoEOptimizer()
    config = optimizer.optimize(
        arch.total_params_billion,
        num_experts=args.num_experts,
        num_gpus=args.gpus,
        gpu_memory_gb=args.memory,
        batch_size=args.batch_size,
    )
    
    if args.json:
        print(json.dumps({
            "num_experts": config.num_experts,
            "experts_per_rank": config.experts_per_rank,
            "expert_parallel_size": config.expert_parallel_size,
            "tensor_parallel_size": config.tensor_parallel_size,
            "data_parallel_size": config.data_parallel_size,
            "capacity_factor": config.capacity_factor,
            "load_balancing_loss_weight": config.load_balancing_loss_weight,
            "memory_per_gpu_gb": config.memory_per_gpu_gb,
            "communication_volume_gb": config.communication_volume_gb,
            "expected_mfu": config.expected_mfu,
            "warnings": config.warnings,
        }, indent=2))
    else:
        print("=" * 70)
        print(f"MOE PARALLELISM CONFIGURATION")
        print("=" * 70)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"Experts: {config.num_experts}")
        print(f"\nParallelism Strategy:")
        print(f"  Expert Parallel (EP): {config.expert_parallel_size}")
        print(f"  Tensor Parallel (TP): {config.tensor_parallel_size}")
        print(f"  Data Parallel (DP):   {config.data_parallel_size}")
        print(f"  Experts per GPU:      {config.experts_per_rank}")
        print(f"\nMemory & Performance:")
        print(f"  Memory per GPU: {config.memory_per_gpu_gb:.1f} GB")
        print(f"  All-to-All Volume: {config.communication_volume_gb:.2f} GB")
        print(f"  Expected MFU: {config.expected_mfu:.1%}")
        print(f"\nHyperparameters:")
        print(f"  Capacity Factor: {config.capacity_factor}")
        print(f"  Load Balance Loss: {config.load_balancing_loss_weight}")
        if config.warnings:
            print("\n⚠️  Warnings:")
            for warn in config.warnings:
                print(f"  • {warn}")
        print("=" * 70)


def cmd_long_context(args):
    """Handle long-context subcommand."""
    from .distributed_training import LongContextOptimizer
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    optimizer = LongContextOptimizer()
    config = optimizer.optimize(
        arch.total_params_billion,
        target_seq_length=args.seq_length,
        num_gpus=args.gpus,
        gpu_memory_gb=args.memory,
        method=args.method,
    )
    
    if args.json:
        print(json.dumps({
            "method": config.method,
            "sequence_length": config.sequence_length,
            "context_parallel_size": config.context_parallel_size,
            "ring_attention_heads": config.ring_attention_heads,
            "memory_savings_pct": config.memory_savings_pct,
            "communication_overhead_pct": config.communication_overhead_pct,
            "expected_throughput_tokens_s": config.expected_throughput_tokens_s,
            "launch_args": config.launch_args,
        }, indent=2))
    else:
        print("=" * 70)
        print(f"LONG CONTEXT OPTIMIZATION")
        print("=" * 70)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"Target Sequence Length: {config.sequence_length:,} tokens")
        print(f"\nMethod: {config.method.upper().replace('_', ' ')}")
        if config.context_parallel_size > 1:
            print(f"Context Parallel Size: {config.context_parallel_size}")
        if config.ring_attention_heads > 0:
            print(f"Ring Attention Heads: {config.ring_attention_heads}")
        print(f"\nPerformance:")
        print(f"  Memory Savings: {config.memory_savings_pct:.0f}%")
        print(f"  Communication Overhead: {config.communication_overhead_pct:.0f}%")
        print(f"  Expected Throughput: {config.expected_throughput_tokens_s:.0f} tokens/s")
        print(f"\nLaunch Arguments:")
        for k, v in config.launch_args.items():
            print(f"  --{k.replace('_', '-')} {v}")
        print("=" * 70)


def cmd_vllm(args):
    """Handle vllm subcommand."""
    from .distributed_training import VLLMConfigGenerator
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    generator = VLLMConfigGenerator()
    
    if args.compare_engines:
        result = generator.compare_engines(args.model, arch.total_params_billion, args.gpus)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("=" * 70)
            print("INFERENCE ENGINE COMPARISON")
            print("=" * 70)
            for name, info in result['engines'].items():
                print(f"\n{name.upper()}")
                print(f"  Best for: {info['best_for']}")
                print(f"  Strengths: {', '.join(info['strengths'])}")
                print(f"  Weaknesses: {', '.join(info['weaknesses'])}")
                print(f"  Setup: {info['setup']}")
            print(f"\n✅ Recommended: {result['recommended'].upper()}")
            print(f"   {result['reason']}")
            print("=" * 70)
        return
    
    quant = None if args.quantization == "none" else args.quantization
    
    config = generator.generate(
        model=args.model,
        model_params_b=arch.total_params_billion,
        num_gpus=args.gpus,
        gpu_memory_gb=args.memory,
        target=args.target,
        max_seq_length=args.max_seq_length,
        quantization=quant,
    )
    
    if args.json:
        print(json.dumps({
            "model": config.model,
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_model_len": config.max_model_len,
            "max_num_seqs": config.max_num_seqs,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "quantization": config.quantization,
            "kv_cache_dtype": config.kv_cache_dtype,
            "enforce_eager": config.enforce_eager,
            "enable_chunked_prefill": config.enable_chunked_prefill,
            "enable_prefix_caching": config.enable_prefix_caching,
            "speculative_model": config.speculative_model,
            "speculative_num_draft_tokens": config.speculative_num_draft_tokens,
            "launch_command": config.launch_command,
            "estimated_throughput_tokens_s": config.estimated_throughput_tokens_s,
            "estimated_latency_ms": config.estimated_latency_ms,
        }, indent=2))
    else:
        print("=" * 70)
        print(f"VLLM CONFIGURATION - {args.target.upper()} OPTIMIZED")
        print("=" * 70)
        print(f"\nModel: {config.model}")
        print(f"\nParallelism:")
        print(f"  Tensor Parallel: {config.tensor_parallel_size}")
        print(f"  Pipeline Parallel: {config.pipeline_parallel_size}")
        print(f"\nBatching:")
        print(f"  Max Sequences: {config.max_num_seqs}")
        print(f"  Max Batched Tokens: {config.max_num_batched_tokens}")
        print(f"  Max Model Length: {config.max_model_len}")
        print(f"\nOptimizations:")
        print(f"  GPU Memory Utilization: {config.gpu_memory_utilization:.0%}")
        print(f"  Quantization: {config.quantization or 'None'}")
        print(f"  KV Cache Dtype: {config.kv_cache_dtype}")
        print(f"  Chunked Prefill: {'✓' if config.enable_chunked_prefill else '✗'}")
        print(f"  Prefix Caching: {'✓' if config.enable_prefix_caching else '✗'}")
        if config.speculative_model:
            print(f"\nSpeculative Decoding:")
            print(f"  Draft Model: {config.speculative_model}")
            print(f"  Draft Tokens: {config.speculative_num_draft_tokens}")
        print(f"\nExpected Performance:")
        print(f"  Throughput: ~{config.estimated_throughput_tokens_s:,} tokens/s")
        print(f"  Latency: ~{config.estimated_latency_ms:.0f} ms")
        print(f"\n{'─' * 70}")
        print("LAUNCH COMMAND:")
        print(f"{'─' * 70}")
        print(config.launch_command)
        print("=" * 70)


def cmd_comm_overlap(args):
    """Handle comm-overlap subcommand."""
    from .distributed_training import CommunicationOverlapAnalyzer
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    overlap_analyzer = CommunicationOverlapAnalyzer()
    result = overlap_analyzer.analyze(
        arch.total_params_billion,
        tp_size=args.tp,
        pp_size=args.pp,
        dp_size=args.dp,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
    )
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 70)
        print("COMMUNICATION OVERLAP ANALYSIS")
        print("=" * 70)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"Parallelism: TP={args.tp}, PP={args.pp}, DP={args.dp}")
        print(f"\nOverlap Opportunities:")
        for opp in result['opportunities']:
            print(f"\n  {opp['type']}:")
            print(f"    Volume: {opp['volume_mb']:.1f} MB")
            print(f"    Potential: {opp['overlap_potential']}")
        print(f"\nRecommendations:")
        for rec in result['recommendations']:
            print(f"  • {rec}")
        print(f"\nEnvironment Variables:")
        for k, v in result['overlap_configs'].items():
            print(f"  export {k}={v}")
        print("=" * 70)


def cmd_export(args):
    """Handle export subcommand."""
    from .config_export import export_training_config, ConfigExporter
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    model_config = {
        "name": args.model,
        "parameters_billions": arch.total_params_billion,
        "max_sequence_length": 4096,
    }
    
    hardware_config = {
        "num_nodes": args.nodes,
        "gpus_per_node": args.gpus,
    }
    
    parallelism_config = {
        "tensor_parallel": args.tp,
        "pipeline_parallel": args.pp,
        "data_parallel": args.dp,
    }
    
    training_config = {
        "batch_size": args.batch_size,
        "zero_stage": args.zero_stage,
    }
    
    result = export_training_config(model_config, hardware_config, parallelism_config, training_config)
    
    if args.output_dir:
        exporter = ConfigExporter()
        from .config_export import ExportedConfig
        config = ExportedConfig(**result)
        files = exporter.export_to_files(config, args.output_dir)
        print(f"Configuration exported to {args.output_dir}:")
        for name, path in files.items():
            print(f"  • {name}: {path}")
    elif args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 70)
        print("EXPORTED TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"\n{result['description']}")
        
        print(f"\n{'─' * 70}")
        print("DEEPSPEED CONFIG (ds_config.json)")
        print(f"{'─' * 70}")
        print(json.dumps(result['deepspeed_config'], indent=2))
        
        print(f"\n{'─' * 70}")
        print("ENVIRONMENT VARIABLES (env.sh)")
        print(f"{'─' * 70}")
        for k, v in result['environment_variables'].items():
            print(f'export {k}="{v}"')
        
        print(f"\n{'─' * 70}")
        print("LAUNCH COMMAND")
        print(f"{'─' * 70}")
        print(result['torchrun_command'])
        
        print(f"\n{'─' * 70}")
        print("Use --output-dir to export files, or --json for full output")
        print("=" * 70)


def cmd_rl(args):
    """Handle rl subcommand."""
    from .rl_optimization import get_rl_optimization, RLAlgorithm
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    model_config = {
        "name": args.model,
        "parameters_billions": arch.total_params_billion,
    }
    
    hardware_config = {
        "gpu_memory_gb": args.gpu_memory,
        "num_gpus": args.gpus,
    }
    
    rl_config = {
        "algorithm": args.algorithm,
        "use_peft": args.use_peft,
        "peft_rank": args.peft_rank,
    }
    
    result = get_rl_optimization(model_config, hardware_config, rl_config)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 70)
        print(f"RL OPTIMIZATION: {args.algorithm.upper()}")
        print("=" * 70)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"Algorithm: {args.algorithm.upper()}")
        print(f"GPUs: {args.gpus}")
        
        mp = result['memory_plan']
        print(f"\n{'─' * 70}")
        print("MEMORY PLAN")
        print(f"{'─' * 70}")
        print(f"  Total GPU memory needed: {mp['total_gpu_memory_needed_gb']:.1f} GB")
        if mp['total_cpu_memory_needed_gb'] > 0:
            print(f"  CPU memory needed: {mp['total_cpu_memory_needed_gb']:.1f} GB")
        
        for model_name, model_info in mp['models'].items():
            if model_info:
                print(f"\n  {model_name.capitalize()}:")
                print(f"    • Strategy: {model_info['strategy']}")
                print(f"    • Device: {model_info['device']}")
                print(f"    • Memory: {model_info['memory_gb']:.1f} GB")
        
        tc = result['training_config']
        print(f"\n{'─' * 70}")
        print("TRAINING CONFIG")
        print(f"{'─' * 70}")
        p = tc['parallelism']
        print(f"  Parallelism: TP={p['actor_tp']}, PP={p['actor_pp']}, DP={p['dp']}")
        print(f"  Reference Strategy: {tc['reference_strategy']}")
        print(f"  PPO Epochs: {tc['ppo_epochs']}")
        
        print(f"\n{'─' * 70}")
        print("LAUNCH COMMAND")
        print(f"{'─' * 70}")
        print(tc['trl_command'])
        
        if mp.get('recommendations'):
            print(f"\n{'─' * 70}")
            print("RECOMMENDATIONS")
            print(f"{'─' * 70}")
            for rec in mp['recommendations']:
                print(f"  • {rec}")
        
        if mp.get('warnings'):
            print(f"\n⚠ WARNINGS:")
            for warn in mp['warnings']:
                print(f"  • {warn}")
        
        print("=" * 70)


def cmd_vllm(args):
    """Handle vllm subcommand."""
    from .vllm_optimization import get_vllm_optimization
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    model_config = {
        "name": args.model,
        "parameters_billions": arch.total_params_billion,
        "max_sequence_length": args.max_seq_length,
    }
    
    hardware_config = {
        "gpu_memory_gb": args.memory,
        "num_gpus": args.gpus,
        "gpu_arch": "hopper",  # Default to H100
    }
    
    sla_config = None
    if args.target == "sla":
        sla_config = {
            "max_ttft_ms": getattr(args, 'ttft_ms', 500),
            "max_itl_ms": getattr(args, 'itl_ms', 50),
        }
    
    result = get_vllm_optimization(model_config, hardware_config, args.target, sla_config)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 70)
        print(f"vLLM OPTIMIZATION: {args.target.upper()}")
        print("=" * 70)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"GPUs: {args.gpus}")
        print(f"Max Sequence Length: {args.max_seq_length}")
        
        cfg = result['config']
        print(f"\n{'─' * 70}")
        print("vLLM CONFIGURATION")
        print(f"{'─' * 70}")
        print(f"  Tensor Parallel: {cfg['tensor_parallel_size']}")
        print(f"  GPU Memory Util: {cfg['gpu_memory_utilization']:.0%}")
        print(f"  KV Cache Dtype: {cfg['kv_cache_dtype']}")
        print(f"  Prefix Caching: {cfg['enable_prefix_caching']}")
        print(f"  Chunked Prefill: {cfg['enable_chunked_prefill']}")
        print(f"  Max Concurrent: {cfg['max_num_seqs']}")
        if cfg.get('speculative_model'):
            print(f"  Speculative Model: {cfg['speculative_model']}")
        if cfg.get('quantization'):
            print(f"  Quantization: {cfg['quantization']}")
        
        perf = result['expected_performance']
        print(f"\n{'─' * 70}")
        print("EXPECTED PERFORMANCE")
        print(f"{'─' * 70}")
        print(f"  Throughput: {perf['throughput_tokens_per_second']:,.0f} tokens/s")
        print(f"  Time to First Token: {perf['time_to_first_token_ms']:.0f} ms")
        print(f"  Inter-Token Latency: {perf['inter_token_latency_ms']:.1f} ms")
        
        print(f"\n{'─' * 70}")
        print("LAUNCH COMMAND")
        print(f"{'─' * 70}")
        print(result['launch_command'])
        
        if result.get('recommendations'):
            print(f"\n{'─' * 70}")
            print("RECOMMENDATIONS")
            print(f"{'─' * 70}")
            for rec in result['recommendations']:
                print(f"  • {rec}")
        
        print("=" * 70)


def cmd_llm_advisor(args):
    """Handle llm-advisor subcommand."""
    from .llm_advisor import LLMOptimizationAdvisor, SystemContext, OptimizationRequest, OptimizationGoal
    from .model_analyzer import ModelAnalyzer
    
    advisor = LLMOptimizationAdvisor(llm_provider=args.provider)
    
    # Analyze model
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    # Build context
    context = SystemContext(
        model_name=args.model,
        model_params_b=arch.total_params_billion,
        num_layers=arch.num_layers,
        hidden_size=arch.hidden_size,
        gpu_count=args.gpus,
        is_training=not args.inference,
    )
    
    goal_map = {
        "throughput": OptimizationGoal.THROUGHPUT,
        "latency": OptimizationGoal.LATENCY,
        "memory": OptimizationGoal.MEMORY,
        "efficiency": OptimizationGoal.EFFICIENCY,
        "cost": OptimizationGoal.COST,
    }
    
    request = OptimizationRequest(
        context=context,
        goal=goal_map.get(args.goal, OptimizationGoal.THROUGHPUT),
    )
    
    advice = advisor.get_advice(request)
    
    if args.json:
        print(json.dumps({
            "summary": advice.summary,
            "recommendations": advice.priority_recommendations,
            "parallelism": advice.parallelism_changes,
            "compound_strategies": advice.compound_strategies,
            "launch_command": advice.launch_command,
            "expected_improvements": advice.expected_improvements,
            "warnings": advice.warnings,
        }, indent=2))
    else:
        print(advisor._format_advice(advice))


def cmd_nccl(args):
    """Handle nccl subcommand."""
    from .troubleshooting import get_nccl_tuning
    
    result = get_nccl_tuning(
        interconnect="infiniband" if args.nodes > 1 else "nvlink",
        debug=args.diagnose,
    )
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("NCCL TUNING RECOMMENDATIONS")
        print("=" * 60)
        print(f"\nCluster: {args.nodes} nodes × {args.gpus} GPUs")
        print(f"Model Size: {args.model_size}B, TP={args.tp}, PP={args.pp}")
        
        print(f"\n{'─' * 60}")
        print("RECOMMENDED ENVIRONMENT VARIABLES")
        print(f"{'─' * 60}")
        for cmd in result['export_commands']:
            print(f"  {cmd}")
        
        print(f"\n{'─' * 60}")
        print("Add to your job script or ~/.bashrc")
        print("=" * 60)


def cmd_rlhf(args):
    """Handle rlhf subcommand - redirects to rl command."""
    # This is an alias for the rl command with RLHF focus
    from .rl_optimization import get_rl_optimization
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    model_config = {
        "name": args.model,
        "parameters_billions": arch.total_params_billion,
    }
    
    hardware_config = {
        "gpu_memory_gb": args.memory,
        "num_gpus": 8,  # Default
    }
    
    rl_config = {
        "algorithm": args.algorithm,
        "use_peft": True,  # Recommend PEFT for RLHF
    }
    
    result = get_rl_optimization(model_config, hardware_config, rl_config)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print(f"RLHF OPTIMIZATION: {args.algorithm.upper()}")
        print("=" * 60)
        mp = result['memory_plan']
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"GPU Memory Needed: {mp['total_gpu_memory_needed_gb']:.1f} GB")
        
        if args.compare:
            print(f"\n{'─' * 60}")
            print("ALGORITHM COMPARISON")
            print(f"{'─' * 60}")
            for algo in result['comparison']:
                print(f"\n  {algo['algorithm']}:")
                print(f"    Memory: {algo['memory_multiplier']}")
                print(f"    Complexity: {algo['complexity']}")
                print(f"    Best for: {algo['best_for']}")
        
        print("=" * 60)


def cmd_moe(args):
    """Handle moe subcommand."""
    from .model_analyzer import ModelAnalyzer
    from .strategy_optimizer import StrategyOptimizer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    num_experts = args.num_experts or arch.num_experts or 8
    
    if args.json:
        result = {
            "model": args.model,
            "num_experts": num_experts,
            "recommendations": {
                "expert_parallel": min(num_experts, args.gpus),
                "tensor_parallel": min(8, args.gpus // min(num_experts, args.gpus)),
            },
        }
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("MoE PARALLELISM OPTIMIZATION")
        print("=" * 60)
        print(f"\nModel: {args.model}")
        print(f"Experts: {num_experts}")
        print(f"GPUs: {args.gpus}")
        
        ep = min(num_experts, args.gpus)
        tp = min(8, args.gpus // ep)
        
        print(f"\n{'─' * 60}")
        print("RECOMMENDED PARALLELISM")
        print(f"{'─' * 60}")
        print(f"  Expert Parallel (EP): {ep}")
        print(f"  Tensor Parallel (TP): {tp}")
        print(f"  Data Parallel (DP): {args.gpus // (ep * tp)}")
        
        print(f"\n{'─' * 60}")
        print("MoE TIPS")
        print(f"{'─' * 60}")
        print("  • EP should divide num_experts evenly")
        print("  • Use capacity factor 1.25 for load balancing")
        print("  • Enable expert load balancing loss")
        print("  • Consider all-to-all communication costs")
        print("=" * 60)


def cmd_long_context(args):
    """Handle long-context subcommand."""
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    # Calculate context parallel needs
    base_kv_memory = 2 * arch.num_layers * arch.hidden_size * args.seq_length * 2 / 1e9
    cp_needed = max(1, int(base_kv_memory / (args.memory * 0.3)))
    
    if args.json:
        result = {
            "model": args.model,
            "sequence_length": args.seq_length,
            "kv_cache_memory_gb": base_kv_memory,
            "recommendations": {
                "context_parallel": min(cp_needed, args.gpus),
                "method": args.method if args.method != "auto" else "context_parallel" if cp_needed > 1 else "flash_attention",
            },
        }
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("LONG CONTEXT OPTIMIZATION")
        print("=" * 60)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"Target Sequence Length: {args.seq_length:,}")
        print(f"Estimated KV Cache: {base_kv_memory:.1f} GB")
        
        print(f"\n{'─' * 60}")
        print("RECOMMENDATIONS")
        print(f"{'─' * 60}")
        
        method = args.method if args.method != "auto" else ("context_parallel" if cp_needed > 1 else "flash_attention_v3")
        
        if method == "context_parallel":
            print(f"  Method: Context Parallel (CP)")
            print(f"  CP Size: {min(cp_needed, args.gpus)}")
            print(f"  Splits sequence across GPUs for KV cache")
        elif method == "ring_attention":
            print(f"  Method: Ring Attention")
            print(f"  Ring Size: {args.gpus}")
            print(f"  Overlaps communication with computation")
        else:
            print(f"  Method: Flash Attention v3")
            print(f"  Single GPU with memory-efficient attention")
        
        print(f"\n{'─' * 60}")
        print("LONG CONTEXT TIPS")
        print(f"{'─' * 60}")
        print("  • Use RoPE scaling for extended context")
        print("  • Enable FP8 KV cache for 2x memory savings")
        print("  • Consider chunked prefill for long prompts")
        print("=" * 60)


def cmd_comm_overlap(args):
    """Handle comm-overlap subcommand."""
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    # Calculate overlap opportunities
    compute_time_ms = args.batch_size * args.seq_length * arch.total_params_billion * 6 / 1000  # Rough estimate
    comm_time_ms = arch.total_params_billion * 2 / args.tp / 100  # Rough estimate
    
    overlap_potential = min(1.0, comm_time_ms / compute_time_ms)
    
    if args.json:
        result = {
            "model": args.model,
            "parallelism": {"tp": args.tp, "pp": args.pp, "dp": args.dp},
            "overlap_analysis": {
                "compute_time_ms": compute_time_ms,
                "communication_time_ms": comm_time_ms,
                "overlap_potential": overlap_potential,
            },
        }
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("COMMUNICATION OVERLAP ANALYSIS")
        print("=" * 60)
        print(f"\nModel: {args.model}")
        print(f"Parallelism: TP={args.tp}, PP={args.pp}, DP={args.dp}")
        
        print(f"\n{'─' * 60}")
        print("OVERLAP OPPORTUNITIES")
        print(f"{'─' * 60}")
        
        if args.dp > 1:
            print("  ✓ Gradient All-Reduce Overlap")
            print("    • Enable: --overlap-grad-reduce")
            print("    • Overlap backward computation with gradient sync")
        
        if args.tp > 1:
            print("  ✓ Tensor Parallel All-Reduce Overlap")
            print("    • Use CUDA streams for async all-reduce")
            print("    • Requires NVLink for best performance")
        
        if args.pp > 1:
            print("  ✓ Pipeline Send/Recv Overlap")
            print("    • Use async P2P operations")
            print("    • Enable: --overlap-p2p")
        
        print(f"\n  Overlap Potential: {overlap_potential:.1%}")
        print("=" * 60)


def cmd_largescale(args):
    """Handle large-scale subcommand."""
    from .large_scale_optimization import get_large_scale_optimization
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze(args.model)
    
    model_config = {
        "name": args.model,
        "parameters_billions": arch.total_params_billion,
        "max_sequence_length": args.seq_length,
        "num_experts": getattr(arch, 'num_experts', 1) or 1,
    }
    
    cluster_config = {
        "num_nodes": args.nodes,
        "gpus_per_node": args.gpus_per_node,
        "gpu_memory_gb": args.gpu_memory,
        "network_type": args.network,
        "inter_node_bandwidth_gbps": 400 if args.network == "infiniband" else 100,
        "intra_node_bandwidth_gbps": 900,
    }
    
    training_config = {
        "global_batch_size": args.batch_size,
    }
    
    result = get_large_scale_optimization(model_config, cluster_config, training_config)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 70)
        print("LARGE-SCALE CLUSTER OPTIMIZATION")
        print("=" * 70)
        print(f"\nModel: {args.model} ({arch.total_params_billion:.1f}B)")
        print(f"Cluster: {args.nodes} nodes × {args.gpus_per_node} GPUs = {args.nodes * args.gpus_per_node} total GPUs")
        print(f"Network: {args.network}")
        
        cfg = result['config']
        p = cfg['parallelism']
        print(f"\n{'─' * 70}")
        print("PARALLELISM CONFIGURATION")
        print(f"{'─' * 70}")
        print(f"  3D Parallelism: TP={p['tp']} × PP={p['pp']} × DP={p['dp']}")
        if p.get('cp', 1) > 1:
            print(f"  Context Parallel: {p['cp']}")
        if p.get('ep', 1) > 1:
            print(f"  Expert Parallel: {p['ep']}")
        
        m = cfg['mapping']
        print(f"\n  Mapping:")
        print(f"    • TP within node: {m['tp_within_node']}")
        print(f"    • PP across nodes: {m['pp_across_nodes']}")
        print(f"    • Hierarchical DP: {m['dp_hierarchical']}")
        
        ft = cfg['fault_tolerance']
        print(f"\n{'─' * 70}")
        print("FAULT TOLERANCE")
        print(f"{'─' * 70}")
        print(f"  Strategy: {ft['strategy']}")
        print(f"  Checkpoint Strategy: {ft['checkpoint_strategy']}")
        print(f"  Checkpoint Interval: {ft['checkpoint_interval_minutes']} minutes")
        
        eff = result['efficiency']
        print(f"\n{'─' * 70}")
        print("EFFICIENCY ANALYSIS")
        print(f"{'─' * 70}")
        print(f"  MFU: {eff['efficiency']['mfu']:.1%}")
        print(f"  Communication Efficiency: {eff['efficiency']['communication_efficiency']:.1%}")
        print(f"\n  Bottleneck:")
        print(f"    • Compute: {eff['bottleneck']['compute_pct']:.0f}%")
        print(f"    • Communication: {eff['bottleneck']['communication_pct']:.0f}%")
        print(f"    • Memory: {eff['bottleneck']['memory_pct']:.0f}%")
        print(f"\n  Cost:")
        print(f"    • ${eff['cost']['cost_per_trillion_tokens']:,.0f} per trillion tokens")
        
        if result.get('recommendations'):
            print(f"\n{'─' * 70}")
            print("RECOMMENDATIONS")
            print(f"{'─' * 70}")
            for rec in result['recommendations']:
                print(f"  {rec}")
        
        print(f"\n{'─' * 70}")
        print("Use --json to get full SLURM script and Kubernetes config")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        prog="parallelism_planner",
        description="Parallelism Strategy Advisor - Comprehensive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # =========================================================================
    # recommend
    # =========================================================================
    recommend_parser = subparsers.add_parser(
        "recommend", 
        help="Get parallelism recommendations (TP/PP/DP/CP/EP)",
        description="Analyze model and hardware to recommend optimal parallelism strategies."
    )
    recommend_parser.add_argument("model", help="Model name or HuggingFace ID")
    recommend_parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size")
    recommend_parser.add_argument("-s", "--seq-length", type=int, default=2048, help="Sequence length")
    recommend_parser.add_argument("-g", "--goal", choices=["throughput", "latency", "memory", "efficiency"], 
                                  default="throughput", help="Optimization goal")
    recommend_parser.add_argument("-t", "--training", action="store_true", help="Configure for training")
    recommend_parser.add_argument("--mock-topology", choices=["8xB200", "8xH100"], help="Use mock topology")
    recommend_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    recommend_parser.set_defaults(func=cmd_recommend)
    
    # =========================================================================
    # sharding
    # =========================================================================
    sharding_parser = subparsers.add_parser(
        "sharding",
        help="Get ZeRO/FSDP/HSDP sharding recommendations",
        description="Analyze sharding strategies for data parallel training."
    )
    sharding_parser.add_argument("model", help="Model name or HuggingFace ID")
    sharding_parser.add_argument("--dp", type=int, default=8, help="Data parallel size")
    sharding_parser.add_argument("--memory", type=float, default=80, help="GPU memory in GB")
    sharding_parser.add_argument("-b", "--batch-size", type=int, default=1, help="Micro-batch size")
    sharding_parser.add_argument("-s", "--seq-length", type=int, default=2048, help="Sequence length")
    sharding_parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    sharding_parser.add_argument("--gpus", type=int, default=8, help="GPUs per node")
    sharding_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    sharding_parser.set_defaults(func=cmd_sharding)
    
    # =========================================================================
    # launch
    # =========================================================================
    launch_parser = subparsers.add_parser(
        "launch",
        help="Generate framework launch commands",
        description="Generate launch commands for torchrun, DeepSpeed, Accelerate, Megatron."
    )
    launch_parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    launch_parser.add_argument("--gpus", type=int, default=8, help="GPUs per node")
    launch_parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    launch_parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    launch_parser.add_argument("--dp", type=int, default=8, help="Data parallel size")
    launch_parser.add_argument("--sharding", default="none", 
                               choices=["none", "zero1", "zero2", "zero3", "fsdp", "hsdp"],
                               help="Sharding strategy")
    launch_parser.add_argument("--micro-batch", type=int, default=1, help="Micro-batch size")
    launch_parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    launch_parser.add_argument("--master-addr", default="localhost", help="Master address")
    launch_parser.add_argument("--master-port", type=int, default=29500, help="Master port")
    launch_parser.add_argument("--script", default="train.py", help="Training script")
    launch_parser.add_argument("--framework", choices=["torchrun", "deepspeed", "accelerate", "megatron"],
                               help="Output specific framework only")
    launch_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    launch_parser.set_defaults(func=cmd_launch)
    
    # =========================================================================
    # pareto
    # =========================================================================
    pareto_parser = subparsers.add_parser(
        "pareto",
        help="Cost/throughput Pareto analysis",
        description="Analyze cost vs throughput tradeoffs and find Pareto-optimal configurations."
    )
    pareto_parser.add_argument("model", help="Model name or HuggingFace ID")
    pareto_parser.add_argument("--gpu-cost", type=float, default=4.0, help="GPU hourly cost ($)")
    pareto_parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size")
    pareto_parser.add_argument("-s", "--seq-length", type=int, default=4096, help="Sequence length")
    pareto_parser.add_argument("-t", "--training", action="store_true", help="Analyze for training")
    pareto_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    pareto_parser.set_defaults(func=cmd_pareto)
    
    # =========================================================================
    # calibrate
    # =========================================================================
    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate from benchmark data",
        description="Load benchmark results and calibrate estimation models."
    )
    calibrate_parser.add_argument("--data-dir", help="Directory with benchmark results")
    calibrate_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    calibrate_parser.set_defaults(func=cmd_calibrate)
    
    # =========================================================================
    # presets
    # =========================================================================
    presets_parser = subparsers.add_parser(
        "presets",
        help="List available model presets",
    )
    presets_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    presets_parser.set_defaults(func=cmd_presets)
    
    # =========================================================================
    # topology
    # =========================================================================
    topology_parser = subparsers.add_parser(
        "topology",
        help="Detect or display hardware topology",
    )
    topology_parser.add_argument("--mock", choices=["8xB200", "8xH100"], help="Use mock topology")
    topology_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    topology_parser.set_defaults(func=cmd_topology)
    
    # =========================================================================
    # analyze
    # =========================================================================
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze model architecture",
    )
    analyze_parser.add_argument("model", help="Model name or HuggingFace ID")
    analyze_parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size for memory estimates")
    analyze_parser.add_argument("-s", "--seq-length", type=int, default=2048, help="Sequence length")
    analyze_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # =========================================================================
    # estimate (training time/cost)
    # =========================================================================
    estimate_parser = subparsers.add_parser(
        "estimate",
        help="Estimate training time and cost",
        description="Estimate training time, cost, and checkpoint storage requirements."
    )
    estimate_parser.add_argument("model", help="Model name or HuggingFace ID")
    estimate_parser.add_argument("--tokens", type=int, default=1_000_000_000_000, help="Total tokens to train (default: 1T)")
    estimate_parser.add_argument("--throughput", type=float, default=100000, help="Tokens per second")
    estimate_parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs")
    estimate_parser.add_argument("--gpu-cost", type=float, default=4.0, help="GPU hourly cost ($)")
    estimate_parser.add_argument("--checkpoint-interval", type=int, default=1_000_000_000, help="Checkpoint every N tokens")
    estimate_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    estimate_parser.set_defaults(func=cmd_estimate)
    
    # =========================================================================
    # compare (multi-model)
    # =========================================================================
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare parallelism for multiple models",
        description="Side-by-side comparison of parallelism recommendations."
    )
    compare_parser.add_argument("models", nargs="+", help="Model names to compare")
    compare_parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size")
    compare_parser.add_argument("-s", "--seq-length", type=int, default=2048, help="Sequence length")
    compare_parser.add_argument("-t", "--training", action="store_true", help="Configure for training")
    compare_parser.add_argument("--mock-topology", choices=["8xB200", "8xH100"], help="Use mock topology")
    compare_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    compare_parser.set_defaults(func=cmd_compare)
    
    # =========================================================================
    # slurm (job script generation)
    # =========================================================================
    slurm_parser = subparsers.add_parser(
        "slurm",
        help="Generate SLURM job script",
        description="Generate a SLURM job script for distributed training."
    )
    slurm_parser.add_argument("--job-name", default="train", help="Job name")
    slurm_parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    slurm_parser.add_argument("--gpus", type=int, default=8, help="GPUs per node")
    slurm_parser.add_argument("--time", type=int, default=24, help="Walltime in hours")
    slurm_parser.add_argument("--partition", default="gpu", help="SLURM partition")
    slurm_parser.add_argument("--account", help="Account/allocation")
    slurm_parser.add_argument("--script", default="train.py", help="Training script")
    slurm_parser.add_argument("--conda-env", help="Conda environment to activate")
    slurm_parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    slurm_parser.set_defaults(func=cmd_slurm)
    
    # =========================================================================
    # validate (configuration validation)
    # =========================================================================
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate parallelism configuration",
        description="Check configuration for correctness and compatibility issues."
    )
    validate_parser.add_argument("model", help="Model name or HuggingFace ID")
    validate_parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    validate_parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    validate_parser.add_argument("--dp", type=int, default=8, help="Data parallel size")
    validate_parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    validate_parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")
    validate_parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size")
    validate_parser.add_argument("-s", "--seq-length", type=int, default=2048, help="Sequence length")
    validate_parser.add_argument("-t", "--training", action="store_true", help="Include training memory")
    validate_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    validate_parser.set_defaults(func=cmd_validate)
    
    # =========================================================================
    # dry-run (quick sanity check)
    # =========================================================================
    dryrun_parser = subparsers.add_parser(
        "dry-run",
        help="Quick dry-run test before training",
        description="Run a quick test to validate GPU memory and basic forward/backward pass."
    )
    dryrun_parser.add_argument("model", help="Model name or HuggingFace ID")
    dryrun_parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    dryrun_parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    dryrun_parser.add_argument("--dp", type=int, default=1, help="Data parallel size")
    dryrun_parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")
    dryrun_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    dryrun_parser.set_defaults(func=cmd_dryrun)
    
    # =========================================================================
    # optimize (advanced optimizations)
    # =========================================================================
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Get advanced optimization recommendations",
        description="Get compound optimization recommendations combining precision, checkpointing, kernels, etc."
    )
    optimize_parser.add_argument("model", help="Model name or HuggingFace ID")
    optimize_parser.add_argument("--goal", choices=["throughput", "memory", "balanced"],
                                 default="balanced", help="Optimization goal")
    optimize_parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size")
    optimize_parser.add_argument("-s", "--seq-length", type=int, default=4096, help="Sequence length")
    optimize_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    optimize_parser.set_defaults(func=cmd_optimize)
    
    # =========================================================================
    # profile (workload-specific profiles)
    # =========================================================================
    profile_parser = subparsers.add_parser(
        "profile",
        help="Get workload-specific performance profiles",
        description="Get optimized settings for specific workloads (pretraining, finetuning, inference, etc.)"
    )
    profile_parser.add_argument("model", nargs="?", help="Model name or HuggingFace ID")
    profile_parser.add_argument("--workload", default="pretraining",
                                choices=["pretraining", "finetuning", "rlhf", "inference", 
                                        "long_context", "moe", "cost_optimized"],
                                help="Workload type")
    profile_parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size")
    profile_parser.add_argument("-s", "--seq-length", type=int, default=4096, help="Sequence length")
    profile_parser.add_argument("--lora", action="store_true", help="Use LoRA for fine-tuning")
    profile_parser.add_argument("--inference-mode", choices=["batch", "streaming", "realtime"],
                                default="batch", help="Inference mode")
    profile_parser.add_argument("--list", action="store_true", help="List available profiles")
    profile_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    profile_parser.set_defaults(func=cmd_profile)
    
    # =========================================================================
    # bottleneck (bottleneck analysis)
    # =========================================================================
    bottleneck_parser = subparsers.add_parser(
        "bottleneck",
        help="Analyze performance bottlenecks",
        description="Identify if training is compute/memory/communication bound."
    )
    bottleneck_parser.add_argument("model", help="Model name or HuggingFace ID")
    bottleneck_parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size")
    bottleneck_parser.add_argument("-s", "--seq-length", type=int, default=4096, help="Sequence length")
    bottleneck_parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    bottleneck_parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    bottleneck_parser.add_argument("--dp", type=int, default=8, help="Data parallel size")
    bottleneck_parser.add_argument("--throughput", type=float, help="Measured throughput (tokens/s)")
    bottleneck_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    bottleneck_parser.set_defaults(func=cmd_bottleneck)
    
    # =========================================================================
    # scaling (scaling analysis)
    # =========================================================================
    scaling_parser = subparsers.add_parser(
        "scaling",
        help="Analyze scaling efficiency",
        description="Project how performance scales with more GPUs."
    )
    scaling_parser.add_argument("model", help="Model name or HuggingFace ID")
    scaling_parser.add_argument("--throughput", type=float, required=True, help="Current throughput (tokens/s)")
    scaling_parser.add_argument("--gpus", type=int, default=8, help="Current GPU count")
    scaling_parser.add_argument("--max-gpus", type=int, default=512, help="Maximum GPUs to project")
    scaling_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    scaling_parser.set_defaults(func=cmd_scaling)
    
    # =========================================================================
    # whatif (what-if analysis)
    # =========================================================================
    whatif_parser = subparsers.add_parser(
        "whatif",
        help="What-if analysis for configuration changes",
        description="Analyze impact of configuration changes."
    )
    whatif_parser.add_argument("model", help="Model name or HuggingFace ID")
    whatif_parser.add_argument("--current-tp", type=int, default=1, help="Current TP")
    whatif_parser.add_argument("--current-pp", type=int, default=1, help="Current PP")
    whatif_parser.add_argument("--current-dp", type=int, default=8, help="Current DP")
    whatif_parser.add_argument("--current-batch", type=int, default=8, help="Current batch size")
    whatif_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    whatif_parser.set_defaults(func=cmd_whatif)
    
    # =========================================================================
    # batch-size (find max batch size)
    # =========================================================================
    batchsize_parser = subparsers.add_parser(
        "batch-size",
        help="Find maximum batch size that fits in memory",
        description="Binary search for max batch size with memory headroom."
    )
    batchsize_parser.add_argument("model", help="Model name or HuggingFace ID")
    batchsize_parser.add_argument("-s", "--seq-length", type=int, default=4096, help="Sequence length")
    batchsize_parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    batchsize_parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    batchsize_parser.add_argument("--dp", type=int, default=8, help="Data parallel size")
    batchsize_parser.add_argument("--target-batch", type=int, default=1024, help="Target effective batch size")
    batchsize_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    batchsize_parser.set_defaults(func=cmd_batchsize)
    
    # =========================================================================
    # auto-tune (automatic configuration tuning)
    # =========================================================================
    autotune_parser = subparsers.add_parser(
        "auto-tune",
        help="Automatically find optimal configuration",
        description="Search for optimal parallelism configuration."
    )
    autotune_parser.add_argument("model", help="Model name or HuggingFace ID")
    autotune_parser.add_argument("--goal", choices=["throughput", "memory", "latency"],
                                 default="throughput", help="Optimization goal")
    autotune_parser.add_argument("--target-batch", type=int, default=1024, help="Target batch size")
    autotune_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    autotune_parser.set_defaults(func=cmd_autotune)
    
    # =========================================================================
    # inference (inference optimization)
    # =========================================================================
    inference_parser = subparsers.add_parser(
        "inference",
        help="Get inference optimization recommendations",
        description="Quantization, KV cache, speculative decoding, engine recommendations."
    )
    inference_parser.add_argument("model", help="Model name or HuggingFace ID")
    inference_parser.add_argument("--goal", choices=["throughput", "latency", "memory"],
                                  default="throughput", help="Optimization goal")
    inference_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    inference_parser.set_defaults(func=cmd_inference)
    
    # =========================================================================
    # troubleshoot (error diagnosis)
    # =========================================================================
    troubleshoot_parser = subparsers.add_parser(
        "troubleshoot",
        help="Diagnose distributed training errors",
        description="Get solutions for common distributed training problems."
    )
    troubleshoot_parser.add_argument("--error", help="Error message to diagnose")
    troubleshoot_parser.add_argument("--list", action="store_true", help="List all known issues")
    troubleshoot_parser.add_argument("--nccl", action="store_true", help="Get NCCL tuning recommendations")
    troubleshoot_parser.add_argument("--interconnect", choices=["nvlink", "infiniband", "ethernet"],
                                     default="nvlink", help="Interconnect type for NCCL tuning")
    troubleshoot_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    troubleshoot_parser.set_defaults(func=cmd_troubleshoot)
    
    # =========================================================================
    # memory (detailed memory breakdown)
    # =========================================================================
    memory_parser = subparsers.add_parser(
        "memory",
        help="Detailed memory breakdown analysis",
        description="Analyze where GPU memory is used and find optimization opportunities."
    )
    memory_parser.add_argument("model", help="Model name or HuggingFace ID")
    memory_parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size")
    memory_parser.add_argument("-s", "--seq-length", type=int, default=4096, help="Sequence length")
    memory_parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    memory_parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    memory_parser.add_argument("--dp", type=int, default=8, help="Data parallel size")
    memory_parser.add_argument("--optimizer", default="adamw", 
                               choices=["adamw", "adamw_8bit", "adafactor", "sgd"],
                               help="Optimizer type")
    memory_parser.add_argument("--no-checkpointing", action="store_true", 
                               help="Disable gradient checkpointing in estimate")
    memory_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    memory_parser.set_defaults(func=cmd_memory)
    
    # =========================================================================
    # export (export complete config)
    # =========================================================================
    export_parser = subparsers.add_parser(
        "export",
        help="Export complete training configuration",
        description="Export DeepSpeed config, environment variables, and launch scripts."
    )
    export_parser.add_argument("model", help="Model name or HuggingFace ID")
    export_parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    export_parser.add_argument("--gpus", type=int, default=8, help="GPUs per node")
    export_parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    export_parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    export_parser.add_argument("--dp", type=int, default=8, help="Data parallel size")
    export_parser.add_argument("-b", "--batch-size", type=int, default=256, help="Global batch size")
    export_parser.add_argument("--zero-stage", type=int, default=2, choices=[0, 1, 2, 3],
                               help="ZeRO stage")
    export_parser.add_argument("--output-dir", help="Output directory for config files")
    export_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    export_parser.set_defaults(func=cmd_export)
    
    # =========================================================================
    # nccl (NCCL tuning - NEW!)
    # =========================================================================
    nccl_parser = subparsers.add_parser(
        "nccl",
        help="NCCL tuning advisor for distributed training",
        description="Get optimal NCCL configuration for your cluster."
    )
    nccl_parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    nccl_parser.add_argument("--gpus", type=int, default=8, help="GPUs per node")
    nccl_parser.add_argument("--model-size", type=float, default=70, help="Model size in billions")
    nccl_parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    nccl_parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    nccl_parser.add_argument("--diagnose", action="store_true", help="Diagnose NCCL issues")
    nccl_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    nccl_parser.set_defaults(func=cmd_nccl)
    
    # =========================================================================
    # rlhf (RLHF memory calculator - NEW!)
    # =========================================================================
    rlhf_parser = subparsers.add_parser(
        "rlhf",
        help="RLHF/DPO memory calculator and optimizer",
        description="Calculate memory requirements for PPO/DPO/ORPO training."
    )
    rlhf_parser.add_argument("model", help="Model name or HuggingFace ID")
    rlhf_parser.add_argument("--algorithm", choices=["ppo", "dpo", "orpo", "kto", "grpo"],
                             default="ppo", help="RLHF algorithm")
    rlhf_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    rlhf_parser.add_argument("--seq-length", type=int, default=2048, help="Sequence length")
    rlhf_parser.add_argument("--memory", type=float, default=80, help="GPU memory in GB")
    rlhf_parser.add_argument("--compare", action="store_true", help="Compare all algorithms")
    rlhf_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    rlhf_parser.set_defaults(func=cmd_rlhf)
    
    # =========================================================================
    # moe (MoE parallelism optimizer - NEW!)
    # =========================================================================
    moe_parser = subparsers.add_parser(
        "moe",
        help="Mixture of Experts parallelism optimizer",
        description="Optimize expert parallelism for MoE models."
    )
    moe_parser.add_argument("model", help="Model name (e.g., mixtral-8x7b)")
    moe_parser.add_argument("--num-experts", type=int, default=8, help="Number of experts")
    moe_parser.add_argument("--gpus", type=int, default=8, help="Total GPUs")
    moe_parser.add_argument("--memory", type=float, default=80, help="GPU memory in GB")
    moe_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    moe_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    moe_parser.set_defaults(func=cmd_moe)
    
    # =========================================================================
    # long-context (Long context optimizer - NEW!)
    # =========================================================================
    longctx_parser = subparsers.add_parser(
        "long-context",
        help="Long context training/inference optimizer",
        description="Optimize for long sequences (Ring Attention, Context Parallel)."
    )
    longctx_parser.add_argument("model", help="Model name or HuggingFace ID")
    longctx_parser.add_argument("--seq-length", type=int, default=128000, help="Target sequence length")
    longctx_parser.add_argument("--gpus", type=int, default=8, help="Total GPUs")
    longctx_parser.add_argument("--memory", type=float, default=80, help="GPU memory in GB")
    longctx_parser.add_argument("--method", choices=["auto", "context_parallel", "ring_attention", "flash_attention_v3"],
                                default="auto", help="Long context method")
    longctx_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    longctx_parser.set_defaults(func=cmd_long_context)
    
    # =========================================================================
    # vllm (vLLM configuration generator - NEW!)
    # =========================================================================
    vllm_parser = subparsers.add_parser(
        "vllm",
        help="vLLM configuration generator",
        description="Generate optimal vLLM server configuration."
    )
    vllm_parser.add_argument("model", help="Model name or HuggingFace ID")
    vllm_parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    vllm_parser.add_argument("--memory", type=float, default=80, help="GPU memory in GB")
    vllm_parser.add_argument("--target", choices=["throughput", "latency", "memory"],
                             default="throughput", help="Optimization target")
    vllm_parser.add_argument("--max-seq-length", type=int, default=8192, help="Max sequence length")
    vllm_parser.add_argument("--quantization", choices=["none", "awq", "gptq", "fp8", "int8"],
                             default="none", help="Quantization method")
    vllm_parser.add_argument("--compare-engines", action="store_true", help="Compare inference engines")
    vllm_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    vllm_parser.set_defaults(func=cmd_vllm)
    
    # =========================================================================
    # comm-overlap (Communication overlap analyzer - NEW!)
    # =========================================================================
    overlap_parser = subparsers.add_parser(
        "comm-overlap",
        help="Communication/computation overlap analyzer",
        description="Analyze opportunities for overlapping communication with computation."
    )
    overlap_parser.add_argument("model", help="Model name or HuggingFace ID")
    overlap_parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    overlap_parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    overlap_parser.add_argument("--dp", type=int, default=8, help="Data parallel size")
    overlap_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    overlap_parser.add_argument("--seq-length", type=int, default=4096, help="Sequence length")
    overlap_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    overlap_parser.set_defaults(func=cmd_comm_overlap)
    
    # =========================================================================
    # large-scale (large cluster optimization)
    # =========================================================================
    largescale_parser = subparsers.add_parser(
        "large-scale",
        help="Large-scale cluster optimization",
        description="Optimize for large distributed training clusters (100s-1000s of GPUs)."
    )
    largescale_parser.add_argument("model", help="Model name or HuggingFace ID")
    largescale_parser.add_argument("--nodes", type=int, default=8, help="Number of nodes")
    largescale_parser.add_argument("--gpus-per-node", type=int, default=8, help="GPUs per node")
    largescale_parser.add_argument("--gpu-memory", type=float, default=80, help="GPU memory in GB")
    largescale_parser.add_argument("--network", choices=["infiniband", "efa", "roce", "ethernet"],
                                   default="infiniband", help="Network type")
    largescale_parser.add_argument("--batch-size", type=int, default=1024, help="Global batch size")
    largescale_parser.add_argument("--seq-length", type=int, default=4096, help="Sequence length")
    largescale_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    largescale_parser.set_defaults(func=cmd_largescale)
    
    # =========================================================================
    # llm-advisor (LLM-powered optimization - NEW!)
    # =========================================================================
    llm_parser = subparsers.add_parser(
        "llm-advisor",
        help="LLM-powered optimization advisor (uses AI for tailored suggestions)",
        description="Get intelligent, context-aware optimization recommendations from LLM."
    )
    llm_parser.add_argument("model", help="Model name or HuggingFace ID")
    llm_parser.add_argument("--goal", choices=["throughput", "latency", "memory", "efficiency", "cost"],
                            default="throughput", help="Optimization goal")
    llm_parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs")
    llm_parser.add_argument("--inference", action="store_true", help="Optimize for inference (default: training)")
    llm_parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                            help="LLM provider")
    llm_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    llm_parser.set_defaults(func=cmd_llm_advisor)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()

