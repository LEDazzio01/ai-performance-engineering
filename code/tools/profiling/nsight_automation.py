#!/usr/bin/env python3
"""Automated Nsight Systems and Nsight Compute profiling for Blackwell.

Provides automated profiling workflows with:
- Metric selection for different workload types
- Batch profiling across multiple configurations
- Report generation with hotspot detection
- Integration with benchmark harness
"""

import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.python.logger import get_logger

logger = get_logger(__name__)


class NsightAutomation:
    """Automated Nsight profiling."""
    
    # Metric sets for different workload types
    METRIC_SETS = {
        'memory_bound': [
            'dram__bytes_read.sum',
            'dram__bytes_write.sum',
            'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum',
            'l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum',
            'lts__t_sectors_op_read.sum',
            'lts__t_sectors_op_write.sum',
            'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct',
        ],
        'compute_bound': [
            'sm__cycles_active.avg',
            'sm__cycles_active.sum',
            'sm__pipe_tensor_cycles_active.avg',
            'smsp__inst_executed.avg',
            'smsp__sass_thread_inst_executed_op_fp16_pred_on.sum',
            'smsp__sass_thread_inst_executed_op_fp32_pred_on.sum',
            'smsp__sass_thread_inst_executed_op_fp64_pred_on.sum',
        ],
        'tensor_core': [
            'sm__pipe_tensor_cycles_active.avg',
            'sm__pipe_tensor_op_hmma_cycles_active.avg',
            'smsp__inst_executed_pipe_tensor.avg',
            'smsp__sass_thread_inst_executed_op_fp16_pred_on.sum',
            'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum',
        ],
        'communication': [
            'nvlink__bytes_read.sum',
            'nvlink__bytes_write.sum',
            'pcie__bytes_read.sum',
            'pcie__bytes_write.sum',
        ],
        'occupancy': [
            'sm__warps_active.avg.pct_of_peak_sustained_active',
            'sm__maximum_warps_per_active_cycle_pct',
            'achieved_occupancy',
        ],
    }
    
    def __init__(self, output_dir: Path = Path("artifacts/nsight")):
        """Initialize Nsight automation.
        
        Args:
            output_dir: Directory for profiling outputs
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check availability
        self.nsys_available = self._check_command("nsys")
        self.ncu_available = self._check_command("ncu")
        
        logger.info(f"Nsight Systems: {'✓' if self.nsys_available else '✗'}")
        logger.info(f"Nsight Compute: {'✓' if self.ncu_available else '✗'}")
    
    def _check_command(self, cmd: str) -> bool:
        """Check if command is available."""
        try:
            subprocess.run([cmd, '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def profile_nsys(
        self,
        command: List[str],
        output_name: str,
        trace_cuda: bool = True,
        trace_nvtx: bool = True,
        trace_osrt: bool = True,
        full_timeline: bool = False,
        trace_forks: bool = True,
        preset: str = "light",
    ) -> Optional[Path]:
        """Run Nsight Systems profiling.
        
        Args:
            command: Command to profile
            output_name: Base name for output file
            trace_cuda: Trace CUDA API calls
            trace_nvtx: Trace NVTX markers
            trace_osrt: Trace OS runtime
            full_timeline: If True, include driver/cu/pti traces and richer capture flags
            trace_forks: If True, trace child processes before exec
        
        Presets:
            - light (default): cuda,nvtx,osrt, no sampling/ctx switch.
            - full: adds cuda-hw, cublas, cusolver, cusparse, cudnn, fork tracing.

        Returns:
            output_path: Path to .nsys-rep file, or None if failed
        """
        if not self.nsys_available:
            logger.error("Nsight Systems not available")
            return None
        
        output_path = self.output_dir / f"{output_name}.nsys-rep"
        
        # Build nsys command
        nsys_cmd = [
            'nsys', 'profile',
            '--output', str(output_path),
            '--force-overwrite', 'true',
        ]
        
        trace_categories = []

        # Apply preset overrides first
        preset_normalized = (preset or "light").strip().lower()
        if preset_normalized == "full":
            full_timeline = True
            trace_forks = True
        elif preset_normalized == "light":
            full_timeline = False
        if trace_cuda:
            trace_categories.append('cuda')
        if trace_nvtx:
            trace_categories.append('nvtx')
        if trace_osrt:
            trace_categories.append('osrt')
        if full_timeline:
            trace_categories.extend(['cuda-hw', 'cublas', 'cusolver', 'cusparse', 'cudnn'])
        if trace_categories:
            # dedupe while preserving order
            seen = set()
            deduped = []
            for cat in trace_categories:
                if cat not in seen:
                    seen.add(cat)
                    deduped.append(cat)
            nsys_cmd.extend(['--trace', ",".join(deduped)])
            if full_timeline or preset_normalized == "full":
                logger.warning("NSYS full timeline enabled: traces will be larger and runs slower. Ensure TMPDIR has ample space.")

        # Prefer no sampling/ctx-switch overhead when hunting source attribution
        nsys_cmd.extend([
            '--sample', 'none',
            '--cpuctxsw', 'none',
            '--cuda-memory-usage', 'true',
            '--cuda-um-gpu-page-faults', 'true',
            '--cuda-um-cpu-page-faults', 'true',
        ])
        if trace_forks:
            nsys_cmd.extend(['--trace-fork-before-exec', 'true'])
        
        nsys_cmd.extend(command)
        
        logger.info(f"Running: {' '.join(nsys_cmd)}")
        
        try:
            result = subprocess.run(
                nsys_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Nsight Systems trace saved to {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            # Automatic fallback: drop full_timeline categories and retry once
            logger.error(f"Nsight Systems failed: {e.stderr}")
            if full_timeline or preset_normalized == "full":
                logger.warning("Retrying NSYS capture with preset=light (reduced trace categories)")
                return self.profile_nsys(
                    command,
                    output_name,
                    trace_cuda=trace_cuda,
                    trace_nvtx=trace_nvtx,
                    trace_osrt=trace_osrt,
                    full_timeline=False,
                    trace_forks=False,
                    preset="light",
                )
            return None
    
    def profile_ncu(
        self,
        command: List[str],
        output_name: str,
        workload_type: str = 'memory_bound',
        kernel_filter: Optional[str] = None,
    ) -> Optional[Path]:
        """Run Nsight Compute profiling.
        
        Args:
            command: Command to profile
            output_name: Base name for output file
            workload_type: Type of workload for metric selection
            kernel_filter: Optional kernel name filter
        
        Returns:
            output_path: Path to .ncu-rep file, or None if failed
        """
        if not self.ncu_available:
            logger.error("Nsight Compute not available")
            return None
        
        output_path = self.output_dir / f"{output_name}.ncu-rep"
        
        # Get metrics for workload type
        metrics = self.METRIC_SETS.get(workload_type, self.METRIC_SETS['memory_bound'])
        
        # Build ncu command
        ncu_cmd = [
            'ncu',
            '--set', 'full',  # Full metric set
            '--target-processes', 'all',
            '--export', str(output_path),
            '--force-overwrite',
        ]
        
        # Add custom metrics
        for metric in metrics:
            ncu_cmd.extend(['--metrics', metric])
        
        # Add kernel filter if specified
        if kernel_filter:
            ncu_cmd.extend(['--kernel-name', kernel_filter])
        
        ncu_cmd.extend(command)
        
        logger.info(f"Running: {' '.join(ncu_cmd[:6])} ...")
        
        try:
            result = subprocess.run(
                ncu_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Nsight Compute report saved to {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Nsight Compute failed: {e.stderr}")
            return None
    
    def batch_profile(
        self,
        configs: List[Dict[str, Any]],
        base_command: List[str]
    ) -> List[Path]:
        """Run batch profiling with multiple configurations.
        
        Args:
            configs: List of config dicts with keys:
                - name: Output name
                - args: Additional command arguments
                - workload_type: Type for metric selection
            base_command: Base command (e.g., ['python', 'script.py'])
        
        Returns:
            output_paths: List of generated report paths
        """
        outputs = []
        
        for config in configs:
            name = config['name']
            args = config.get('args', [])
            workload_type = config.get('workload_type', 'memory_bound')
            
            # Build full command
            full_cmd = base_command + args
            
            logger.info(f"Profiling configuration: {name}")
            
            # Run Nsight Compute
            ncu_path = self.profile_ncu(
                full_cmd,
                f"{name}_ncu",
                workload_type=workload_type
            )
            
            if ncu_path:
                outputs.append(ncu_path)
        
        logger.info(f"Batch profiling complete: {len(outputs)} reports generated")
        return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Automated Nsight Profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile with Nsight Systems
  python nsight_automation.py --tool nsys --output my_trace \\
    -- python ch1/optimized_performance.py
  
  # Profile with Nsight Compute (memory-bound)
  python nsight_automation.py --tool ncu --output my_profile \\
    --workload-type memory_bound -- python ch7/optimized_hbm3ecopy.py
  
  # Batch profiling
  python nsight_automation.py --batch-config configs.json
        """
    )
    
    parser.add_argument('--tool', type=str, choices=['nsys', 'ncu'],
                       help='Profiling tool')
    parser.add_argument('--output', type=str, required=True,
                       help='Output base name')
    parser.add_argument('--workload-type', type=str,
                       choices=list(NsightAutomation.METRIC_SETS.keys()),
                       default='memory_bound',
                       help='Workload type for metric selection')
    parser.add_argument('--kernel-filter', type=str,
                       help='Filter kernels by name pattern')
    parser.add_argument('--trace-cuda', action='store_true', default=True, help='Trace CUDA API (nsys)')
    parser.add_argument('--trace-nvtx', action='store_true', default=True, help='Trace NVTX ranges (nsys)')
    parser.add_argument('--trace-osrt', action='store_true', default=True, help='Trace OS runtime (nsys)')
    parser.add_argument('--full-timeline', action='store_true', default=False, help='Enable richer NSYS tracing (cuda-hw, cublas, cusolver, cusparse, cudnn)')
    parser.add_argument('--trace-forks', action='store_true', default=False, help='Trace child processes before exec (nsys)')
    parser.add_argument('--preset', type=str, default='light', choices=['light', 'full'],
                        help='NSYS preset: light (default) or full (adds cuda-hw/cublas/cusolver/cusparse/cudnn and fork tracing)')
    parser.add_argument('--batch-config', type=Path,
                       help='JSON config for batch profiling')
    parser.add_argument('command', nargs='*',
                       help='Command to profile (after --)')
    
    args = parser.parse_args()
    
    # Create automation
    automation = NsightAutomation()
    
    # Batch mode
    if args.batch_config:
        with open(args.batch_config) as f:
            configs = json.load(f)
        
        outputs = automation.batch_profile(
            configs=configs['profiles'],
            base_command=configs['base_command']
        )
        
        print(f"\n{'='*60}")
        print(f"Batch Profiling Complete")
        print(f"{'='*60}")
        print(f"Reports generated: {len(outputs)}")
        for path in outputs:
            print(f"  - {path}")
        print(f"{'='*60}\n")
        return
    
    # Single profile mode
    if not args.command:
        parser.error("Command required (use -- before command)")
    
    if args.tool == 'nsys':
        output = automation.profile_nsys(
            args.command,
            args.output,
            trace_cuda=args.trace_cuda,
            trace_nvtx=args.trace_nvtx,
            trace_osrt=args.trace_osrt,
            full_timeline=args.full_timeline,
            trace_forks=args.trace_forks,
            preset=args.preset,
        )
    elif args.tool == 'ncu':
        output = automation.profile_ncu(
            args.command,
            args.output,
            workload_type=args.workload_type,
            kernel_filter=args.kernel_filter
        )
    else:
        parser.error("--tool required")
    
    if output:
        print(f"\n{'='*60}")
        print(f"Profiling Complete")
        print(f"{'='*60}")
        print(f"Output: {output}")
        if args.tool == 'nsys':
            print(f"Preset: {args.preset} (full_timeline={args.full_timeline or args.preset=='full'})")
            if args.preset == 'full' or args.full_timeline:
                print("Warning: NSYS full timeline enabled; captures run longer and produce larger traces. Ensure TMPDIR has space.")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
