"""
Verification Report Generator - Create detailed HTML/JSON reports for benchmark verification.

Provides comprehensive verification reporting including:
- Verification status by benchmark
- Quarantine summary and details
- Theoretical peak calculations
- Efficiency metrics (% of theoretical peak)
- Anti-cheat check results
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parents[3]))

try:
    from core.benchmark.quarantine import QuarantineManager
    from core.benchmark.verification import QuarantineReason
except ImportError:
    QuarantineManager = None
    QuarantineReason = None


# =============================================================================
# THEORETICAL PEAK CALCULATOR
# =============================================================================

@dataclass
class TheoreticalPeak:
    """Theoretical peak performance for a GPU."""
    gpu_name: str
    architecture: str
    
    # Compute peaks (TFLOPS)
    fp32_tflops: float = 0.0
    fp16_tflops: float = 0.0
    bf16_tflops: float = 0.0
    fp8_tflops: float = 0.0
    int8_tops: float = 0.0
    
    # Memory peaks (GB/s)
    memory_bandwidth_gbps: float = 0.0
    l2_cache_bandwidth_gbps: float = 0.0
    
    # Other
    sm_count: int = 0
    memory_size_gb: float = 0.0
    tdp_watts: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# GPU specifications database
GPU_THEORETICAL_PEAKS: Dict[str, TheoreticalPeak] = {
    "B200": TheoreticalPeak(
        gpu_name="NVIDIA B200",
        architecture="Blackwell",
        fp32_tflops=80.0,
        fp16_tflops=2250.0,
        bf16_tflops=2250.0,
        fp8_tflops=4500.0,
        int8_tops=4500.0,
        memory_bandwidth_gbps=8000.0,
        l2_cache_bandwidth_gbps=32000.0,  # ~4x HBM bandwidth
        sm_count=148,
        memory_size_gb=192.0,
        tdp_watts=1000.0,
    ),
    "B100": TheoreticalPeak(
        gpu_name="NVIDIA B100",
        architecture="Blackwell",
        fp32_tflops=60.0,
        fp16_tflops=1800.0,
        bf16_tflops=1800.0,
        fp8_tflops=3600.0,
        int8_tops=3600.0,
        memory_bandwidth_gbps=8000.0,
        l2_cache_bandwidth_gbps=32000.0,
        sm_count=132,
        memory_size_gb=192.0,
        tdp_watts=700.0,
    ),
    "H100_SXM": TheoreticalPeak(
        gpu_name="NVIDIA H100 SXM",
        architecture="Hopper",
        fp32_tflops=67.0,
        fp16_tflops=1979.0,
        bf16_tflops=1979.0,
        fp8_tflops=3958.0,
        int8_tops=3958.0,
        memory_bandwidth_gbps=3350.0,
        l2_cache_bandwidth_gbps=13400.0,
        sm_count=132,
        memory_size_gb=80.0,
        tdp_watts=700.0,
    ),
    "H100_PCIe": TheoreticalPeak(
        gpu_name="NVIDIA H100 PCIe",
        architecture="Hopper",
        fp32_tflops=51.0,
        fp16_tflops=1513.0,
        bf16_tflops=1513.0,
        fp8_tflops=3026.0,
        int8_tops=3026.0,
        memory_bandwidth_gbps=2000.0,
        l2_cache_bandwidth_gbps=8000.0,
        sm_count=114,
        memory_size_gb=80.0,
        tdp_watts=350.0,
    ),
    "H200": TheoreticalPeak(
        gpu_name="NVIDIA H200",
        architecture="Hopper",
        fp32_tflops=67.0,
        fp16_tflops=1979.0,
        bf16_tflops=1979.0,
        fp8_tflops=3958.0,
        int8_tops=3958.0,
        memory_bandwidth_gbps=4800.0,  # HBM3e
        l2_cache_bandwidth_gbps=19200.0,
        sm_count=132,
        memory_size_gb=141.0,
        tdp_watts=700.0,
    ),
    "A100_SXM": TheoreticalPeak(
        gpu_name="NVIDIA A100 SXM",
        architecture="Ampere",
        fp32_tflops=19.5,
        fp16_tflops=312.0,
        bf16_tflops=312.0,
        fp8_tflops=0.0,  # No FP8 support
        int8_tops=624.0,
        memory_bandwidth_gbps=2039.0,
        l2_cache_bandwidth_gbps=8156.0,
        sm_count=108,
        memory_size_gb=80.0,
        tdp_watts=400.0,
    ),
    "A100_PCIe": TheoreticalPeak(
        gpu_name="NVIDIA A100 PCIe",
        architecture="Ampere",
        fp32_tflops=19.5,
        fp16_tflops=312.0,
        bf16_tflops=312.0,
        fp8_tflops=0.0,
        int8_tops=624.0,
        memory_bandwidth_gbps=1935.0,
        l2_cache_bandwidth_gbps=7740.0,
        sm_count=108,
        memory_size_gb=80.0,
        tdp_watts=300.0,
    ),
    "L40S": TheoreticalPeak(
        gpu_name="NVIDIA L40S",
        architecture="Ada Lovelace",
        fp32_tflops=91.6,
        fp16_tflops=733.0,
        bf16_tflops=733.0,
        fp8_tflops=733.0,
        int8_tops=733.0,
        memory_bandwidth_gbps=864.0,
        l2_cache_bandwidth_gbps=3456.0,
        sm_count=142,
        memory_size_gb=48.0,
        tdp_watts=350.0,
    ),
    "RTX_4090": TheoreticalPeak(
        gpu_name="NVIDIA RTX 4090",
        architecture="Ada Lovelace",
        fp32_tflops=82.6,
        fp16_tflops=660.0,
        bf16_tflops=660.0,
        fp8_tflops=1320.0,
        int8_tops=1320.0,
        memory_bandwidth_gbps=1008.0,
        l2_cache_bandwidth_gbps=4032.0,
        sm_count=128,
        memory_size_gb=24.0,
        tdp_watts=450.0,
    ),
}


def get_theoretical_peak(gpu_name: str) -> Optional[TheoreticalPeak]:
    """
    Get theoretical peak performance for a GPU.
    
    Args:
        gpu_name: GPU name string (fuzzy matching supported)
        
    Returns:
        TheoreticalPeak if found, None otherwise
    """
    gpu_upper = gpu_name.upper()
    
    # Direct key match
    for key, peak in GPU_THEORETICAL_PEAKS.items():
        if key in gpu_upper or key.replace("_", " ") in gpu_upper:
            return peak
    
    # Fuzzy match by name
    for key, peak in GPU_THEORETICAL_PEAKS.items():
        if peak.gpu_name.upper() in gpu_upper or gpu_upper in peak.gpu_name.upper():
            return peak
    
    # Pattern matching
    if "B200" in gpu_upper or "BLACKWELL" in gpu_upper:
        return GPU_THEORETICAL_PEAKS.get("B200")
    if "B100" in gpu_upper:
        return GPU_THEORETICAL_PEAKS.get("B100")
    if "H200" in gpu_upper:
        return GPU_THEORETICAL_PEAKS.get("H200")
    if "H100" in gpu_upper:
        return GPU_THEORETICAL_PEAKS.get("H100_SXM" if "SXM" in gpu_upper else "H100_PCIe")
    if "A100" in gpu_upper:
        return GPU_THEORETICAL_PEAKS.get("A100_SXM" if "SXM" in gpu_upper else "A100_PCIe")
    if "L40" in gpu_upper:
        return GPU_THEORETICAL_PEAKS.get("L40S")
    if "4090" in gpu_upper or "RTX 4090" in gpu_upper:
        return GPU_THEORETICAL_PEAKS.get("RTX_4090")
    
    return None


def calculate_efficiency(
    achieved_tflops: float,
    theoretical_tflops: float,
) -> float:
    """Calculate efficiency as percentage of theoretical peak."""
    if theoretical_tflops <= 0:
        return 0.0
    return (achieved_tflops / theoretical_tflops) * 100.0


def calculate_roofline_bound(
    arithmetic_intensity: float,  # FLOP/byte
    peak_tflops: float,
    peak_bandwidth_gbps: float,
) -> str:
    """Determine if workload is compute-bound or memory-bound."""
    if peak_bandwidth_gbps <= 0:
        return "unknown"
    
    # Ridge point = peak_tflops / (peak_bandwidth_gbps / 1000)
    peak_bandwidth_tbps = peak_bandwidth_gbps / 1000.0
    ridge_point = peak_tflops / peak_bandwidth_tbps if peak_bandwidth_tbps > 0 else float('inf')
    
    if arithmetic_intensity < ridge_point:
        return "memory_bound"
    else:
        return "compute_bound"


# =============================================================================
# VERIFICATION DATA STRUCTURES
# =============================================================================

@dataclass
class VerificationBenchmarkResult:
    """Verification result for a single benchmark."""
    name: str
    chapter: str = ""
    
    # Verification status
    verified: bool = False
    verification_passed: bool = False
    quarantined: bool = False
    quarantine_reason: Optional[str] = None
    
    # Performance data
    baseline_time_ms: float = 0.0
    optimized_time_ms: float = 0.0
    speedup: float = 1.0
    
    # Anti-cheat checks
    jitter_check_passed: Optional[bool] = None
    fresh_input_check_passed: Optional[bool] = None
    workload_invariant_passed: Optional[bool] = None
    signature_match: Optional[bool] = None
    
    # Theoretical peak comparison
    achieved_tflops: Optional[float] = None
    theoretical_tflops: Optional[float] = None
    efficiency_percent: Optional[float] = None
    
    # Memory
    achieved_bandwidth_gbps: Optional[float] = None
    theoretical_bandwidth_gbps: Optional[float] = None
    memory_efficiency_percent: Optional[float] = None
    
    # Notes
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class VerificationReportData:
    """Complete verification report data."""
    title: str = "Benchmark Verification Report"
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # GPU info
    gpu_name: str = ""
    gpu_architecture: str = ""
    theoretical_peak: Optional[TheoreticalPeak] = None
    
    # Summary
    total_benchmarks: int = 0
    verified_count: int = 0
    passed_count: int = 0
    failed_count: int = 0
    quarantined_count: int = 0
    
    # Verification coverage
    coverage_percent: float = 0.0
    pass_rate_percent: float = 0.0
    
    # Quarantine breakdown
    quarantine_by_reason: Dict[str, int] = field(default_factory=dict)
    
    # Benchmarks
    benchmarks: List[VerificationBenchmarkResult] = field(default_factory=list)
    
    # Anti-cheat summary
    jitter_checks_run: int = 0
    jitter_checks_passed: int = 0
    fresh_input_checks_run: int = 0
    fresh_input_checks_passed: int = 0
    workload_checks_run: int = 0
    workload_checks_passed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "title": self.title,
            "generated_at": self.generated_at,
            "gpu_name": self.gpu_name,
            "gpu_architecture": self.gpu_architecture,
            "summary": {
                "total_benchmarks": self.total_benchmarks,
                "verified_count": self.verified_count,
                "passed_count": self.passed_count,
                "failed_count": self.failed_count,
                "quarantined_count": self.quarantined_count,
                "coverage_percent": round(self.coverage_percent, 1),
                "pass_rate_percent": round(self.pass_rate_percent, 1),
            },
            "quarantine_by_reason": self.quarantine_by_reason,
            "anti_cheat_summary": {
                "jitter_checks": {"run": self.jitter_checks_run, "passed": self.jitter_checks_passed},
                "fresh_input_checks": {"run": self.fresh_input_checks_run, "passed": self.fresh_input_checks_passed},
                "workload_checks": {"run": self.workload_checks_run, "passed": self.workload_checks_passed},
            },
            "benchmarks": [b.to_dict() for b in self.benchmarks],
        }
        
        if self.theoretical_peak:
            data["theoretical_peak"] = self.theoretical_peak.to_dict()
        
        return data


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class VerificationReportGenerator:
    """
    Generate HTML and JSON verification reports.
    
    Usage:
        generator = VerificationReportGenerator()
        report = generator.create_report(benchmark_results, gpu_name="H100")
        
        # Generate HTML
        generator.generate_html(report, "verification_report.html")
        
        # Generate JSON
        generator.generate_json(report, "verification_report.json")
    """
    
    def __init__(self, quarantine_manager: Optional["QuarantineManager"] = None):
        """
        Initialize the report generator.
        
        Args:
            quarantine_manager: Optional QuarantineManager for quarantine data
        """
        self.quarantine_manager = quarantine_manager
    
    def create_report(
        self,
        benchmark_results: List[Dict[str, Any]],
        gpu_name: str = "",
        title: str = "Benchmark Verification Report",
    ) -> VerificationReportData:
        """
        Create a verification report from benchmark results.
        
        Args:
            benchmark_results: List of benchmark result dictionaries
            gpu_name: GPU name for theoretical peak lookup
            title: Report title
            
        Returns:
            VerificationReportData
        """
        report = VerificationReportData(title=title)
        
        # Get GPU info
        if gpu_name:
            report.gpu_name = gpu_name
            theoretical = get_theoretical_peak(gpu_name)
            if theoretical:
                report.theoretical_peak = theoretical
                report.gpu_architecture = theoretical.architecture
        
        # Process benchmarks
        for result in benchmark_results:
            bench = self._process_benchmark_result(result, report.theoretical_peak)
            report.benchmarks.append(bench)
            
            # Update counters
            report.total_benchmarks += 1
            if bench.verified:
                report.verified_count += 1
                if bench.verification_passed:
                    report.passed_count += 1
                else:
                    report.failed_count += 1
            if bench.quarantined:
                report.quarantined_count += 1
                reason = bench.quarantine_reason or "unknown"
                report.quarantine_by_reason[reason] = report.quarantine_by_reason.get(reason, 0) + 1
            
            # Anti-cheat counters
            if bench.jitter_check_passed is not None:
                report.jitter_checks_run += 1
                if bench.jitter_check_passed:
                    report.jitter_checks_passed += 1
            if bench.fresh_input_check_passed is not None:
                report.fresh_input_checks_run += 1
                if bench.fresh_input_check_passed:
                    report.fresh_input_checks_passed += 1
            if bench.workload_invariant_passed is not None:
                report.workload_checks_run += 1
                if bench.workload_invariant_passed:
                    report.workload_checks_passed += 1
        
        # Calculate percentages
        if report.total_benchmarks > 0:
            report.coverage_percent = (report.verified_count / report.total_benchmarks) * 100
        if report.verified_count > 0:
            report.pass_rate_percent = (report.passed_count / report.verified_count) * 100
        
        # Add quarantine data if manager available
        if self.quarantine_manager:
            self._add_quarantine_data(report)
        
        return report
    
    def _process_benchmark_result(
        self,
        result: Dict[str, Any],
        theoretical: Optional[TheoreticalPeak],
    ) -> VerificationBenchmarkResult:
        """Process a single benchmark result dict into VerificationBenchmarkResult."""
        bench = VerificationBenchmarkResult(
            name=result.get("name", result.get("example", "unknown")),
            chapter=result.get("chapter", ""),
            verified=result.get("verified", False),
            verification_passed=result.get("verification_passed", result.get("equivalent", False)),
            quarantined=result.get("quarantined", False),
            quarantine_reason=result.get("quarantine_reason"),
            baseline_time_ms=result.get("baseline_time_ms", result.get("baseline_mean", 0)),
            optimized_time_ms=result.get("optimized_time_ms", result.get("optimized_mean", 0)),
            speedup=result.get("speedup", 1.0),
            jitter_check_passed=result.get("jitter_check_passed"),
            fresh_input_check_passed=result.get("fresh_input_check_passed"),
            workload_invariant_passed=result.get("workload_invariant_passed"),
            signature_match=result.get("signature_match", result.get("equivalent", None)),
            notes=result.get("notes", ""),
        )
        
        # Calculate efficiency if we have theoretical peak and performance data
        if theoretical:
            achieved_tflops = result.get("achieved_tflops", result.get("tflops"))
            if achieved_tflops:
                bench.achieved_tflops = achieved_tflops
                bench.theoretical_tflops = theoretical.fp16_tflops  # Default to FP16
                bench.efficiency_percent = calculate_efficiency(
                    achieved_tflops, theoretical.fp16_tflops
                )
            
            achieved_bw = result.get("achieved_bandwidth_gbps", result.get("bandwidth_gbps"))
            if achieved_bw:
                bench.achieved_bandwidth_gbps = achieved_bw
                bench.theoretical_bandwidth_gbps = theoretical.memory_bandwidth_gbps
                bench.memory_efficiency_percent = calculate_efficiency(
                    achieved_bw, theoretical.memory_bandwidth_gbps
                )
        
        return bench
    
    def _add_quarantine_data(self, report: VerificationReportData) -> None:
        """Add quarantine data from QuarantineManager."""
        if not self.quarantine_manager:
            return
        
        records = self.quarantine_manager.get_all_records()
        for path, record in records.items():
            reason = record.quarantine_reason.value if hasattr(record.quarantine_reason, 'value') else str(record.quarantine_reason)
            report.quarantine_by_reason[reason] = report.quarantine_by_reason.get(reason, 0) + 1
    
    def generate_json(
        self,
        report: VerificationReportData,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Generate JSON report.
        
        Args:
            report: VerificationReportData
            output_path: Output file path
            
        Returns:
            Path to generated file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        return output_path
    
    def generate_html(
        self,
        report: VerificationReportData,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Generate HTML report.
        
        Args:
            report: VerificationReportData
            output_path: Output file path
            
        Returns:
            Path to generated file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        html = self._generate_html_content(report)
        output_path.write_text(html)
        
        return output_path
    
    def _generate_html_content(self, report: VerificationReportData) -> str:
        """Generate HTML content for the report."""
        
        # Status colors
        def status_color(passed: bool) -> str:
            return "#22c55e" if passed else "#ef4444"
        
        def status_icon(passed: bool) -> str:
            return "✅" if passed else "❌"
        
        # Generate benchmark rows
        benchmark_rows = ""
        for b in sorted(report.benchmarks, key=lambda x: (not x.verification_passed, -x.speedup)):
            verify_status = status_icon(b.verification_passed) if b.verified else "⏸️"
            quarantine_badge = '<span style="background: #dc2626; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px;">QUARANTINED</span>' if b.quarantined else ""
            
            efficiency_str = f"{b.efficiency_percent:.1f}%" if b.efficiency_percent else "-"
            mem_efficiency_str = f"{b.memory_efficiency_percent:.1f}%" if b.memory_efficiency_percent else "-"
            
            # Anti-cheat badges
            checks_html = ""
            if b.jitter_check_passed is not None:
                checks_html += f'<span style="color: {status_color(b.jitter_check_passed)};">J</span> '
            if b.fresh_input_check_passed is not None:
                checks_html += f'<span style="color: {status_color(b.fresh_input_check_passed)};">F</span> '
            if b.workload_invariant_passed is not None:
                checks_html += f'<span style="color: {status_color(b.workload_invariant_passed)};">W</span>'
            
            benchmark_rows += f"""
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #334155;">
                    {b.name} {quarantine_badge}
                    <div style="font-size: 11px; color: #64748b;">{b.chapter}</div>
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #334155; text-align: center;">{verify_status}</td>
                <td style="padding: 12px; border-bottom: 1px solid #334155; text-align: right;">{b.baseline_time_ms:.2f}</td>
                <td style="padding: 12px; border-bottom: 1px solid #334155; text-align: right;">{b.optimized_time_ms:.2f}</td>
                <td style="padding: 12px; border-bottom: 1px solid #334155; text-align: right; font-weight: bold; color: {status_color(b.speedup > 1)};">{b.speedup:.2f}x</td>
                <td style="padding: 12px; border-bottom: 1px solid #334155; text-align: center;">{efficiency_str}</td>
                <td style="padding: 12px; border-bottom: 1px solid #334155; text-align: center;">{mem_efficiency_str}</td>
                <td style="padding: 12px; border-bottom: 1px solid #334155; text-align: center; font-family: monospace;">{checks_html}</td>
            </tr>
            """
        
        # Generate quarantine breakdown
        quarantine_breakdown = ""
        for reason, count in sorted(report.quarantine_by_reason.items(), key=lambda x: -x[1]):
            quarantine_breakdown += f"""
            <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #334155;">
                <span style="font-family: monospace; font-size: 12px;">{reason}</span>
                <span style="color: #ef4444; font-weight: bold;">{count}</span>
            </div>
            """
        
        # Theoretical peak section
        peak_section = ""
        if report.theoretical_peak:
            p = report.theoretical_peak
            peak_section = f"""
            <div class="card">
                <div class="card-title">🎯 Theoretical Peak ({p.gpu_name})</div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                    <div>
                        <div style="font-size: 24px; font-weight: bold; color: #00f5d4;">{p.fp16_tflops:.0f}</div>
                        <div style="font-size: 12px; color: #94a3b8;">FP16 TFLOPS</div>
                    </div>
                    <div>
                        <div style="font-size: 24px; font-weight: bold; color: #00f5d4;">{p.fp8_tflops:.0f}</div>
                        <div style="font-size: 12px; color: #94a3b8;">FP8 TFLOPS</div>
                    </div>
                    <div>
                        <div style="font-size: 24px; font-weight: bold; color: #00f5d4;">{p.memory_bandwidth_gbps:.0f}</div>
                        <div style="font-size: 12px; color: #94a3b8;">GB/s Bandwidth</div>
                    </div>
                </div>
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #334155;">
                    <div style="font-size: 12px; color: #64748b;">
                        Architecture: {p.architecture} | SMs: {p.sm_count} | Memory: {p.memory_size_gb:.0f}GB | TDP: {p.tdp_watts:.0f}W
                    </div>
                </div>
            </div>
            """
        
        # Anti-cheat summary
        anticheat_html = f"""
        <div class="card">
            <div class="card-title">🛡️ Anti-Cheat Checks</div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                <div style="text-align: center;">
                    <div style="font-size: 20px; font-weight: bold; color: {status_color(report.jitter_checks_passed == report.jitter_checks_run and report.jitter_checks_run > 0)};">
                        {report.jitter_checks_passed}/{report.jitter_checks_run}
                    </div>
                    <div style="font-size: 12px; color: #94a3b8;">Jitter Checks</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 20px; font-weight: bold; color: {status_color(report.fresh_input_checks_passed == report.fresh_input_checks_run and report.fresh_input_checks_run > 0)};">
                        {report.fresh_input_checks_passed}/{report.fresh_input_checks_run}
                    </div>
                    <div style="font-size: 12px; color: #94a3b8;">Fresh Input Checks</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 20px; font-weight: bold; color: {status_color(report.workload_checks_passed == report.workload_checks_run and report.workload_checks_run > 0)};">
                        {report.workload_checks_passed}/{report.workload_checks_run}
                    </div>
                    <div style="font-size: 12px; color: #94a3b8;">Workload Checks</div>
                </div>
            </div>
            <div style="margin-top: 15px; font-size: 11px; color: #64748b;">
                <strong>Legend:</strong> J = Jitter Check (output changes with perturbed input) | 
                F = Fresh Input Check (output changes with different seed) | 
                W = Workload Invariant (same bytes/tokens/ops)
            </div>
        </div>
        """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{report.title}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: #0f172a;
            color: #ffffff;
            line-height: 1.6;
            padding: 40px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 30px;
            border-bottom: 2px solid #00f5d4;
        }}
        
        .header h1 {{
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #00f5d4, #9d4edd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header .subtitle {{
            font-size: 16px;
            color: #94a3b8;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .summary-card {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        
        .summary-value {{
            font-size: 36px;
            font-weight: 700;
        }}
        
        .summary-label {{
            font-size: 14px;
            color: #94a3b8;
            margin-top: 5px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #334155;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: rgba(255,255,255,0.02);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        th {{
            background: rgba(255,255,255,0.05);
            padding: 15px 12px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        .grid-3 {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
        }}
        
        .card {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 20px;
        }}
        
        .card-title {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 15px;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #334155;
            text-align: center;
            font-size: 12px;
            color: #64748b;
        }}
        
        @media print {{
            body {{ 
                background: white; 
                color: #1e293b;
                padding: 20px;
            }}
            .summary-card, .card {{
                background: #f8fafc;
                border-color: #e2e8f0;
            }}
            th {{
                background: #f1f5f9;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{report.title}</h1>
            <div class="subtitle">Generated on {datetime.now().strftime("%B %d, %Y at %H:%M")}</div>
            {f'<div style="margin-top: 10px; font-size: 14px; color: #64748b;">GPU: {report.gpu_name}</div>' if report.gpu_name else ''}
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-value" style="color: #00f5d4;">{report.total_benchmarks}</div>
                <div class="summary-label">Total Benchmarks</div>
            </div>
            <div class="summary-card">
                <div class="summary-value" style="color: #22c55e;">{report.passed_count}</div>
                <div class="summary-label">Verified & Passed</div>
            </div>
            <div class="summary-card">
                <div class="summary-value" style="color: #ef4444;">{report.failed_count}</div>
                <div class="summary-label">Failed Verification</div>
            </div>
            <div class="summary-card">
                <div class="summary-value" style="color: #f59e0b;">{report.quarantined_count}</div>
                <div class="summary-label">Quarantined</div>
            </div>
            <div class="summary-card">
                <div class="summary-value" style="color: #00f5d4;">{report.coverage_percent:.1f}%</div>
                <div class="summary-label">Verification Coverage</div>
            </div>
        </div>
        
        <div class="section">
            <div class="grid-2">
                {peak_section}
                <div class="card">
                    <div class="card-title">⚠️ Quarantine Breakdown</div>
                    {quarantine_breakdown if quarantine_breakdown else '<div style="color: #22c55e;">✅ No quarantined benchmarks!</div>'}
                </div>
            </div>
        </div>
        
        <div class="section">
            {anticheat_html}
        </div>
        
        <div class="section">
            <h2 class="section-title">📊 Benchmark Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th style="text-align: center;">Verified</th>
                        <th style="text-align: right;">Baseline (ms)</th>
                        <th style="text-align: right;">Optimized (ms)</th>
                        <th style="text-align: right;">Speedup</th>
                        <th style="text-align: center;">Compute Eff.</th>
                        <th style="text-align: center;">Memory Eff.</th>
                        <th style="text-align: center;">Checks</th>
                    </tr>
                </thead>
                <tbody>
                    {benchmark_rows}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>AI Performance Engineering • Verification Report • {report.generated_at}</p>
        </div>
    </div>
</body>
</html>"""
        
        return html


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_verification_report(
    benchmark_results: Union[str, Path, List[Dict[str, Any]]],
    output_path: Union[str, Path],
    format: str = "html",
    gpu_name: str = "",
    title: str = "Benchmark Verification Report",
    quarantine_path: Optional[Path] = None,
) -> Path:
    """
    Convenience function to generate a verification report.
    
    Args:
        benchmark_results: Benchmark results (JSON file path or list of dicts)
        output_path: Output file path
        format: "html" or "json"
        gpu_name: GPU name for theoretical peak lookup
        title: Report title
        quarantine_path: Optional path to quarantine.json
        
    Returns:
        Path to generated report
    """
    # Load data if path
    if isinstance(benchmark_results, (str, Path)):
        with open(benchmark_results) as f:
            data = json.load(f)
        benchmark_results = data.get("benchmarks", data.get("results", []))
        # Try to extract chapter results
        if not benchmark_results and isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    benchmark_results = value
                    break
    
    # Initialize quarantine manager
    qm = None
    if quarantine_path and QuarantineManager:
        qm = QuarantineManager(quarantine_path=quarantine_path)
    
    # Create generator and report
    generator = VerificationReportGenerator(quarantine_manager=qm)
    report = generator.create_report(benchmark_results, gpu_name=gpu_name, title=title)
    
    # Generate output
    output_path = Path(output_path)
    if format == "json":
        return generator.generate_json(report, output_path)
    else:
        return generator.generate_html(report, output_path)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point for verification report generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate verification reports")
    parser.add_argument("input", help="Input JSON file with benchmark results")
    parser.add_argument("-o", "--output", default="verification_report.html", help="Output file path")
    parser.add_argument("-f", "--format", choices=["html", "json"], default="html", help="Output format")
    parser.add_argument("--gpu", default="", help="GPU name for theoretical peak lookup")
    parser.add_argument("--title", default="Benchmark Verification Report", help="Report title")
    parser.add_argument("--quarantine", default=None, help="Path to quarantine.json")
    
    args = parser.parse_args()
    
    quarantine_path = Path(args.quarantine) if args.quarantine else None
    
    output = generate_verification_report(
        args.input,
        args.output,
        format=args.format,
        gpu_name=args.gpu,
        title=args.title,
        quarantine_path=quarantine_path,
    )
    
    print(f"✅ Report generated: {output}")


if __name__ == "__main__":
    main()


