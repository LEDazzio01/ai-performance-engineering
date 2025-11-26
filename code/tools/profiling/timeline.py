"""
Timeline Generator - Create CPU/GPU timeline visualizations.

Features:
- Parallel CPU and GPU operation visualization
- Stream-level GPU activity
- Memory transfer highlighting
- Kernel overlap detection
- Export to Chrome trace format and custom JSON
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


class EventType(Enum):
    """Types of timeline events."""
    CPU_OP = "cpu_op"
    CUDA_KERNEL = "cuda_kernel"
    CUDA_MEMCPY = "cuda_memcpy"
    CUDA_SYNC = "cuda_sync"
    CUDA_RUNTIME = "cuda_runtime"
    PYTHON = "python"


@dataclass
class TimelineEvent:
    """A single event in the timeline."""
    name: str
    event_type: EventType
    start_us: float
    duration_us: float
    thread_id: int = 0
    stream_id: int = 0
    device_id: int = 0
    args: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def end_us(self) -> float:
        return self.start_us + self.duration_us
    
    def to_chrome_trace(self) -> Dict[str, Any]:
        """Convert to Chrome trace format."""
        return {
            "name": self.name,
            "cat": self.event_type.value,
            "ph": "X",  # Complete event
            "ts": self.start_us,
            "dur": self.duration_us,
            "tid": self.thread_id,
            "pid": self.device_id,
            "args": self.args,
        }


@dataclass
class TimelineData:
    """Complete timeline data."""
    events: List[TimelineEvent] = field(default_factory=list)
    cpu_events: List[TimelineEvent] = field(default_factory=list)
    gpu_events: List[TimelineEvent] = field(default_factory=list)
    streams: Dict[int, List[TimelineEvent]] = field(default_factory=dict)
    
    total_time_us: float = 0
    cpu_active_time_us: float = 0
    gpu_active_time_us: float = 0
    overlap_time_us: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "events": [e.__dict__ for e in self.events],
            "summary": {
                "total_time_us": self.total_time_us,
                "cpu_active_time_us": self.cpu_active_time_us,
                "gpu_active_time_us": self.gpu_active_time_us,
                "overlap_time_us": self.overlap_time_us,
                "cpu_utilization": self.cpu_active_time_us / max(self.total_time_us, 1),
                "gpu_utilization": self.gpu_active_time_us / max(self.total_time_us, 1),
            },
            "streams": {
                str(k): len(v) for k, v in self.streams.items()
            },
        }


class TimelineGenerator:
    """
    Generate CPU/GPU timeline visualizations.
    
    Usage:
        generator = TimelineGenerator()
        
        with torch.profiler.profile(...) as prof:
            model(input)
        
        timeline = generator.from_profiler(prof)
        generator.export_chrome_trace(timeline, "trace.json")
    """
    
    def __init__(
        self,
        min_duration_us: float = 1.0,
        include_python_events: bool = True,
        include_cuda_runtime: bool = True,
    ):
        self.min_duration_us = min_duration_us
        self.include_python_events = include_python_events
        self.include_cuda_runtime = include_cuda_runtime
    
    def from_profiler(self, prof) -> TimelineData:
        """
        Generate timeline data from torch.profiler.
        
        Args:
            prof: torch.profiler.profile object
            
        Returns:
            TimelineData with all events
        """
        timeline = TimelineData()
        
        try:
            # Access profiler events
            for event in prof.events():
                if event.duration_ns < self.min_duration_us * 1000:
                    continue
                
                # Determine event type
                event_type = self._classify_event(event)
                
                if event_type == EventType.PYTHON and not self.include_python_events:
                    continue
                if event_type == EventType.CUDA_RUNTIME and not self.include_cuda_runtime:
                    continue
                
                te = TimelineEvent(
                    name=event.name,
                    event_type=event_type,
                    start_us=event.start_ns / 1000,
                    duration_us=event.duration_ns / 1000,
                    thread_id=getattr(event, 'thread', 0),
                    device_id=getattr(event, 'device_index', 0) if hasattr(event, 'device_index') else 0,
                )
                
                timeline.events.append(te)
                
                # Categorize
                if event_type in [EventType.CPU_OP, EventType.PYTHON]:
                    timeline.cpu_events.append(te)
                elif event_type in [EventType.CUDA_KERNEL, EventType.CUDA_MEMCPY]:
                    timeline.gpu_events.append(te)
                    stream = getattr(event, 'stream', 0)
                    if stream not in timeline.streams:
                        timeline.streams[stream] = []
                    timeline.streams[stream].append(te)
            
            # Calculate statistics
            self._calculate_statistics(timeline)
            
        except Exception as e:
            # Add error event
            timeline.events.append(TimelineEvent(
                name=f"Error: {e}",
                event_type=EventType.CPU_OP,
                start_us=0,
                duration_us=1,
            ))
        
        return timeline
    
    def from_chrome_trace(self, trace_path: Path) -> TimelineData:
        """
        Generate timeline data from Chrome trace JSON.
        
        Args:
            trace_path: Path to Chrome trace JSON
            
        Returns:
            TimelineData
        """
        trace_path = Path(trace_path)
        timeline = TimelineData()
        
        with open(trace_path) as f:
            trace = json.load(f)
        
        events = trace if isinstance(trace, list) else trace.get('traceEvents', [])
        
        for event in events:
            if event.get('ph') != 'X':  # Only complete events
                continue
            
            dur = event.get('dur', 0)
            if dur < self.min_duration_us:
                continue
            
            cat = event.get('cat', '')
            event_type = self._classify_category(cat)
            
            te = TimelineEvent(
                name=event.get('name', 'unknown'),
                event_type=event_type,
                start_us=event.get('ts', 0),
                duration_us=dur,
                thread_id=event.get('tid', 0),
                device_id=event.get('pid', 0),
                args=event.get('args', {}),
            )
            
            timeline.events.append(te)
            
            if event_type in [EventType.CPU_OP, EventType.PYTHON]:
                timeline.cpu_events.append(te)
            elif event_type in [EventType.CUDA_KERNEL, EventType.CUDA_MEMCPY]:
                timeline.gpu_events.append(te)
        
        self._calculate_statistics(timeline)
        
        return timeline
    
    def _classify_event(self, event) -> EventType:
        """Classify a profiler event by type."""
        name = event.name.lower()
        
        if hasattr(event, 'device_type'):
            if 'cuda' in str(event.device_type).lower():
                if 'memcpy' in name:
                    return EventType.CUDA_MEMCPY
                elif 'sync' in name:
                    return EventType.CUDA_SYNC
                return EventType.CUDA_KERNEL
        
        if 'cuda' in name:
            if 'memcpy' in name or 'copy' in name:
                return EventType.CUDA_MEMCPY
            elif 'sync' in name:
                return EventType.CUDA_SYNC
            elif 'launch' in name or 'runtime' in name:
                return EventType.CUDA_RUNTIME
            return EventType.CUDA_KERNEL
        
        if 'python' in name or 'py:' in name:
            return EventType.PYTHON
        
        return EventType.CPU_OP
    
    def _classify_category(self, cat: str) -> EventType:
        """Classify event by category string."""
        cat = cat.lower()
        
        if 'kernel' in cat or 'cuda' in cat:
            return EventType.CUDA_KERNEL
        elif 'memcpy' in cat or 'memory' in cat:
            return EventType.CUDA_MEMCPY
        elif 'sync' in cat:
            return EventType.CUDA_SYNC
        elif 'python' in cat:
            return EventType.PYTHON
        elif 'runtime' in cat:
            return EventType.CUDA_RUNTIME
        
        return EventType.CPU_OP
    
    def _calculate_statistics(self, timeline: TimelineData):
        """Calculate timeline statistics."""
        if not timeline.events:
            return
        
        # Total time span
        min_start = min(e.start_us for e in timeline.events)
        max_end = max(e.end_us for e in timeline.events)
        timeline.total_time_us = max_end - min_start
        
        # Active times (handling overlaps within category)
        timeline.cpu_active_time_us = self._calculate_active_time(timeline.cpu_events)
        timeline.gpu_active_time_us = self._calculate_active_time(timeline.gpu_events)
        
        # Calculate CPU/GPU overlap
        timeline.overlap_time_us = self._calculate_overlap(
            timeline.cpu_events, timeline.gpu_events
        )
    
    def _calculate_active_time(self, events: List[TimelineEvent]) -> float:
        """Calculate total active time accounting for overlaps."""
        if not events:
            return 0
        
        # Merge overlapping intervals
        sorted_events = sorted(events, key=lambda e: e.start_us)
        merged = []
        
        for event in sorted_events:
            if merged and event.start_us <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], event.end_us))
            else:
                merged.append((event.start_us, event.end_us))
        
        return sum(end - start for start, end in merged)
    
    def _calculate_overlap(
        self,
        events_a: List[TimelineEvent],
        events_b: List[TimelineEvent],
    ) -> float:
        """Calculate time where both CPU and GPU are active."""
        if not events_a or not events_b:
            return 0
        
        # Simple approximation: check for overlapping intervals
        overlap_time = 0
        
        for a in events_a:
            for b in events_b:
                # Check if intervals overlap
                overlap_start = max(a.start_us, b.start_us)
                overlap_end = min(a.end_us, b.end_us)
                
                if overlap_end > overlap_start:
                    overlap_time += overlap_end - overlap_start
        
        return overlap_time
    
    def export_chrome_trace(
        self,
        timeline: TimelineData,
        path: Path,
    ):
        """Export timeline to Chrome trace format."""
        path = Path(path)
        
        trace_events = [e.to_chrome_trace() for e in timeline.events]
        
        # Add metadata
        trace_events.insert(0, {
            "name": "process_name",
            "ph": "M",
            "pid": 0,
            "args": {"name": "GPU"},
        })
        trace_events.insert(0, {
            "name": "process_name",
            "ph": "M",
            "pid": 1,
            "args": {"name": "CPU"},
        })
        
        path.write_text(json.dumps({"traceEvents": trace_events}, indent=2))
    
    def export_visualization_data(
        self,
        timeline: TimelineData,
        path: Path,
    ):
        """Export data formatted for web visualization."""
        path = Path(path)
        
        # Group events by type for visualization
        data = {
            "summary": {
                "total_time_ms": timeline.total_time_us / 1000,
                "cpu_time_ms": timeline.cpu_active_time_us / 1000,
                "gpu_time_ms": timeline.gpu_active_time_us / 1000,
                "overlap_ms": timeline.overlap_time_us / 1000,
            },
            "cpu_timeline": [
                {
                    "name": e.name,
                    "start_ms": e.start_us / 1000,
                    "duration_ms": e.duration_us / 1000,
                }
                for e in sorted(timeline.cpu_events, key=lambda x: x.start_us)[:500]
            ],
            "gpu_timeline": [
                {
                    "name": e.name,
                    "start_ms": e.start_us / 1000,
                    "duration_ms": e.duration_us / 1000,
                    "stream": e.stream_id,
                }
                for e in sorted(timeline.gpu_events, key=lambda x: x.start_us)[:500]
            ],
            "streams": {
                str(stream_id): [
                    {
                        "name": e.name,
                        "start_ms": e.start_us / 1000,
                        "duration_ms": e.duration_us / 1000,
                    }
                    for e in sorted(events, key=lambda x: x.start_us)[:100]
                ]
                for stream_id, events in timeline.streams.items()
            },
        }
        
        path.write_text(json.dumps(data, indent=2))
    
    def generate_html_viewer(
        self,
        timeline: TimelineData,
        path: Path,
    ):
        """Generate standalone HTML timeline viewer."""
        path = Path(path)
        
        # Prepare data for visualization
        viz_data = {
            "cpu": [
                {"name": e.name, "start": e.start_us, "end": e.end_us}
                for e in timeline.cpu_events[:200]
            ],
            "gpu": [
                {"name": e.name, "start": e.start_us, "end": e.end_us, "stream": e.stream_id}
                for e in timeline.gpu_events[:200]
            ],
            "stats": {
                "total_ms": timeline.total_time_us / 1000,
                "cpu_ms": timeline.cpu_active_time_us / 1000,
                "gpu_ms": timeline.gpu_active_time_us / 1000,
            }
        }
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>CPU/GPU Timeline</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0f172a;
            color: #e2e8f0;
        }}
        h1 {{ font-size: 24px; margin-bottom: 10px; }}
        .stats {{
            display: flex;
            gap: 30px;
            margin-bottom: 20px;
            padding: 15px;
            background: #1e293b;
            border-radius: 8px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3b82f6;
        }}
        .stat-label {{
            font-size: 12px;
            color: #94a3b8;
        }}
        #timeline {{
            background: #1e293b;
            border-radius: 8px;
            padding: 20px;
        }}
        .track-label {{
            font-size: 12px;
            fill: #94a3b8;
        }}
        .cpu-event {{
            fill: #3b82f6;
            opacity: 0.8;
        }}
        .gpu-event {{
            fill: #22c55e;
            opacity: 0.8;
        }}
        .event:hover {{
            opacity: 1;
            stroke: white;
            stroke-width: 1px;
        }}
        .tooltip {{
            position: absolute;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 4px;
            padding: 8px;
            font-size: 12px;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <h1>⏱️ CPU/GPU Timeline</h1>
    <div class="stats">
        <div class="stat-item">
            <div class="stat-value" id="total-time">-</div>
            <div class="stat-label">Total Time</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" id="cpu-time">-</div>
            <div class="stat-label">CPU Active</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" id="gpu-time">-</div>
            <div class="stat-label">GPU Active</div>
        </div>
    </div>
    <div id="timeline"></div>
    <div class="tooltip" style="display: none;"></div>
    
    <script>
        const data = {json.dumps(viz_data)};
        
        // Update stats
        document.getElementById('total-time').textContent = data.stats.total_ms.toFixed(2) + 'ms';
        document.getElementById('cpu-time').textContent = data.stats.cpu_ms.toFixed(2) + 'ms';
        document.getElementById('gpu-time').textContent = data.stats.gpu_ms.toFixed(2) + 'ms';
        
        // Timeline visualization
        const width = document.getElementById('timeline').clientWidth - 40;
        const height = 300;
        const margin = {{top: 30, right: 20, bottom: 30, left: 60}};
        
        const svg = d3.select('#timeline')
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${{margin.left}},${{margin.top}})`);
        
        // Scales
        const allEvents = [...data.cpu, ...data.gpu];
        const xMin = d3.min(allEvents, d => d.start) || 0;
        const xMax = d3.max(allEvents, d => d.end) || 1;
        
        const x = d3.scaleLinear()
            .domain([xMin, xMax])
            .range([0, width]);
        
        // Track labels
        svg.append('text')
            .attr('class', 'track-label')
            .attr('x', -50)
            .attr('y', 50)
            .text('CPU');
        
        svg.append('text')
            .attr('class', 'track-label')
            .attr('x', -50)
            .attr('y', 150)
            .text('GPU');
        
        // CPU events
        svg.selectAll('.cpu-event')
            .data(data.cpu)
            .enter()
            .append('rect')
            .attr('class', 'event cpu-event')
            .attr('x', d => x(d.start))
            .attr('y', 20)
            .attr('width', d => Math.max(1, x(d.end) - x(d.start)))
            .attr('height', 60)
            .attr('rx', 2)
            .on('mouseover', function(event, d) {{
                d3.select('.tooltip')
                    .style('display', 'block')
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 10) + 'px')
                    .html(`<strong>${{d.name}}</strong><br>${{((d.end - d.start) / 1000).toFixed(3)}}ms`);
            }})
            .on('mouseout', function() {{
                d3.select('.tooltip').style('display', 'none');
            }});
        
        // GPU events
        svg.selectAll('.gpu-event')
            .data(data.gpu)
            .enter()
            .append('rect')
            .attr('class', 'event gpu-event')
            .attr('x', d => x(d.start))
            .attr('y', 120)
            .attr('width', d => Math.max(1, x(d.end) - x(d.start)))
            .attr('height', 60)
            .attr('rx', 2)
            .on('mouseover', function(event, d) {{
                d3.select('.tooltip')
                    .style('display', 'block')
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 10) + 'px')
                    .html(`<strong>${{d.name}}</strong><br>${{((d.end - d.start) / 1000).toFixed(3)}}ms`);
            }})
            .on('mouseout', function() {{
                d3.select('.tooltip').style('display', 'none');
            }});
        
        // X axis
        svg.append('g')
            .attr('transform', `translate(0,${{height - 50}})`)
            .call(d3.axisBottom(x).tickFormat(d => (d / 1000).toFixed(1) + 'ms'))
            .selectAll('text')
            .style('fill', '#94a3b8');
        
        svg.selectAll('.domain, .tick line')
            .style('stroke', '#334155');
    </script>
</body>
</html>'''
        
        path.write_text(html)

