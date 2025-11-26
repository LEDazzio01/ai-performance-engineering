"""
Flame Graph Generator - Create flame graph visualizations from profiling data.

Outputs data compatible with d3-flame-graph for web visualization.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class FlameNode:
    """A node in the flame graph tree."""
    name: str
    value: float  # Time in microseconds
    children: List['FlameNode'] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "children": [c.to_dict() for c in self.children]
        }
    
    def add_child(self, child: 'FlameNode') -> 'FlameNode':
        # Check if child with same name exists
        for existing in self.children:
            if existing.name == child.name:
                existing.value += child.value
                return existing
        self.children.append(child)
        return child


class FlameGraphGenerator:
    """
    Generate flame graph data from torch.profiler traces.
    
    Usage:
        generator = FlameGraphGenerator()
        
        with torch.profiler.profile(...) as prof:
            model(input)
        
        flame_data = generator.from_profiler(prof)
        generator.export(flame_data, "flame.json")
    """
    
    def __init__(
        self,
        min_duration_us: float = 10.0,
        max_depth: int = 50,
        group_small_kernels: bool = True,
        small_kernel_threshold_pct: float = 1.0,
    ):
        self.min_duration_us = min_duration_us
        self.max_depth = max_depth
        self.group_small_kernels = group_small_kernels
        self.small_kernel_threshold_pct = small_kernel_threshold_pct
    
    def from_profiler(
        self,
        prof,
        root_name: str = "GPU Execution",
    ) -> Dict[str, Any]:
        """
        Generate flame graph data from torch.profiler.
        
        Args:
            prof: torch.profiler.profile object
            root_name: Name for the root node
            
        Returns:
            Flame graph data dict compatible with d3-flame-graph
        """
        root = FlameNode(name=root_name, value=0)
        
        try:
            # Get events with stack information
            events = prof.key_averages(group_by_stack_n=self.max_depth)
            
            total_time = sum(e.cuda_time_total for e in events if e.cuda_time_total > 0)
            small_threshold = total_time * (self.small_kernel_threshold_pct / 100)
            
            small_kernels_time = 0
            
            for event in events:
                if event.cuda_time_total < self.min_duration_us:
                    continue
                
                # Group small kernels
                if self.group_small_kernels and event.cuda_time_total < small_threshold:
                    small_kernels_time += event.cuda_time_total
                    continue
                
                # Build path from stack
                if hasattr(event, 'stack') and event.stack:
                    self._add_event_to_tree(root, event)
                else:
                    # No stack info - add directly under root
                    kernel_name = self._clean_kernel_name(event.key)
                    child = FlameNode(
                        name=kernel_name,
                        value=event.cuda_time_total
                    )
                    root.add_child(child)
            
            # Add grouped small kernels
            if small_kernels_time > 0:
                root.add_child(FlameNode(
                    name="[small kernels]",
                    value=small_kernels_time
                ))
            
            # Calculate root value
            root.value = sum(c.value for c in root.children)
            
        except Exception as e:
            root.children.append(FlameNode(
                name=f"Error: {e}",
                value=1
            ))
        
        return root.to_dict()
    
    def from_chrome_trace(
        self,
        trace_path: Path,
        root_name: str = "GPU Execution",
    ) -> Dict[str, Any]:
        """
        Generate flame graph from Chrome trace JSON.
        
        Args:
            trace_path: Path to Chrome trace JSON file
            root_name: Name for root node
            
        Returns:
            Flame graph data dict
        """
        trace_path = Path(trace_path)
        
        with open(trace_path) as f:
            trace = json.load(f)
        
        root = FlameNode(name=root_name, value=0)
        
        # Chrome trace format
        events = trace if isinstance(trace, list) else trace.get('traceEvents', [])
        
        # Process complete events (ph: 'X')
        for event in events:
            if event.get('ph') != 'X':
                continue
            
            name = event.get('name', 'unknown')
            dur = event.get('dur', 0)  # Duration in microseconds
            cat = event.get('cat', '')
            
            if dur < self.min_duration_us:
                continue
            
            # Use category as parent if available
            if cat and cat != name:
                parent = root.add_child(FlameNode(name=cat, value=0))
                parent.add_child(FlameNode(name=name, value=dur))
            else:
                root.add_child(FlameNode(name=name, value=dur))
        
        # Recalculate values
        self._recalculate_values(root)
        
        return root.to_dict()
    
    def from_kernel_list(
        self,
        kernels: List[Dict[str, Any]],
        root_name: str = "GPU Execution",
    ) -> Dict[str, Any]:
        """
        Generate flame graph from a list of kernel timings.
        
        Args:
            kernels: List of dicts with 'name' and 'time_us' keys
            root_name: Name for root node
            
        Returns:
            Flame graph data dict
        """
        root = FlameNode(name=root_name, value=0)
        
        # Group by kernel type
        kernel_groups: Dict[str, List[Dict]] = {}
        
        for kernel in kernels:
            name = kernel.get('name', 'unknown')
            
            # Extract kernel type (e.g., 'gemm' from 'volta_h884gemm_128x128')
            kernel_type = self._extract_kernel_type(name)
            
            if kernel_type not in kernel_groups:
                kernel_groups[kernel_type] = []
            kernel_groups[kernel_type].append(kernel)
        
        # Build tree
        for kernel_type, group_kernels in kernel_groups.items():
            type_node = FlameNode(
                name=kernel_type,
                value=sum(k.get('time_us', 0) for k in group_kernels)
            )
            
            for kernel in group_kernels:
                type_node.children.append(FlameNode(
                    name=kernel.get('name', 'unknown'),
                    value=kernel.get('time_us', 0)
                ))
            
            root.add_child(type_node)
        
        root.value = sum(c.value for c in root.children)
        
        return root.to_dict()
    
    def _add_event_to_tree(self, root: FlameNode, event) -> None:
        """Add a profiler event to the flame tree using its stack."""
        current = root
        
        # Process stack frames (bottom to top)
        if hasattr(event, 'stack') and event.stack:
            for frame in reversed(event.stack[:self.max_depth]):
                frame_name = self._clean_frame_name(str(frame))
                
                # Find or create child
                child = None
                for existing in current.children:
                    if existing.name == frame_name:
                        child = existing
                        break
                
                if child is None:
                    child = FlameNode(name=frame_name, value=0)
                    current.children.append(child)
                
                current = child
        
        # Add kernel at leaf
        kernel_name = self._clean_kernel_name(event.key)
        leaf = FlameNode(name=kernel_name, value=event.cuda_time_total)
        current.add_child(leaf)
    
    def _clean_kernel_name(self, name: str) -> str:
        """Clean up kernel name for display."""
        if not name:
            return "unknown"
        
        # Remove namespace prefixes
        if '::' in name:
            name = name.split('::')[-1]
        
        # Truncate very long names
        if len(name) > 60:
            name = name[:57] + "..."
        
        return name
    
    def _clean_frame_name(self, frame: str) -> str:
        """Clean up stack frame name."""
        # Get just the first line
        frame = frame.split('\n')[0]
        
        # Remove file paths, keep function name
        if '/' in frame:
            parts = frame.split('/')
            frame = parts[-1] if parts else frame
        
        # Truncate
        if len(frame) > 50:
            frame = frame[:47] + "..."
        
        return frame
    
    def _extract_kernel_type(self, name: str) -> str:
        """Extract kernel type from kernel name."""
        name_lower = name.lower()
        
        if 'gemm' in name_lower or 'matmul' in name_lower:
            return "Matrix Multiply"
        elif 'conv' in name_lower:
            return "Convolution"
        elif 'softmax' in name_lower:
            return "Softmax"
        elif 'relu' in name_lower or 'gelu' in name_lower or 'silu' in name_lower:
            return "Activation"
        elif 'norm' in name_lower or 'layernorm' in name_lower or 'batchnorm' in name_lower:
            return "Normalization"
        elif 'attention' in name_lower:
            return "Attention"
        elif 'elementwise' in name_lower or 'pointwise' in name_lower:
            return "Elementwise"
        elif 'reduce' in name_lower or 'sum' in name_lower:
            return "Reduction"
        elif 'copy' in name_lower or 'memcpy' in name_lower:
            return "Memory Copy"
        elif 'fill' in name_lower or 'zero' in name_lower:
            return "Memory Fill"
        else:
            return "Other"
    
    def _recalculate_values(self, node: FlameNode) -> float:
        """Recursively recalculate node values from children."""
        if not node.children:
            return node.value
        
        child_sum = sum(self._recalculate_values(c) for c in node.children)
        node.value = max(node.value, child_sum)
        return node.value
    
    def export(
        self,
        data: Dict[str, Any],
        path: Path,
        format: str = "json",
    ) -> None:
        """
        Export flame graph data to file.
        
        Args:
            data: Flame graph data dict
            path: Output path
            format: Output format ('json' or 'html')
        """
        path = Path(path)
        
        if format == "json":
            path.write_text(json.dumps(data, indent=2))
        
        elif format == "html":
            html = self._generate_html(data)
            path.write_text(html)
    
    def _generate_html(self, data: Dict[str, Any]) -> str:
        """Generate standalone HTML flame graph viewer."""
        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GPU Flame Graph</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0f172a;
            color: #e2e8f0;
        }}
        h1 {{
            margin-bottom: 20px;
            font-size: 24px;
        }}
        #chart {{
            width: 100%;
            height: calc(100vh - 100px);
        }}
        .d3-flame-graph rect {{
            stroke: #1e293b;
            stroke-width: 1px;
        }}
        .d3-flame-graph-tip {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 4px;
            padding: 8px;
            color: #e2e8f0;
        }}
    </style>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3-flame-graph@4.1.3/dist/d3-flamegraph.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/d3-flame-graph@4.1.3/dist/d3-flamegraph.css">
</head>
<body>
    <h1>🔥 GPU Flame Graph</h1>
    <div id="chart"></div>
    <script>
        const data = {json.dumps(data)};
        
        const chart = flamegraph()
            .width(document.getElementById('chart').clientWidth)
            .cellHeight(20)
            .transitionDuration(300)
            .minFrameSize(2)
            .transitionEase(d3.easeCubic)
            .sort(true)
            .title("")
            .selfValue(false)
            .inverted(false)
            .setColorMapper(function(d) {{
                const colors = [
                    '#f97316', '#f59e0b', '#84cc16', '#22c55e',
                    '#14b8a6', '#06b6d4', '#3b82f6', '#8b5cf6',
                    '#d946ef', '#ec4899'
                ];
                return colors[Math.abs(d.data.name.split('').reduce((a, b) => a + b.charCodeAt(0), 0)) % colors.length];
            }});
        
        d3.select("#chart")
            .datum(data)
            .call(chart);
        
        // Resize handling
        window.addEventListener('resize', function() {{
            chart.width(document.getElementById('chart').clientWidth);
            d3.select("#chart").datum(data).call(chart);
        }});
    </script>
</body>
</html>'''

