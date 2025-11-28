#!/usr/bin/env python3
"""Rich Terminal UI for Benchmark Analysis using curses."""

import curses
import json
from typing import Optional, List, Dict, Any

from tools.analysis.performance_analyzer import (
    PerformanceAnalyzer,
    load_benchmark_data as load_benchmark_results,
)


class BenchmarkTUI:
    """Interactive terminal UI for benchmark analysis."""
    
    MENU_ITEMS = [
        ("🚀 Speed Leaderboard", "speed"),
        ("💾 Memory Leaderboard", "memory"),
        ("⭐ Pareto Frontier", "pareto"),
        ("📜 Summary", "summary"),
        ("🔍 What-If Solver", "whatif"),
        ("🔗 Optimization Stacking", "stacking"),
        ("⚡ Power Efficiency", "power"),
        ("💰 Cost Analysis", "cost"),
        ("📈 Scaling Analysis", "scaling"),
        ("📊 Trade-off Chart", "tradeoff"),
        ("⚙️  Settings", "settings"),
        ("❌ Exit", "exit"),
    ]
    
    def __init__(self, data_path: Optional[str] = None):
        self.current_menu = 0
        self.current_view = "menu"
        self.scroll_offset = 0
        self.data_cache = {}
        self.handler = None
        self.whatif_params = {"vram": "", "latency": "", "memory": ""}
        self.whatif_field = 0
        self.data_path = data_path
        
    def init_handler(self):
        """Initialize the data handler."""
        loader = (lambda: load_benchmark_results(self.data_path)) if self.data_path else load_benchmark_results
        self.handler = PerformanceAnalyzer(loader)
    
    def run(self, stdscr):
        """Main TUI loop."""
        self.stdscr = stdscr
        self.init_handler()
        
        # Setup curses
        curses.curs_set(0)  # Hide cursor
        curses.start_color()
        curses.use_default_colors()
        
        # Define color pairs
        curses.init_pair(1, curses.COLOR_GREEN, -1)   # Success/speed
        curses.init_pair(2, curses.COLOR_CYAN, -1)    # Info/memory
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # Warning/highlight
        curses.init_pair(4, curses.COLOR_RED, -1)     # Error/danger
        curses.init_pair(5, curses.COLOR_MAGENTA, -1) # Accent
        curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Selected
        
        self.COLOR_SUCCESS = curses.color_pair(1)
        self.COLOR_INFO = curses.color_pair(2)
        self.COLOR_WARN = curses.color_pair(3)
        self.COLOR_DANGER = curses.color_pair(4)
        self.COLOR_ACCENT = curses.color_pair(5)
        self.COLOR_SELECTED = curses.color_pair(6)
        
        while True:
            self.stdscr.clear()
            self.draw_header()
            
            if self.current_view == "menu":
                self.draw_menu()
            elif self.current_view == "speed":
                self.draw_leaderboard("speed")
            elif self.current_view == "memory":
                self.draw_leaderboard("memory")
            elif self.current_view == "pareto":
                self.draw_pareto()
            elif self.current_view == "summary":
                self.draw_summary()
            elif self.current_view == "whatif":
                self.draw_whatif()
            elif self.current_view == "stacking":
                self.draw_stacking()
            elif self.current_view == "power":
                self.draw_power()
            elif self.current_view == "cost":
                self.draw_cost()
            elif self.current_view == "scaling":
                self.draw_scaling()
            elif self.current_view == "tradeoff":
                self.draw_tradeoff()
            elif self.current_view == "settings":
                self.draw_settings()
            
            self.draw_footer()
            self.stdscr.refresh()
            
            # Handle input
            key = self.stdscr.getch()
            if not self.handle_input(key):
                break
    
    def draw_header(self):
        """Draw the header bar."""
        h, w = self.stdscr.getmaxyx()
        title = "═══ BENCHMARK ANALYSIS TUI ═══"
        self.stdscr.attron(curses.A_BOLD | self.COLOR_ACCENT)
        self.stdscr.addstr(0, (w - len(title)) // 2, title)
        self.stdscr.attroff(curses.A_BOLD | self.COLOR_ACCENT)
        self.stdscr.addstr(1, 0, "─" * w)
    
    def draw_footer(self):
        """Draw the footer with controls."""
        h, w = self.stdscr.getmaxyx()
        if self.current_view == "menu":
            controls = "↑↓ Navigate │ Enter Select │ q Quit"
        elif self.current_view == "whatif":
            controls = "↑↓ Fields │ Tab Next │ Enter Search │ Esc Back"
        else:
            controls = "↑↓ Scroll │ Esc Back │ q Quit"
        
        self.stdscr.addstr(h-2, 0, "─" * w)
        self.stdscr.attron(self.COLOR_INFO)
        self.stdscr.addstr(h-1, (w - len(controls)) // 2, controls)
        self.stdscr.attroff(self.COLOR_INFO)
    
    def draw_menu(self):
        """Draw the main menu."""
        h, w = self.stdscr.getmaxyx()
        start_y = 4
        
        for i, (label, _) in enumerate(self.MENU_ITEMS):
            y = start_y + i
            if y >= h - 3:
                break
            
            if i == self.current_menu:
                self.stdscr.attron(self.COLOR_SELECTED | curses.A_BOLD)
                self.stdscr.addstr(y, 2, f" ▶ {label} ".ljust(w - 4))
                self.stdscr.attroff(self.COLOR_SELECTED | curses.A_BOLD)
            else:
                self.stdscr.addstr(y, 2, f"   {label}")
    
    def draw_leaderboard(self, board_type: str):
        """Draw speed or memory leaderboard."""
        h, w = self.stdscr.getmaxyx()
        
        if "leaderboards" not in self.data_cache:
            self.data_cache["leaderboards"] = self.handler.get_categorized_leaderboards()
        
        data = self.data_cache["leaderboards"]
        boards = data.get("leaderboards", {})
        board = boards.get(board_type, {})
        entries = board.get("entries", [])
        
        title = "🚀 SPEED LEADERBOARD" if board_type == "speed" else "💾 MEMORY LEADERBOARD"
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(3, 2, title)
        self.stdscr.attroff(curses.A_BOLD)
        
        # Column headers
        self.stdscr.attron(self.COLOR_INFO)
        self.stdscr.addstr(5, 2, f"{'Rank':<6}{'Benchmark':<45}{'Metric':<15}")
        self.stdscr.attroff(self.COLOR_INFO)
        self.stdscr.addstr(6, 2, "─" * 66)
        
        max_rows = h - 10
        visible = entries[self.scroll_offset:self.scroll_offset + max_rows]
        
        for i, entry in enumerate(visible):
            y = 7 + i
            rank = entry.get("rank", i + 1)
            name = entry.get("name", "")[:42]
            metric = entry.get("primary_metric", "")
            
            color = self.COLOR_SUCCESS if rank <= 3 else curses.A_NORMAL
            self.stdscr.attron(color)
            self.stdscr.addstr(y, 2, f"#{rank:<5}{name:<45}{metric:<15}")
            self.stdscr.attroff(color)
        
        # Scroll indicator
        if len(entries) > max_rows:
            pct = int((self.scroll_offset / max(1, len(entries) - max_rows)) * 100)
            self.stdscr.addstr(3, w - 15, f"[{pct:3d}% scrolled]")
    
    def draw_pareto(self):
        """Draw Pareto frontier."""
        h, w = self.stdscr.getmaxyx()
        
        if "pareto" not in self.data_cache:
            self.data_cache["pareto"] = self.handler.get_pareto_frontier()
        
        data = self.data_cache["pareto"]
        frontier = data.get("pareto_frontier", [])
        
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(3, 2, f"⭐ PARETO OPTIMAL ({data.get('pareto_count', 0)} / {data.get('total_count', 0)})")
        self.stdscr.attroff(curses.A_BOLD)
        
        self.stdscr.attron(self.COLOR_INFO)
        self.stdscr.addstr(5, 2, f"{'Benchmark':<40}{'Speedup':<12}{'Memory':<12}")
        self.stdscr.attroff(self.COLOR_INFO)
        self.stdscr.addstr(6, 2, "─" * 64)
        
        max_rows = h - 10
        visible = frontier[self.scroll_offset:self.scroll_offset + max_rows]
        
        for i, entry in enumerate(visible):
            y = 7 + i
            name = entry.get("name", "")[:38]
            speedup = f"{entry.get('speedup', 1):.2f}x"
            mem = f"-{entry.get('memory_savings', 0):.0f}%" if entry.get('memory_savings') else "N/A"
            
            self.stdscr.attron(self.COLOR_SUCCESS)
            self.stdscr.addstr(y, 2, f"⭐ {name:<38}{speedup:<12}{mem:<12}")
            self.stdscr.attroff(self.COLOR_SUCCESS)
    
    def draw_summary(self):
        """Draw quick summary stats."""
        h, _ = self.stdscr.getmaxyx()
        data = load_benchmark_results(self.data_path)
        summary = data.get("summary", {})
        total = summary.get("total_benchmarks", len(data.get("benchmarks", [])))
        avg_speedup = summary.get("avg_speedup", 0)
        max_speedup = summary.get("max_speedup", 0)
        successful = summary.get("successful")
        failed = summary.get("failed")
        memory_optimizations = summary.get("memory_optimizations")
        speed_optimizations = summary.get("speed_optimizations")
        
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(3, 2, "📜 BENCHMARK SUMMARY")
        self.stdscr.attroff(curses.A_BOLD)
        
        lines = [
            f"Total benchmarks: {total}",
            f"Average speedup:  {avg_speedup:.2f}x",
            f"Max speedup:      {max_speedup:.2f}x",
        ]
        if successful is not None:
            lines.append(f"Successful:       {successful} | Failed: {failed or 0}")
        if memory_optimizations is not None:
            lines.append(f"Memory-focused:   {memory_optimizations} | Speed-focused: {speed_optimizations or 0}")
        lines.append(f"Timestamp:        {data.get('timestamp', 'N/A')}")
        
        for idx, line in enumerate(lines):
            if 6 + idx >= h - 2:
                break
            self.stdscr.addstr(5 + idx, 2, line)
    
    def draw_whatif(self):
        """Draw What-If solver with input fields."""
        h, w = self.stdscr.getmaxyx()
        
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(3, 2, "🔍 WHAT-IF CONSTRAINT SOLVER")
        self.stdscr.attroff(curses.A_BOLD)
        
        fields = [
            ("Max VRAM (GB):", "vram"),
            ("Max Latency (ms):", "latency"),
            ("Memory Budget (GB):", "memory"),
        ]
        
        for i, (label, key) in enumerate(fields):
            y = 5 + i * 2
            is_selected = i == self.whatif_field
            
            self.stdscr.addstr(y, 4, label)
            
            value = self.whatif_params.get(key, "")
            box = f"[{value:<15}]"
            
            if is_selected:
                self.stdscr.attron(self.COLOR_SELECTED)
            self.stdscr.addstr(y, 25, box)
            if is_selected:
                self.stdscr.attroff(self.COLOR_SELECTED)
        
        # Search button
        y = 5 + len(fields) * 2
        btn = "[ Search ]"
        if self.whatif_field == len(fields):
            self.stdscr.attron(self.COLOR_SELECTED | curses.A_BOLD)
        self.stdscr.addstr(y, 25, btn)
        if self.whatif_field == len(fields):
            self.stdscr.attroff(self.COLOR_SELECTED | curses.A_BOLD)
        
        # Results
        if "whatif_results" in self.data_cache:
            results = self.data_cache["whatif_results"]
            y = 13
            self.stdscr.addstr(y, 2, "─" * 60)
            y += 1
            self.stdscr.attron(self.COLOR_SUCCESS)
            self.stdscr.addstr(y, 2, f"Matching: {results.get('matching_count', 0)} / {results.get('total_benchmarks', 0)}")
            self.stdscr.attroff(self.COLOR_SUCCESS)
            
            y += 2
            for r in results.get("recommendations", [])[:8]:
                if y >= h - 4:
                    break
                self.stdscr.addstr(y, 4, f"• {r['name'][:40]}: {r['speedup']:.2f}x")
                y += 1
    
    def draw_stacking(self):
        """Draw optimization stacking guide."""
        h, w = self.stdscr.getmaxyx()
        
        if "stacking" not in self.data_cache:
            self.data_cache["stacking"] = self.handler.get_optimization_stacking()
        
        data = self.data_cache["stacking"]
        
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(3, 2, "🔗 OPTIMIZATION STACKING GUIDE")
        self.stdscr.attroff(curses.A_BOLD)
        
        y = 5
        self.stdscr.attron(self.COLOR_SUCCESS | curses.A_BOLD)
        self.stdscr.addstr(y, 2, "✅ Compatible Combinations:")
        self.stdscr.attroff(self.COLOR_SUCCESS | curses.A_BOLD)
        y += 1
        
        for c in data.get("compatible_combinations", [])[:5]:
            if y >= h - 10:
                break
            self.stdscr.addstr(y, 4, f"{c['opt1']} + {c['opt2']}")
            self.stdscr.attron(self.COLOR_INFO)
            self.stdscr.addstr(y + 1, 6, f"→ {c['benefit'][:60]}")
            self.stdscr.attroff(self.COLOR_INFO)
            y += 2
        
        y += 1
        self.stdscr.attron(self.COLOR_DANGER | curses.A_BOLD)
        self.stdscr.addstr(y, 2, "❌ Incompatible:")
        self.stdscr.attroff(self.COLOR_DANGER | curses.A_BOLD)
        y += 1
        
        for c in data.get("incompatible_combinations", [])[:3]:
            if y >= h - 4:
                break
            self.stdscr.addstr(y, 4, f"{c['opt1']} + {c['opt2']}: {c['reason'][:50]}")
            y += 1
    
    def draw_power(self):
        """Draw power efficiency analysis."""
        h, w = self.stdscr.getmaxyx()
        
        if "power" not in self.data_cache:
            self.data_cache["power"] = self.handler.get_power_efficiency()
        
        data = self.data_cache["power"]
        
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(3, 2, f"⚡ POWER EFFICIENCY (ops/watt)")
        self.stdscr.attroff(curses.A_BOLD)
        
        self.stdscr.addstr(4, 2, f"Avg Power: {data.get('avg_power_w', 0):.0f}W │ Benchmarks: {data.get('total_benchmarks_with_power', 0)}")
        
        if data.get("most_efficient"):
            e = data["most_efficient"]
            self.stdscr.attron(self.COLOR_WARN)
            self.stdscr.addstr(6, 2, f"🏆 Most Efficient: {e['name']}")
            self.stdscr.addstr(7, 4, f"{e['ops_per_watt']:.2f} ops/watt │ {e['power_w']:.0f}W │ {e['speedup']:.2f}x")
            self.stdscr.attroff(self.COLOR_WARN)
        
        self.stdscr.addstr(9, 2, "─" * 60)
        
        max_rows = h - 13
        rankings = data.get("efficiency_rankings", [])[self.scroll_offset:self.scroll_offset + max_rows]
        
        for i, e in enumerate(rankings):
            y = 10 + i
            self.stdscr.addstr(y, 2, f"{i+1+self.scroll_offset:2}. {e['name'][:40]:<42} {e['ops_per_watt']:.2f}")
    
    def draw_cost(self):
        """Draw cost analysis."""
        h, w = self.stdscr.getmaxyx()
        
        if "cost" not in self.data_cache:
            self.data_cache["cost"] = self.handler.get_cost_analysis()
        
        data = self.data_cache["cost"]
        
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(3, 2, "💰 COST ANALYSIS")
        self.stdscr.attroff(curses.A_BOLD)
        
        self.stdscr.addstr(4, 2, f"GPU: {data.get('assumed_gpu', 'B200')} @ ${data.get('hourly_rate', 5):.2f}/hr")
        
        if data.get("highest_savings"):
            h_data = data["highest_savings"]
            self.stdscr.attron(self.COLOR_SUCCESS)
            self.stdscr.addstr(6, 2, f"🏆 Highest Savings: {h_data['name']}")
            self.stdscr.addstr(7, 4, f"${h_data['baseline_cost_per_m']:.4f} → ${h_data['optimized_cost_per_m']:.4f} per 1M ops ({h_data['savings_pct']:.0f}%)")
            self.stdscr.attroff(self.COLOR_SUCCESS)
        
        self.stdscr.addstr(9, 2, "─" * 60)
        self.stdscr.attron(self.COLOR_INFO)
        self.stdscr.addstr(10, 2, f"{'Benchmark':<45}{'Savings':<10}")
        self.stdscr.attroff(self.COLOR_INFO)
        
        max_rows = h - 14
        rankings = data.get("cost_rankings", [])[self.scroll_offset:self.scroll_offset + max_rows]
        
        for i, c in enumerate(rankings):
            y = 11 + i
            self.stdscr.addstr(y, 2, f"{c['name'][:43]:<45}{c['savings_pct']:.0f}%")
    
    def draw_scaling(self):
        """Draw scaling analysis."""
        h, w = self.stdscr.getmaxyx()
        
        if "scaling" not in self.data_cache:
            self.data_cache["scaling"] = self.handler.get_scaling_analysis()
        
        data = self.data_cache["scaling"]
        
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(3, 2, "📈 SCALING ANALYSIS")
        self.stdscr.attroff(curses.A_BOLD)
        
        self.stdscr.attron(self.COLOR_INFO)
        self.stdscr.addstr(5, 2, f"💡 {data.get('key_insight', '')[:70]}")
        self.stdscr.attroff(self.COLOR_INFO)
        
        y = 7
        for r in data.get("scaling_recommendations", []):
            if y >= h - 5:
                break
            self.stdscr.attron(self.COLOR_WARN | curses.A_BOLD)
            self.stdscr.addstr(y, 2, r["factor"])
            self.stdscr.attroff(self.COLOR_WARN | curses.A_BOLD)
            self.stdscr.addstr(y + 1, 4, r["insight"][:70])
            self.stdscr.attron(self.COLOR_SUCCESS)
            self.stdscr.addstr(y + 2, 4, f"→ {r['recommendation'][:66]}")
            self.stdscr.attroff(self.COLOR_SUCCESS)
            y += 4
    
    def draw_tradeoff(self):
        """Draw ASCII trade-off scatter chart."""
        h, w = self.stdscr.getmaxyx()
        
        if "pareto" not in self.data_cache:
            self.data_cache["pareto"] = self.handler.get_pareto_frontier()
        
        data = self.data_cache["pareto"]
        
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(3, 2, "📊 SPEED vs MEMORY TRADE-OFF")
        self.stdscr.attroff(curses.A_BOLD)
        
        # Chart dimensions
        chart_h = min(20, h - 10)
        chart_w = min(60, w - 10)
        start_y = 5
        start_x = 8
        
        # Draw axes
        for y in range(chart_h):
            self.stdscr.addstr(start_y + y, start_x - 1, "│")
        self.stdscr.addstr(start_y + chart_h, start_x - 1, "└" + "─" * chart_w)
        
        # Y-axis label
        self.stdscr.addstr(start_y, 2, "Mem%")
        self.stdscr.addstr(start_y + chart_h - 1, 2, "0%")
        
        # X-axis label  
        self.stdscr.addstr(start_y + chart_h + 1, start_x + chart_w // 2 - 3, "Speedup →")
        
        # Plot points
        points = data.get("pareto_frontier", []) + data.get("non_pareto", [])[:50]
        
        max_speedup = max((p.get("speedup", 1) for p in points), default=10)
        max_mem = max((p.get("memory_savings", 0) for p in points), default=100)
        
        for p in points:
            speedup = p.get("speedup", 1)
            mem = p.get("memory_savings", 0)
            
            x = int((speedup / max(max_speedup, 1)) * (chart_w - 2)) + start_x
            y = start_y + chart_h - 1 - int((mem / max(max_mem, 1)) * (chart_h - 2))
            
            x = max(start_x, min(x, start_x + chart_w - 1))
            y = max(start_y, min(y, start_y + chart_h - 1))
            
            is_pareto = p in data.get("pareto_frontier", [])
            char = "★" if is_pareto else "·"
            color = self.COLOR_WARN if is_pareto else self.COLOR_INFO
            
            try:
                self.stdscr.attron(color)
                self.stdscr.addstr(y, x, char)
                self.stdscr.attroff(color)
            except:
                pass
        
        # Legend
        self.stdscr.addstr(start_y + chart_h + 2, start_x, "★ = Pareto optimal   · = Other")
    
    def draw_settings(self):
        """Draw settings panel (GPU pricing config)."""
        h, w = self.stdscr.getmaxyx()
        
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(3, 2, "⚙️  SETTINGS")
        self.stdscr.attroff(curses.A_BOLD)
        
        # GPU pricing
        gpu_pricing = {
            "B200": 5.00,
            "H100": 3.50,
            "A100": 2.00,
            "L40S": 1.50,
            "A10G": 1.00,
        }
        
        self.stdscr.addstr(5, 2, "GPU Pricing ($/hr):")
        y = 7
        for gpu, price in gpu_pricing.items():
            self.stdscr.addstr(y, 4, f"{gpu}: ${price:.2f}")
            y += 1
        
        self.stdscr.attron(self.COLOR_INFO)
        self.stdscr.addstr(y + 1, 2, "Note: Edit tools/dashboard/server.py to modify pricing")
        self.stdscr.attroff(self.COLOR_INFO)
    
    def handle_input(self, key: int) -> bool:
        """Handle keyboard input. Return False to exit."""
        if key == ord('q') or key == ord('Q'):
            return False
        
        if self.current_view == "menu":
            return self.handle_menu_input(key)
        elif self.current_view == "whatif":
            return self.handle_whatif_input(key)
        else:
            return self.handle_view_input(key)
    
    def handle_menu_input(self, key: int) -> bool:
        """Handle input in menu view."""
        if key == curses.KEY_UP:
            self.current_menu = max(0, self.current_menu - 1)
        elif key == curses.KEY_DOWN:
            self.current_menu = min(len(self.MENU_ITEMS) - 1, self.current_menu + 1)
        elif key in (curses.KEY_ENTER, 10, 13):
            _, action = self.MENU_ITEMS[self.current_menu]
            if action == "exit":
                return False
            self.current_view = action
            self.scroll_offset = 0
        return True
    
    def handle_whatif_input(self, key: int) -> bool:
        """Handle input in What-If view."""
        if key == 27:  # Escape
            self.current_view = "menu"
            return True
        
        fields = ["vram", "latency", "memory"]
        
        if key == curses.KEY_UP:
            self.whatif_field = max(0, self.whatif_field - 1)
        elif key == curses.KEY_DOWN or key == 9:  # Down or Tab
            self.whatif_field = min(len(fields), self.whatif_field + 1)
        elif key in (curses.KEY_ENTER, 10, 13):
            if self.whatif_field == len(fields):  # Search button
                self.run_whatif_search()
        elif key == curses.KEY_BACKSPACE or key == 127:
            if self.whatif_field < len(fields):
                field = fields[self.whatif_field]
                self.whatif_params[field] = self.whatif_params[field][:-1]
        elif 48 <= key <= 57 or key == 46:  # Numbers and decimal
            if self.whatif_field < len(fields):
                field = fields[self.whatif_field]
                if len(self.whatif_params[field]) < 10:
                    self.whatif_params[field] += chr(key)
        return True
    
    def run_whatif_search(self):
        """Execute What-If search."""
        params = {}
        if self.whatif_params["vram"]:
            params["vram"] = [self.whatif_params["vram"]]
        if self.whatif_params["latency"]:
            params["latency"] = [self.whatif_params["latency"]]
        if self.whatif_params["memory"]:
            params["memory_budget"] = [self.whatif_params["memory"]]
        
        self.data_cache["whatif_results"] = self.handler.get_whatif_recommendations(params)
    
    def handle_view_input(self, key: int) -> bool:
        """Handle input in data views."""
        if key == 27:  # Escape
            self.current_view = "menu"
            self.scroll_offset = 0
            return True
        
        if key == curses.KEY_UP:
            self.scroll_offset = max(0, self.scroll_offset - 1)
        elif key == curses.KEY_DOWN:
            self.scroll_offset += 1
        elif key == curses.KEY_PPAGE:  # Page Up
            self.scroll_offset = max(0, self.scroll_offset - 10)
        elif key == curses.KEY_NPAGE:  # Page Down
            self.scroll_offset += 10
        
        return True


def run_tui(data_file: Optional[str] = None):
    """Entry point for TUI."""
    tui = BenchmarkTUI(data_file)
    curses.wrapper(tui.run)


if __name__ == "__main__":
    run_tui()
