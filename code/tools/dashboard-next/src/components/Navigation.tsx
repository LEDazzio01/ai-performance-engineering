'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import Image from 'next/image';
import { cn } from '@/lib/utils';
import { Benchmark } from '@/types';
import {
  BarChart3,
  GitCompare,
  Brain,
  Cpu,
  Flame,
  HardDrive,
  Zap,
  Microscope,
  Rocket,
  PieChart,
  Sparkles,
  Network,
  Server,
  Gamepad2,
  Gauge,
  History,
  Package,
  Bell,
  Palette,
  RefreshCw,
  Command,
  Settings,
  Volume2,
  VolumeX,
  Timer,
  Search,
  Target,
  Table,
  Keyboard,
} from 'lucide-react';

export const tabs = [
  { id: 'overview', label: 'Overview', icon: BarChart3, shortcut: '1' },
  { id: 'compare', label: 'Compare', icon: GitCompare, shortcut: '2' },
  { id: 'insights', label: 'LLM Insights', icon: Brain, shortcut: '3' },
  { id: 'roofline', label: 'Roofline', icon: Cpu, shortcut: '4' },
  { id: 'profiler', label: 'Profiler', icon: Flame, shortcut: '5' },
  { id: 'memory', label: 'Memory', icon: HardDrive, shortcut: '6' },
  { id: 'compile', label: 'Compile', icon: Zap, gradient: 'from-yellow-500/20 to-orange-500/20' },
  { id: 'deepprofile', label: 'Deep Profile', icon: Microscope, gradient: 'from-blue-500/20 to-cyan-500/20' },
  { id: 'liveopt', label: 'Live Optimizer', icon: Rocket, gradient: 'from-green-500/20 to-emerald-500/20' },
  { id: 'analysis', label: 'Analysis', icon: PieChart },
  { id: 'advanced', label: 'Advanced', icon: Sparkles, gradient: 'from-purple-500/20 to-blue-500/20' },
  { id: 'multigpu', label: 'Multi-GPU', icon: Network, gradient: 'from-green-500/20 to-teal-500/20' },
  { id: 'distributed', label: 'Distributed', icon: Server, gradient: 'from-red-500/20 to-orange-500/20' },
  { id: 'rlhf', label: 'RL/RLHF', icon: Gamepad2, gradient: 'from-purple-500/20 to-pink-500/20' },
  { id: 'inference', label: 'Inference', icon: Gauge, gradient: 'from-sky-500/20 to-cyan-500/20' },
  { id: 'history', label: 'History', icon: History },
  { id: 'batchopt', label: 'Batch Opt', icon: Package },
  { id: 'webhooks', label: 'Webhooks', icon: Bell },
  { id: 'microbench', label: 'Microbench', icon: Timer, gradient: 'from-amber-500/20 to-orange-500/20' },
  { id: 'themes', label: 'Themes', icon: Palette },
];

interface NavigationProps {
  activeTab: string;
  onTabChange: (tabId: string) => void;
  onRefresh: () => void;
  isRefreshing: boolean;
  onOpenShortcuts?: () => void;
  onOpenRun?: () => void;
  onOpenTargets?: () => void;
  onOpenMatrix?: () => void;
  onOpenFocus?: () => void;
  onToggleAutoRefresh?: () => void;
  autoRefresh?: boolean;
  benchmarks?: Benchmark[];
}

export function Navigation({
  activeTab,
  onTabChange,
  onRefresh,
  isRefreshing,
  onOpenShortcuts,
  onOpenRun,
  onOpenTargets,
  onOpenMatrix,
  onOpenFocus,
  onToggleAutoRefresh,
  autoRefresh,
  benchmarks = [],
}: NavigationProps) {
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [commandQuery, setCommandQuery] = useState('');
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [timerRunning, setTimerRunning] = useState(false);

  // Timer effect
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (timerRunning) {
      interval = setInterval(() => setElapsedTime((t) => t + 1), 1000);
    }
    return () => clearInterval(interval);
  }, [timerRunning]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Command palette
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setShowCommandPalette(true);
      }
      // Close on escape
      if (e.key === 'Escape') {
        setShowCommandPalette(false);
      }
      // Tab shortcuts 1-9
      if (!showCommandPalette && e.key >= '1' && e.key <= '9') {
        const idx = parseInt(e.key) - 1;
        if (tabs[idx]) {
          onTabChange(tabs[idx].id);
        }
      }
      // Refresh with R
      if (!showCommandPalette && e.key === 'r' && !e.metaKey && !e.ctrlKey) {
        onRefresh();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showCommandPalette, onTabChange, onRefresh]);

  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const filteredTabs = tabs.filter(
    (tab) =>
      tab.label.toLowerCase().includes(commandQuery.toLowerCase()) ||
      tab.id.toLowerCase().includes(commandQuery.toLowerCase())
  );

  const filteredBenchmarks = useMemo(() => {
    if (!commandQuery) return [];
    return benchmarks
      .filter(
        (b) =>
          b.name.toLowerCase().includes(commandQuery.toLowerCase()) ||
          b.chapter.toLowerCase().includes(commandQuery.toLowerCase())
      )
      .slice(0, 6);
  }, [benchmarks, commandQuery]);

  const actionCommands = [
    onOpenFocus && { label: 'Focus Mode', action: onOpenFocus, icon: Target },
    onOpenRun && { label: 'Run Benchmark', action: onOpenRun, icon: Gauge },
    onOpenTargets && { label: 'Performance Targets', action: onOpenTargets, icon: Keyboard },
    onOpenMatrix && { label: 'Comparison Matrix', action: onOpenMatrix, icon: Table },
    onToggleAutoRefresh && {
      label: autoRefresh ? 'Auto-Refresh Off' : 'Auto-Refresh On',
      action: onToggleAutoRefresh,
      icon: RefreshCw,
    },
    onOpenShortcuts && { label: 'Keyboard Shortcuts', action: onOpenShortcuts, icon: Command },
  ].filter(Boolean) as { label: string; action: () => void; icon: any }[];

  return (
    <>
      <nav className="fixed top-0 left-0 right-0 z-50 bg-brand-bg/90 backdrop-blur-xl border-b border-white/5">
        <div className="px-4 lg:px-6">
          {/* Top row - brand and actions */}
          <div className="flex items-center justify-between h-14 border-b border-white/5">
            {/* Brand */}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl overflow-hidden">
                <Image
                  src="/ai_sys_perf_engg_cover_cheetah_sm.png"
                  alt="AI Systems Performance"
                  width={40}
                  height={40}
                  className="w-full h-full object-cover"
                  priority
                />
              </div>
              <div className="hidden sm:block">
                <h1 className="text-lg font-bold text-accent-primary">
                  AI Systems Performance
                </h1>
                <p className="text-xs text-white/50">OPTIMIZATION DASHBOARD</p>
              </div>
            </div>

            {/* Right side actions */}
            <div className="flex items-center gap-2">
              {/* Timer */}
              <button
                onClick={() => setTimerRunning(!timerRunning)}
                className={cn(
                  'flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-mono transition-all',
                  timerRunning
                    ? 'bg-accent-success/20 text-accent-success border border-accent-success/30'
                    : 'bg-white/5 text-white/60 hover:text-white'
                )}
              >
                <Timer className="w-4 h-4" />
                {formatTime(elapsedTime)}
              </button>

              {/* Audio toggle */}
              <button
                onClick={() => setAudioEnabled(!audioEnabled)}
                className={cn(
                  'p-2 rounded-lg transition-all',
                  audioEnabled
                    ? 'bg-accent-primary/20 text-accent-primary'
                    : 'text-white/40 hover:text-white hover:bg-white/5'
                )}
              >
                {audioEnabled ? <Volume2 className="w-5 h-5" /> : <VolumeX className="w-5 h-5" />}
              </button>

              {/* Command palette */}
              <button
                onClick={() => setShowCommandPalette(true)}
                className="p-2 rounded-lg text-white/40 hover:text-white hover:bg-white/5 transition-all"
                title="Command Palette (⌘K)"
              >
                <Command className="w-5 h-5" />
              </button>

              {/* Refresh */}
              <button
                onClick={onRefresh}
                disabled={isRefreshing}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all',
                  'bg-gradient-to-r from-accent-primary to-accent-secondary text-black',
                  isRefreshing && 'opacity-70'
                )}
              >
                <RefreshCw className={cn('w-4 h-4', isRefreshing && 'animate-spin')} />
                <span className="hidden sm:inline">Refresh</span>
              </button>
            </div>
          </div>

          {/* Tabs row */}
          <div className="flex items-center gap-1 py-2 overflow-x-auto hide-scrollbar">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => onTabChange(tab.id)}
                  className={cn(
                    'flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap',
                    tab.gradient && `bg-gradient-to-r ${tab.gradient}`,
                    activeTab === tab.id
                      ? 'bg-accent-primary/20 text-accent-primary border border-accent-primary/30'
                      : 'text-white/60 hover:text-white hover:bg-white/5'
                  )}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden xl:inline">{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Command Palette Modal */}
      {showCommandPalette && (
        <div
          className="fixed inset-0 z-[9999] bg-black/70 backdrop-blur-sm flex items-start justify-center pt-[15vh]"
          onClick={() => setShowCommandPalette(false)}
        >
          <div
            className="w-[600px] max-w-[90vw] bg-brand-card border border-white/10 rounded-2xl shadow-2xl overflow-hidden animate-slide-in"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Search input */}
            <div className="flex items-center gap-3 p-4 border-b border-white/5">
              <Search className="w-5 h-5 text-white/40" />
              <input
                type="text"
                placeholder="Search commands..."
                value={commandQuery}
                onChange={(e) => setCommandQuery(e.target.value)}
                className="flex-1 bg-transparent text-white text-lg outline-none placeholder:text-white/40"
                autoFocus
              />
              <kbd className="px-2 py-1 bg-white/5 border border-white/10 rounded text-xs text-white/40 font-mono">
                ESC
              </kbd>
            </div>

            {/* Results */}
            <div className="max-h-[420px] overflow-y-auto space-y-2">
              <div>
                <div className="p-2 text-xs text-white/40 uppercase tracking-wider">Navigation</div>
                {filteredTabs.map((tab) => {
                  const Icon = tab.icon;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => {
                        onTabChange(tab.id);
                        setShowCommandPalette(false);
                        setCommandQuery('');
                      }}
                      className="w-full flex items-center gap-3 px-4 py-3 hover:bg-white/5 transition-colors"
                    >
                      <Icon className="w-5 h-5 text-white/60" />
                      <div className="flex-1 text-left">
                        <div className="text-white font-medium">{tab.label}</div>
                        <div className="text-sm text-white/40">Go to {tab.label} tab</div>
                      </div>
                      {tab.shortcut && (
                        <kbd className="px-2 py-1 bg-white/5 border border-white/10 rounded text-xs text-white/40 font-mono">
                          {tab.shortcut}
                        </kbd>
                      )}
                    </button>
                  );
                })}
              </div>

              {filteredBenchmarks.length > 0 && (
                <div>
                  <div className="p-2 text-xs text-white/40 uppercase tracking-wider">Benchmarks</div>
                  {filteredBenchmarks.map((b, idx) => (
                    <div key={idx} className="flex items-center justify-between px-4 py-3">
                      <div>
                        <div className="text-white font-medium">{b.name}</div>
                        <div className="text-xs text-white/40">{b.chapter}</div>
                      </div>
                      {b.speedup && (
                        <span className="text-accent-primary font-mono text-sm">
                          {b.speedup.toFixed(2)}x
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {actionCommands.length > 0 && (
                <div>
                  <div className="p-2 text-xs text-white/40 uppercase tracking-wider">Actions</div>
                  {actionCommands.map((item, idx) => {
                    const Icon = item.icon;
                    return (
                      <button
                        key={idx}
                        onClick={() => {
                          item.action();
                          setShowCommandPalette(false);
                          setCommandQuery('');
                        }}
                        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-white/5 transition-colors"
                      >
                        <Icon className="w-5 h-5 text-white/60" />
                        <div className="flex-1 text-left">
                          <div className="text-white font-medium">{item.label}</div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center gap-6 px-4 py-3 border-t border-white/5 text-xs text-white/40">
              <span>↑↓ Navigate</span>
              <span>↵ Select</span>
              <span>ESC Close</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
