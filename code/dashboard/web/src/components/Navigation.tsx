'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import Image from 'next/image';
import { cn } from '@/lib/utils';
import { Benchmark } from '@/types';
import { getLLMStatus } from '@/lib/api';
import { useTheme } from '@/lib/ThemeContext';
import { SavingsHeaderWidget } from '@/components/SavingsHeaderWidget';
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
  Volume2,
  VolumeX,
  Timer,
  Search,
  Target,
  Table,
  Keyboard,
  FileText,
  ChevronDown,
  ChevronRight,
  Settings,
  Activity,
  Layers,
  Wrench,
  LucideIcon,
  Wifi,
  WifiOff,
  Moon,
  Sun,
} from 'lucide-react';

// Individual tab definitions
export const tabs = [
  { id: 'overview', label: 'Overview', icon: BarChart3, shortcut: '1' },
  { id: 'compare', label: 'Compare', icon: GitCompare, shortcut: '2' },
  { id: 'insights', label: 'LLM Insights', icon: Brain, shortcut: '3' },
  { id: 'roofline', label: 'Roofline', icon: Cpu, shortcut: '4' },
  { id: 'profiler', label: 'Profiler', icon: Flame, shortcut: '5' },
  { id: 'memory', label: 'Memory', icon: HardDrive, shortcut: '6' },
  { id: 'compile', label: 'Compile', icon: Zap },
  { id: 'deepprofile', label: 'Deep Profile', icon: Microscope },
  { id: 'liveopt', label: 'Live Optimizer', icon: Rocket },
  { id: 'analysis', label: 'Analysis', icon: PieChart },
  { id: 'advanced', label: 'Settings', icon: Sparkles },
  { id: 'multigpu', label: 'Multi-GPU', icon: Network },
  { id: 'distributed', label: 'Distributed', icon: Server },
  { id: 'reports', label: 'Reports', icon: FileText },
  { id: 'rlhf', label: 'RL/RLHF', icon: Gamepad2 },
  { id: 'inference', label: 'Inference', icon: Gauge },
  { id: 'history', label: 'History', icon: History },
  { id: 'batchopt', label: 'Batch Size', icon: Package },
  { id: 'webhooks', label: 'Webhooks', icon: Bell },
  { id: 'hwbench', label: 'Hardware Bench', icon: Timer },
  { id: 'themes', label: 'Themes', icon: Palette },
  { id: 'aiassistant', label: 'AI Assistant', icon: Brain },
];

// Grouped navigation structure
interface TabGroup {
  id: string;
  label: string;
  icon: LucideIcon;
  color: string;
  tabs: string[]; // tab IDs
}

const tabGroups: TabGroup[] = [
  {
    id: 'overview',
    label: 'Overview',
    icon: BarChart3,
    color: 'text-accent-primary',
    tabs: ['overview'],
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: PieChart,
    color: 'text-accent-info',
    tabs: ['compare', 'analysis', 'history', 'reports'],
  },
  {
    id: 'profiling',
    label: 'Profiling',
    icon: Flame,
    color: 'text-accent-warning',
    tabs: ['roofline', 'profiler', 'memory', 'deepprofile', 'hwbench'],
  },
  {
    id: 'optimization',
    label: 'Optimization',
    icon: Rocket,
    color: 'text-accent-success',
    tabs: ['compile', 'liveopt', 'batchopt'],
  },
  {
    id: 'ai',
    label: 'AI / LLM',
    icon: Brain,
    color: 'text-accent-secondary',
    tabs: ['insights', 'aiassistant', 'rlhf', 'inference'],
  },
  {
    id: 'infrastructure',
    label: 'Infrastructure',
    icon: Server,
    color: 'text-blue-400',
    tabs: ['multigpu', 'distributed'],
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: Settings,
    color: 'text-white/60',
    tabs: ['advanced', 'webhooks', 'themes'],
  },
];

// Get tab data by ID
function getTabById(id: string) {
  return tabs.find((t) => t.id === id);
}

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
  const [expandedGroup, setExpandedGroup] = useState<string | null>(null);
  const [dropdownPosition, setDropdownPosition] = useState<{left: number; top: number; width: number} | null>(null);
  const [llmStatus, setLlmStatus] = useState<{available: boolean; provider?: string; model?: string} | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const groupButtonRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  
  // Theme context for dark/light mode toggle
  const { availableThemes, selectedThemeId, setSelectedThemeId, setColorMode, resolvedColorMode } = useTheme();
  const isDarkMode = resolvedColorMode === 'dark';
  const findThemeForMode = useCallback(
    (mode: 'dark' | 'light') => {
      if (!availableThemes?.length) return null;
      if (mode === 'light') {
        return availableThemes.find((t) => t.id.toLowerCase().includes('light')) || null;
      }
      return availableThemes.find((t) => !t.id.toLowerCase().includes('light')) || availableThemes[0] || null;
    },
    [availableThemes]
  );
  
  const toggleDarkMode = useCallback(() => {
    const targetMode: 'dark' | 'light' = isDarkMode ? 'light' : 'dark';
    setColorMode(targetMode);

    const nextTheme = findThemeForMode(targetMode);
    if (nextTheme) {
      setSelectedThemeId(nextTheme.id);
    }
  }, [findThemeForMode, isDarkMode, setColorMode, setSelectedThemeId]);

  // Fetch LLM status on mount
  useEffect(() => {
    const fetchLLMStatus = async () => {
      try {
        const status = await getLLMStatus() as any;
        setLlmStatus({
          available: status?.available || status?.llm_available || false,
          provider: status?.provider,
          model: status?.model,
        });
      } catch {
        setLlmStatus({ available: false });
      }
    };
    fetchLLMStatus();
    // Refresh every 60 seconds
    const interval = setInterval(fetchLLMStatus, 60000);
    return () => clearInterval(interval);
  }, []);

  // Find which group the active tab belongs to
  const activeGroup = useMemo(() => {
    return tabGroups.find((g) => g.tabs.includes(activeTab))?.id || 'overview';
  }, [activeTab]);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setExpandedGroup(null);
        setDropdownPosition(null);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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
        setExpandedGroup(null);
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
  ].filter(Boolean) as { label: string; action: () => void; icon: LucideIcon }[];

  const handleGroupClick = (groupId: string) => {
    const group = tabGroups.find((g) => g.id === groupId);
    if (!group) return;
    
    // If it's a single-tab group (like Overview), just navigate
    if (group.tabs.length === 1) {
      onTabChange(group.tabs[0]);
      setExpandedGroup(null);
      setDropdownPosition(null);
    } else {
      // Toggle dropdown
      if (expandedGroup === groupId) {
        setExpandedGroup(null);
        setDropdownPosition(null);
      } else {
        setExpandedGroup(groupId);
      }
    }
  };

  const handleTabSelect = (tabId: string) => {
    onTabChange(tabId);
    setExpandedGroup(null);
    setDropdownPosition(null);
  };

  const updateDropdownPosition = useCallback((groupId: string) => {
    const btn = groupButtonRefs.current[groupId];
    if (!btn) return;
    const rect = btn.getBoundingClientRect();
    const minWidth = Math.max(rect.width, 180);
    const viewportWidth = window.innerWidth;
    const padding = 12;
    const clampedLeft = Math.min(
      Math.max(rect.left, padding),
      Math.max(viewportWidth - minWidth - padding, padding)
    );
    setDropdownPosition({
      left: clampedLeft,
      top: rect.bottom + 6,
      width: minWidth,
    });
  }, []);

  useEffect(() => {
    if (!expandedGroup) return;
    updateDropdownPosition(expandedGroup);
    const handleWindowChange = () => updateDropdownPosition(expandedGroup);
    window.addEventListener('resize', handleWindowChange);
    window.addEventListener('scroll', handleWindowChange, true);
    return () => {
      window.removeEventListener('resize', handleWindowChange);
      window.removeEventListener('scroll', handleWindowChange, true);
    };
  }, [expandedGroup, updateDropdownPosition]);

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
              {/* $ SAVINGS - FRONT AND CENTER */}
              <SavingsHeaderWidget />

              {/* Dark/Light Mode Toggle */}
              <button
                onClick={toggleDarkMode}
                className={cn(
                  'p-2 rounded-lg transition-all',
                  isDarkMode
                    ? 'text-accent-warning hover:bg-accent-warning/10'
                    : 'text-accent-info hover:bg-accent-info/10'
                )}
                title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
              >
                {isDarkMode ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
              </button>

              {/* LLM Status Indicator */}
              <div
                className={cn(
                  'flex items-center gap-2 px-3 py-1.5 rounded-full text-sm transition-all',
                  llmStatus?.available
                    ? 'bg-accent-success/20 text-accent-success border border-accent-success/30'
                    : 'bg-white/5 text-white/40 border border-white/10'
                )}
                title={llmStatus?.available 
                  ? `LLM: ${llmStatus.provider || 'Connected'} (${llmStatus.model || 'default'})` 
                  : 'LLM: Not configured - Set ANTHROPIC_API_KEY or OPENAI_API_KEY'}
              >
                {llmStatus?.available ? (
                  <Wifi className="w-4 h-4" />
                ) : (
                  <WifiOff className="w-4 h-4" />
                )}
                <span className="hidden lg:inline text-xs">
                  {llmStatus?.available ? (llmStatus.provider || 'LLM') : 'No LLM'}
                </span>
              </div>

              {/* Timer */}
              <button
                onClick={() => setTimerRunning(!timerRunning)}
                title="Session Timer - Click to start/stop"
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
                title={audioEnabled ? 'Disable audio notifications' : 'Enable audio notifications'}
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
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-white/40 hover:text-white hover:bg-white/5 transition-all border border-white/10"
                title="Command Palette (⌘K / Ctrl+K)"
              >
                <Command className="w-4 h-4" />
                <kbd className="hidden md:inline text-[10px] px-1.5 py-0.5 rounded bg-white/10 font-mono">⌘K</kbd>
              </button>

              {/* Refresh */}
              <button
                onClick={onRefresh}
                disabled={isRefreshing}
                title="Refresh data (R)"
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

          {/* Grouped tabs row */}
          <div
            className="relative flex items-center gap-1 py-2 overflow-x-auto overflow-y-visible hide-scrollbar select-none"
            ref={dropdownRef}
          >
            {tabGroups.map((group) => {
              const GroupIcon = group.icon;
              const isActive = activeGroup === group.id;
              const isExpanded = expandedGroup === group.id;
              const isSingleTab = group.tabs.length === 1;
              
              // Get active tab label if in this group
              const activeTabInGroup = group.tabs.includes(activeTab) ? getTabById(activeTab) : null;
              const displayLabel = activeTabInGroup && !isSingleTab 
                ? activeTabInGroup.label 
                : group.label;

              return (
                <div key={group.id} className="relative">
                  <button
                    ref={(el) => { groupButtonRefs.current[group.id] = el; }}
                    onClick={() => handleGroupClick(group.id)}
                    className={cn(
                      'flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap select-none',
                      isActive
                        ? 'bg-accent-primary/20 text-accent-primary border border-accent-primary/30'
                        : 'text-white/60 hover:text-white hover:bg-white/5'
                    )}
                  >
                    <GroupIcon className={cn('w-4 h-4', isActive && group.color)} />
                    <span>{displayLabel}</span>
                    {!isSingleTab && (
                      <ChevronDown 
                        className={cn(
                          'w-3 h-3 transition-transform',
                          isExpanded && 'rotate-180'
                        )} 
                      />
                    )}
                  </button>

                  {/* Dropdown menu */}
                  {isExpanded && !isSingleTab && expandedGroup === group.id && dropdownPosition && (
                    <div
                      className="fixed py-1 bg-brand-card border border-white/10 rounded-xl shadow-xl z-[90] animate-slide-in"
                      style={{
                        top: dropdownPosition.top,
                        left: dropdownPosition.left,
                        minWidth: dropdownPosition.width,
                      }}
                    >
                      {group.tabs.map((tabId) => {
                        const tab = getTabById(tabId);
                        if (!tab) return null;
                        const TabIcon = tab.icon;
                        const isTabActive = activeTab === tabId;
                        
                        return (
                          <button
                            key={tabId}
                            onClick={() => handleTabSelect(tabId)}
                            className={cn(
                              'w-full flex items-center gap-3 px-4 py-2.5 text-sm transition-colors select-none',
                              isTabActive
                                ? 'bg-accent-primary/20 text-accent-primary'
                                : 'text-white/70 hover:text-white hover:bg-white/5'
                            )}
                          >
                            <TabIcon className="w-4 h-4" />
                            <span>{tab.label}</span>
                            {tab.shortcut && (
                              <kbd className="ml-auto px-1.5 py-0.5 bg-white/5 border border-white/10 rounded text-[10px] text-white/40 font-mono">
                                {tab.shortcut}
                              </kbd>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}

            {/* Quick navigation dots for mobile */}
            <div className="flex items-center gap-1 ml-auto pl-4 border-l border-white/10 xl:hidden">
              {tabs.slice(0, 6).map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => onTabChange(tab.id)}
                    className={cn(
                      'p-2 rounded-lg transition-all',
                      activeTab === tab.id
                        ? 'bg-accent-primary/20 text-accent-primary'
                        : 'text-white/40 hover:text-white hover:bg-white/5'
                    )}
                    title={`${tab.label} (${tab.shortcut})`}
                  >
                    <Icon className="w-4 h-4" />
                  </button>
                );
              })}
            </div>
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
                placeholder="Search tabs, benchmarks, actions..."
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
            <div className="max-h-[420px] overflow-y-auto">
              {/* Grouped Navigation */}
              {tabGroups.map((group) => {
                const groupTabs = group.tabs
                  .map(getTabById)
                  .filter(Boolean)
                  .filter(
                    (tab) =>
                      !commandQuery ||
                      tab!.label.toLowerCase().includes(commandQuery.toLowerCase()) ||
                      tab!.id.toLowerCase().includes(commandQuery.toLowerCase())
                  );

                if (groupTabs.length === 0) return null;

                const GroupIcon = group.icon;

                return (
                  <div key={group.id}>
                    <div className="flex items-center gap-2 px-4 py-2 text-xs text-white/40 uppercase tracking-wider">
                      <GroupIcon className={cn('w-3 h-3', group.color)} />
                      {group.label}
                    </div>
                    {groupTabs.map((tab) => {
                      if (!tab) return null;
                      const Icon = tab.icon;
                      return (
                        <button
                          key={tab.id}
                          onClick={() => {
                            onTabChange(tab.id);
                            setShowCommandPalette(false);
                            setCommandQuery('');
                          }}
                          className={cn(
                            'w-full flex items-center gap-3 px-4 py-3 hover:bg-white/5 transition-colors',
                            activeTab === tab.id && 'bg-accent-primary/10'
                          )}
                        >
                          <Icon className="w-5 h-5 text-white/60" />
                          <div className="flex-1 text-left">
                            <div className="text-white font-medium">{tab.label}</div>
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
                );
              })}

              {filteredBenchmarks.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 px-4 py-2 text-xs text-white/40 uppercase tracking-wider">
                    <Activity className="w-3 h-3 text-accent-success" />
                    Benchmarks
                  </div>
                  {filteredBenchmarks.map((b, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between px-4 py-3 hover:bg-white/5 transition-colors cursor-pointer"
                      onClick={() => {
                        onTabChange('overview');
                        setShowCommandPalette(false);
                        setCommandQuery('');
                      }}
                    >
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

              {actionCommands.length > 0 && (!commandQuery || 'actions'.includes(commandQuery.toLowerCase())) && (
                <div>
                  <div className="flex items-center gap-2 px-4 py-2 text-xs text-white/40 uppercase tracking-wider">
                    <Wrench className="w-3 h-3 text-accent-warning" />
                    Actions
                  </div>
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
