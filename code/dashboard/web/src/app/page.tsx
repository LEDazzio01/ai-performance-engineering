'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  getBenchmarkData,
  getGpuInfo,
  compareRuns as compareRunsApi,
  generateLaunchPlan as generateLaunchPlanApi,
  getRooflineSweep,
  exportPDF,
  exportHTML,
} from '@/lib/api';
import { Navigation } from '@/components/Navigation';
import { StatsCard } from '@/components/StatsCard';
import { SpeedupChart } from '@/components/SpeedupChart';
import { StatusChart } from '@/components/StatusChart';
import { BenchmarkTable } from '@/components/BenchmarkTable';
import { GpuCard } from '@/components/GpuCard';
import { GpuStatusWidget } from '@/components/GpuStatusWidget';
import { SoftwareStackWidget } from '@/components/SoftwareStackWidget';
import { DependenciesWidget } from '@/components/DependenciesWidget';
import { ExportMenu } from '@/components/ExportMenu';
import { ReportsTab } from '@/components/tabs/ReportsTab';
import { PinnedBar } from '@/components/PinnedBar';
import {
  ShortcutsModal,
  PerformanceTargetsModal,
  type PerformanceTargets,
  RunBenchmarkModal,
  FocusOverlay,
  ComparisonMatrixModal,
  GpuControlPanel,
  CudaEnvCard,
  CodeDiffModal,
  FilterBar,
  FilterState,
  GpuThermalMonitor,
  LiveGpuMonitor,
  InsightsCard,
  RegressionAlerts,
  DiagnosticsCard,
  SystemContextCard,
  AvailableBenchmarksCard,
  PreflightChecklist,
  VarianceAnalysis,
} from '@/components';
import { useToast } from '@/components/Toast';
import {
  CompareTab,
  LLMInsightsTab,
  RooflineTab,
  ProfilerTab,
  MemoryTab,
  CompileTab,
  DeepProfileTab,
  LiveOptimizerTab,
  AnalysisTab,
  AdvancedTab,
  MultiGpuTab,
  DistributedTab,
  RLHFTab,
  InferenceTab,
  HistoryTab,
  BatchSizeOptimizationTab,
  WebhooksTab,
  ThemesTab,
  HwbenchTab,
  AIAssistantTab,
} from '@/components/tabs';
import {
  Zap,
  TrendingUp,
  CheckCircle,
  Clock,
  Loader2,
  AlertTriangle,
  RefreshCw,
  Pause,
  Target,
  Table,
  FileCode2,
  Keyboard,
  Gauge,
} from 'lucide-react';
import { Benchmark, BenchmarkData } from '@/types';

function normalizeBenchmarkData(data: BenchmarkData) {
  if (!data?.benchmarks) return data;

  const benchmarks = data.benchmarks.map((b) => {
    const rawSpeedup = b.raw_speedup ?? b.speedup ?? 0;
    return {
      ...b,
      raw_speedup: rawSpeedup,
      speedup: rawSpeedup,
      speedup_capped: false,
    };
  });

  const succeeded = benchmarks.filter((b) => b.status === 'succeeded' && typeof b.speedup === 'number');
  const failed = benchmarks.filter((b) => b.status === 'failed');
  const skipped = benchmarks.filter((b) => b.status === 'skipped');
  const total = benchmarks.length;
  const speedups = succeeded.map((b) => b.speedup || 0);
  const summary = {
    total,
    succeeded: succeeded.length,
    failed: failed.length,
    skipped: skipped.length,
    avg_speedup: speedups.length ? speedups.reduce((sum, s) => sum + s, 0) / speedups.length : 0,
    max_speedup: speedups.length ? Math.max(...speedups) : 0,
    min_speedup: speedups.length ? Math.min(...speedups) : 0,
  };

  return { ...data, benchmarks, summary };
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const [benchmarkData, setBenchmarkData] = useState<any>(null);
  const [gpuInfo, setGpuInfo] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [pinnedBenchmarks, setPinnedBenchmarks] = useState<Set<string>>(new Set());
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [filters, setFilters] = useState<FilterState>({
    search: '',
    chapter: '',
    type: '',
    status: '',
    goal: '',
    speedup: '',
  });
  const [favorites, setFavorites] = useState<Set<string>>(new Set());
  const [filterPresets, setFilterPresets] = useState<Record<string, FilterState>>({});
  const [filteredBenchmarks, setFilteredBenchmarks] = useState<Benchmark[]>([]);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [showTargets, setShowTargets] = useState(false);
  const [showRunModal, setShowRunModal] = useState(false);
  const [showMatrix, setShowMatrix] = useState(false);
  const [focusBenchmark, setFocusBenchmark] = useState<Benchmark | null>(null);
  const [showFocus, setShowFocus] = useState(false);
  const [codeDiffTarget, setCodeDiffTarget] = useState<{ chapter: string; name: string } | null>(null);
  const [compareInputs, setCompareInputs] = useState({ baseline: 'benchmark_test_results.json', candidate: 'benchmark_test_results.json', top: 5 });
  const [compareResult, setCompareResult] = useState<{ regressions: any[]; improvements: any[] } | null>(null);
  const [launchPlanInputs, setLaunchPlanInputs] = useState({
    model_params: 70,
    nodes: 1,
    gpus: 8,
    tp: 1,
    pp: 1,
    dp: 1,
    batch_size: 1,
    script: 'train.py',
    extra_args: '',
  });
  const [launchPlan, setLaunchPlan] = useState<{ command: string; plan: any } | null>(null);
  const [rooflineSweep, setRooflineSweep] = useState<{ size_mb: number; rows: { stride: number; bandwidth_gbps: number }[] }>({ size_mb: 32, rows: [] });
  const [targets, setTargets] = useState<PerformanceTargets>({
    minAvgSpeedup: 2.0,
    maxRegressions: 0,
    passRate: 95,
    maxMemoryUtil: 80,
  });
  const { showToast } = useToast();

  const applyFilters = useCallback(
    (list: Benchmark[], f: FilterState) => {
      return list.filter((b) => {
        const matchesSearch =
          !f.search ||
          b.name.toLowerCase().includes(f.search.toLowerCase()) ||
          b.chapter.toLowerCase().includes(f.search.toLowerCase());
        const matchesChapter = !f.chapter || b.chapter === f.chapter;
        const matchesType = !f.type || b.type === f.type;
        const matchesStatus = !f.status || b.status === f.status;
        const matchesGoal = !f.goal || b.optimization_goal === f.goal;
        const matchesSpeedup =
          !f.speedup ||
          (f.speedup === 'regression' && (b.speedup || 0) < 1) ||
          (Number(f.speedup) > 0 && (b.speedup || 0) >= Number(f.speedup));
        return matchesSearch && matchesChapter && matchesType && matchesStatus && matchesGoal && matchesSpeedup;
      });
    },
    []
  );

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const [benchmarks, gpu] = await Promise.all([
        getBenchmarkData(),
        getGpuInfo(),
      ]);
      const normalized = normalizeBenchmarkData(benchmarks as BenchmarkData);
      setBenchmarkData(normalized);
      setGpuInfo(gpu);
    } catch (e) {
      console.error('Failed to load data:', e);
      setError(e instanceof Error ? e.message : 'Failed to connect to backend');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleCompareRuns = useCallback(async () => {
    try {
      const res = await compareRunsApi({
        baseline: compareInputs.baseline,
        candidate: compareInputs.candidate,
        top: compareInputs.top,
      });
      setCompareResult(res as any);
      showToast('Compared runs', 'success');
    } catch (e) {
      showToast('Compare failed', 'error');
    }
  }, [compareInputs, showToast]);

  const handleLaunchPlan = useCallback(async () => {
    try {
      const res = await generateLaunchPlanApi(launchPlanInputs as any);
      setLaunchPlan(res as any);
      showToast('Launch plan generated', 'success');
    } catch (e) {
      showToast('Launch plan failed', 'error');
    }
  }, [launchPlanInputs, showToast]);

  const handleRooflineSweep = useCallback(async () => {
    try {
      const res = await getRooflineSweep(rooflineSweep?.size_mb || 32);
      setRooflineSweep(res as any);
      showToast('Roofline sweep updated', 'success');
    } catch (e) {
      showToast('Roofline sweep failed', 'error');
    }
  }, [rooflineSweep?.size_mb, showToast]);

  const quickDownload = useCallback(
    async (type: 'pdf' | 'html') => {
      try {
        const blob = type === 'pdf' ? await exportPDF() : await exportHTML();
        const filename = `report_${new Date().toISOString().split('T')[0]}.${type}`;
        const url = URL.createObjectURL(blob as Blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast(`Downloaded ${filename}`, 'success');
      } catch (e) {
        showToast('Download failed', 'error');
      }
    },
    [showToast]
  );

  // Apply filters when data or filters change
  useEffect(() => {
    if (!benchmarkData?.benchmarks) return;
    setFilteredBenchmarks(applyFilters(benchmarkData.benchmarks, filters));
  }, [benchmarkData, filters, applyFilters]);

  const filteredSummary = useMemo(() => {
    const total = filteredBenchmarks.length;
    const succeeded = filteredBenchmarks.filter((b) => b.status === 'succeeded').length;
    const failed = filteredBenchmarks.filter((b) => b.status === 'failed').length;
    const skipped = filteredBenchmarks.filter((b) => b.status === 'skipped').length;
    const speedups = filteredBenchmarks.filter((b) => b.status === 'succeeded' && typeof b.speedup === 'number');
    const avg_speedup = speedups.length > 0 ? speedups.reduce((sum, b) => sum + (b.speedup || 0), 0) / speedups.length : 0;
    const max_speedup = speedups.length > 0 ? Math.max(...speedups.map((b) => b.speedup || 0)) : 0;
    const min_speedup = speedups.length > 0 ? Math.min(...speedups.map((b) => b.speedup || 0)) : 0;
    return { total, succeeded, failed, skipped, avg_speedup, max_speedup, min_speedup };
  }, [filteredBenchmarks]);

  // Auto-refresh effect
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      loadData();
    }, 10000); // Every 10 seconds
    
    return () => clearInterval(interval);
  }, [autoRefresh, loadData]);

  // Load pinned from URL on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const pinned = params.get('pinned');
    if (pinned) {
      setPinnedBenchmarks(new Set(pinned.split(',').filter(Boolean)));
    }
    try {
      const savedFavs = localStorage.getItem('dashboard_favorites');
      if (savedFavs) {
        setFavorites(new Set(JSON.parse(savedFavs)));
      }
    } catch {
      // ignore
    }
    setFilters((prev) => ({
      ...prev,
      search: params.get('search') || '',
      chapter: params.get('chapter') || '',
      type: params.get('type') || '',
      status: params.get('status') || '',
      goal: params.get('goal') || '',
      speedup: params.get('speedup') || '',
    }));

    try {
      const storedPresets = localStorage.getItem('dashboard_filter_presets');
      if (storedPresets) {
        setFilterPresets(JSON.parse(storedPresets));
      }
    } catch {
      // ignore storage errors
    }
  }, []);

  // Load saved targets
  useEffect(() => {
    try {
      const saved = localStorage.getItem('performance_targets');
      if (saved) {
        const parsed = JSON.parse(saved);
        setTargets((prev) => ({ ...prev, ...parsed }));
      }
    } catch {
      // ignore parsing errors
    }
  }, []);

  // Sync pinned to URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (pinnedBenchmarks.size > 0) {
      params.set('pinned', Array.from(pinnedBenchmarks).join(','));
    } else {
      params.delete('pinned');
    }
    if (filters.search) params.set('search', filters.search);
    else params.delete('search');
    ['chapter', 'type', 'status', 'goal', 'speedup'].forEach((key) => {
      const val = (filters as any)[key];
      if (val) params.set(key, val);
      else params.delete(key);
    });
    const newUrl = `${window.location.pathname}${params.toString() ? '?' + params.toString() : ''}`;
    window.history.replaceState({}, '', newUrl);
  }, [pinnedBenchmarks, filters]);

  // Persist favorites
  useEffect(() => {
    try {
      localStorage.setItem('dashboard_favorites', JSON.stringify(Array.from(favorites)));
    } catch {
      // ignore
    }
  }, [favorites]);

  const handleRefresh = () => {
    setRefreshing(true);
    loadData();
    showToast('🔄 Data refreshed', 'info');
  };

  const handleTogglePin = (key: string) => {
    setPinnedBenchmarks((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
        showToast(`📌 Unpinned ${key.split(':')[1]}`, 'info');
      } else {
        next.add(key);
        showToast(`📌 Pinned ${key.split(':')[1]}`, 'success');
      }
      return next;
    });
  };

  const handleClearPinned = () => {
    setPinnedBenchmarks(new Set());
    showToast('📌 All pins cleared', 'info');
  };

  const handleToggleFavorite = (key: string) => {
    setFavorites((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
        showToast(`⭐ Removed ${key.split(':')[1]}`, 'info');
      } else {
        next.add(key);
        showToast(`⭐ Favorited ${key.split(':')[1]}`, 'success');
      }
      return next;
    });
  };

  const handleClearFavorites = () => {
    setFavorites(new Set());
    showToast('⭐ Favorites cleared', 'info');
  };

  const handleComparePinned = () => {
    setActiveTab('compare');
    showToast('📌 Switched to Compare tab with pinned benchmarks', 'info');
  };

  const saveFilterPreset = (name: string) => {
    const next = { ...filterPresets, [name]: filters };
    setFilterPresets(next);
    try {
      localStorage.setItem('dashboard_filter_presets', JSON.stringify(next));
    } catch {
      // ignore
    }
  };

  const loadFilterPreset = (name: string) => {
    const preset = filterPresets[name];
    if (preset) setFilters(preset);
  };

  const deleteFilterPreset = (name: string) => {
    const next = { ...filterPresets };
    delete next[name];
    setFilterPresets(next);
    try {
      localStorage.setItem('dashboard_filter_presets', JSON.stringify(next));
    } catch {
      // ignore
    }
  };

  const clearFilters = () => {
    setFilters({
      search: '',
      chapter: '',
      type: '',
      status: '',
      goal: '',
      speedup: '',
    });
  };

  const exportFilteredJson = () => {
    const payload = {
      exported_at: new Date().toISOString(),
      filters,
      pinned: Array.from(pinnedBenchmarks),
      benchmarks: filteredBenchmarks,
      summary: filteredSummary,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'filtered_benchmarks.json';
    a.click();
    URL.revokeObjectURL(url);
    showToast('📤 Exported filtered JSON', 'success');
  };

  const exportFilteredCsv = () => {
    const header = 'Chapter,Benchmark,Baseline (ms),Optimized (ms),Speedup,Status';
    const rows = filteredBenchmarks.map((b) =>
      // Prefer raw speedup in exports so the true value is preserved
      [
        b.chapter,
        b.name,
        b.baseline_time_ms?.toFixed?.(3) ?? '',
        b.optimized_time_ms?.toFixed?.(3) ?? '',
        (b.raw_speedup ?? b.speedup)?.toFixed?.(2) ?? '',
        b.status,
      ].join(',')
    );
    const blob = new Blob([header + '\n' + rows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'filtered_benchmarks.csv';
    a.click();
    URL.revokeObjectURL(url);
    showToast('📥 Exported filtered CSV', 'success');
  };

  const topBenchmark = useMemo(() => {
    if (!filteredBenchmarks.length) return null;
    return [...filteredBenchmarks].sort((a, b) => (b.speedup || 0) - (a.speedup || 0))[0];
  }, [filteredBenchmarks]);

  const openFocus = useCallback((benchmark?: Benchmark | null) => {
    const target = benchmark || topBenchmark;
    if (target) {
      setFocusBenchmark(target);
      setShowFocus(true);
    } else {
      showToast('No benchmark available to focus', 'error');
    }
  }, [showToast, topBenchmark]);

  const openCodeDiff = useCallback((benchmark?: Benchmark | null) => {
    const target = benchmark || topBenchmark;
    if (target) {
      setCodeDiffTarget({ chapter: target.chapter, name: target.name });
    } else {
      showToast('No benchmark selected for code diff', 'error');
    }
  }, [showToast, topBenchmark]);

  // Global keyboard shortcuts mirroring legacy index.html
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      const inInput = target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || target?.tagName === 'SELECT' || target?.isContentEditable;

      if (e.key === '/' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        document.getElementById('globalFilterSearch')?.focus();
        return;
      }
      if (inInput) return;

      const key = e.key.toLowerCase();
      if (key === 'z') {
        e.preventDefault();
        openFocus();
      } else if (key === 'b') {
        e.preventDefault();
        setShowRunModal(true);
      } else if (key === 't') {
        e.preventDefault();
        setShowTargets(true);
      } else if (key === 'a') {
        e.preventDefault();
        setAutoRefresh((prev) => !prev);
        showToast('Toggled auto-refresh', 'info');
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [openFocus, showToast]);

  const budgetSummary = useMemo(() => {
    const summary =
      filteredSummary.total > 0
        ? filteredSummary
        : benchmarkData?.summary || { total: 0, succeeded: 0, failed: 0, skipped: 0, avg_speedup: 0 };
    const passRate = summary && summary.total > 0 ? (summary.succeeded / summary.total) * 100 : undefined;
    const memoryUtil =
      gpuInfo && gpuInfo.memory_total
        ? (gpuInfo.memory_used / gpuInfo.memory_total) * 100
        : undefined;
    return {
      avgSpeedup: summary?.avg_speedup,
      regressions: summary?.failed ?? 0,
      passRate,
      memoryUtil,
    };
  }, [benchmarkData, gpuInfo, filteredSummary]);

  const budgetViolations = useMemo(() => {
    const issues: string[] = [];
    if (budgetSummary.avgSpeedup !== undefined && budgetSummary.avgSpeedup < targets.minAvgSpeedup) {
      issues.push(`Average speedup ${budgetSummary.avgSpeedup?.toFixed?.(2) || '0'}x below target ${targets.minAvgSpeedup}x`);
    }
    if (budgetSummary.regressions !== undefined && budgetSummary.regressions > targets.maxRegressions) {
      issues.push(`Regressions ${budgetSummary.regressions} exceed budget ${targets.maxRegressions}`);
    }
    if (budgetSummary.passRate !== undefined && budgetSummary.passRate < targets.passRate) {
      issues.push(`Pass rate ${budgetSummary.passRate.toFixed(0)}% below target ${targets.passRate}%`);
    }
    if (budgetSummary.memoryUtil !== undefined && budgetSummary.memoryUtil > targets.maxMemoryUtil) {
      issues.push(`Memory utilization ${budgetSummary.memoryUtil.toFixed(0)}% above ${targets.maxMemoryUtil}%`);
    }
    return issues;
  }, [budgetSummary, targets]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-accent-primary mx-auto mb-4" />
          <p className="text-white/50">Connecting to backend...</p>
          <p className="text-xs text-white/30 mt-2">Make sure the Python server is running on port 6970</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center max-w-md">
          <AlertTriangle className="w-16 h-16 text-accent-danger mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">Connection Failed</h2>
          <p className="text-white/50 mb-4">{error}</p>
          <div className="bg-white/5 rounded-lg p-4 text-left text-sm font-mono text-white/70 mb-4">
            <p className="mb-2"># Start the backend server:</p>
            <p className="text-accent-primary">python -m dashboard.api.server --port 6970</p>
          </div>
          <button
            onClick={handleRefresh}
            className="px-6 py-2 bg-accent-primary text-black rounded-lg font-medium hover:opacity-90"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <Navigation
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onRefresh={handleRefresh}
        isRefreshing={refreshing}
        onOpenShortcuts={() => setShowShortcuts(true)}
        onOpenRun={() => setShowRunModal(true)}
        onOpenTargets={() => setShowTargets(true)}
        onOpenMatrix={() => setShowMatrix(true)}
        onOpenFocus={() => openFocus()}
        onToggleAutoRefresh={() => setAutoRefresh((prev) => !prev)}
        autoRefresh={autoRefresh}
        benchmarks={benchmarkData?.benchmarks || []}
      />

      {/* Status bar */}
      <div className="fixed top-28 left-0 right-0 z-40 px-4 lg:px-6 py-2 flex flex-wrap items-center justify-center gap-4 bg-brand-bg/50 backdrop-blur-sm border-b border-white/5">
        <GpuStatusWidget />
        <SoftwareStackWidget />
        <DependenciesWidget />
        
        {/* Auto-refresh toggle */}
        <button
          onClick={() => setAutoRefresh(!autoRefresh)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm transition-all ${
            autoRefresh
              ? 'bg-accent-success/20 text-accent-success border border-accent-success/30'
              : 'bg-white/5 text-white/50 hover:text-white'
          }`}
        >
          {autoRefresh ? (
            <>
              <div className="w-2 h-2 rounded-full bg-accent-success animate-pulse" />
              Auto-refresh ON
            </>
          ) : (
            <>
              <Pause className="w-3 h-3" />
              Auto-refresh
            </>
          )}
        </button>

        <div className="flex items-center gap-2">
          <ExportMenu />
          <button
            onClick={() => quickDownload('pdf')}
            className="px-3 py-2 bg-white/10 hover:bg-white/20 border border-white/10 rounded text-xs text-white"
          >
            Download PDF
          </button>
          <button
            onClick={() => quickDownload('html')}
            className="px-3 py-2 bg-white/10 hover:bg-white/20 border border-white/10 rounded text-xs text-white"
          >
            Download HTML
          </button>
        </div>
      </div>

      <main className="pt-44 px-4 lg:px-8 pb-8">
        {benchmarkData && (
          <FilterBar
            benchmarks={benchmarkData.benchmarks || []}
            filters={filters}
            onChange={setFilters}
            presets={filterPresets}
            onSavePreset={saveFilterPreset}
            onLoadPreset={loadFilterPreset}
            onDeletePreset={deleteFilterPreset}
            onClear={clearFilters}
            onExportJson={exportFilteredJson}
            onExportCsv={exportFilteredCsv}
          />
        )}

        {budgetViolations.length > 0 && (
          <div className="mb-4 p-4 rounded-xl border border-accent-warning/30 bg-accent-warning/10 text-sm text-white space-y-1">
            <div className="font-semibold text-accent-warning">Budget warnings</div>
            {budgetViolations.map((msg, idx) => (
              <div key={idx}>• {msg}</div>
            ))}
          </div>
        )}

        {/* Regression detection banner */}
        <div className="mb-4">
          <RegressionAlerts />
        </div>

        {/* Pinned bar */}
        {pinnedBenchmarks.size > 0 && (
          <div className="mb-6">
            <PinnedBar
              pinnedBenchmarks={pinnedBenchmarks}
              onRemove={handleTogglePin}
              onClear={handleClearPinned}
              onCompare={handleComparePinned}
            />
          </div>
        )}

        {favorites.size > 0 && (
          <div className="mb-6 card">
            <div className="card-header">
              <div className="flex items-center gap-2">
                <span className="text-lg">⭐</span>
                <h3 className="text-white font-semibold">Favorites</h3>
              </div>
              <button
                onClick={handleClearFavorites}
                className="text-sm text-white/60 hover:text-white"
              >
                Clear
              </button>
            </div>
            <div className="card-body grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {Array.from(favorites).map((key) => {
                const [chapter, name] = key.split(':');
                return (
                  <div
                    key={key}
                    className="p-3 rounded-lg bg-white/5 border border-white/10 flex items-center justify-between"
                  >
                    <div>
                      <div className="text-white font-semibold">{name}</div>
                      <div className="text-xs text-white/50">{chapter}</div>
                    </div>
                    <button
                      onClick={() => handleToggleFavorite(key)}
                      className="p-2 text-accent-secondary"
                    >
                      ⭐
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Overview Tab */}
        {activeTab === 'overview' && benchmarkData && (
          <div className="space-y-6">
            {/* Quick actions */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3">
              <button
                onClick={() => openFocus(topBenchmark)}
                className="p-4 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 transition-colors text-left"
              >
                <div className="flex items-center gap-2 text-sm text-white/60 mb-1">
                  <Target className="w-4 h-4 text-accent-primary" />
                  Focus Mode
                </div>
                <div className="text-lg font-bold text-white">
                  {topBenchmark ? `${topBenchmark.speedup?.toFixed?.(2)}x` : '—'}
                </div>
                <div className="text-xs text-white/40">Highlight the top-performing benchmark</div>
              </button>

              <button
                onClick={() => setShowRunModal(true)}
                className="p-4 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 transition-colors text-left"
              >
                <div className="flex items-center gap-2 text-sm text-white/60 mb-1">
                  <Gauge className="w-4 h-4 text-accent-info" />
                  Run Benchmark
                </div>
                <div className="text-lg font-bold text-white">Quick runner</div>
                <div className="text-xs text-white/40">Launch a single target without leaving the UI</div>
              </button>

              <button
                onClick={() => setShowMatrix(true)}
                className="p-4 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 transition-colors text-left"
              >
                <div className="flex items-center gap-2 text-sm text-white/60 mb-1">
                  <Table className="w-4 h-4 text-accent-warning" />
                  Comparison Matrix
                </div>
                <div className="text-lg font-bold text-white">Side-by-side</div>
                <div className="text-xs text-white/40">Matrix view + CSV export (from the legacy UI)</div>
              </button>

              <button
                onClick={() => openCodeDiff(topBenchmark)}
                className="p-4 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 transition-colors text-left"
              >
                <div className="flex items-center gap-2 text-sm text-white/60 mb-1">
                  <FileCode2 className="w-4 h-4 text-accent-secondary" />
                  Code Diff
                </div>
                <div className="text-lg font-bold text-white">Baseline vs optimized</div>
                <div className="text-xs text-white/40">Open the same diff viewer from index.html</div>
              </button>

              <button
                onClick={() => setShowTargets(true)}
                className="p-4 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 transition-colors text-left"
              >
                <div className="flex items-center gap-2 text-sm text-white/60 mb-1">
                  <Zap className="w-4 h-4 text-accent-success" />
                  Performance Targets
                </div>
                <div className="text-lg font-bold text-white">Guardrails</div>
                <div className="text-xs text-white/40">Set budgets for regressions and memory</div>
              </button>

              <button
                onClick={() => setShowShortcuts(true)}
                className="p-4 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 transition-colors text-left"
              >
                <div className="flex items-center gap-2 text-sm text-white/60 mb-1">
                  <Keyboard className="w-4 h-4 text-accent-primary" />
                  Shortcuts
                </div>
                <div className="text-lg font-bold text-white">⌘ / Ctrl + K</div>
                <div className="text-xs text-white/40">All keyboard shortcuts from the original UI</div>
              </button>
            </div>

            {/* Budget snapshot */}
            <div className="card p-5">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Target className="w-5 h-5 text-accent-primary" />
                  <h3 className="text-white font-semibold">Performance Budget</h3>
                </div>
                <button
                  onClick={() => setShowTargets(true)}
                  className="px-3 py-1.5 bg-accent-primary/20 text-accent-primary rounded-lg text-sm"
                >
                  Edit Targets
                </button>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                {[
                  {
                    label: 'Avg Speedup',
                    target: `${targets.minAvgSpeedup}x`,
                    current: budgetSummary.avgSpeedup ? `${budgetSummary.avgSpeedup.toFixed(2)}x` : '—',
                    ok: budgetSummary.avgSpeedup !== undefined && budgetSummary.avgSpeedup >= targets.minAvgSpeedup,
                  },
                  {
                    label: 'Regressions',
                    target: `${targets.maxRegressions}`,
                    current: budgetSummary.regressions ?? '—',
                    ok: budgetSummary.regressions !== undefined && budgetSummary.regressions <= targets.maxRegressions,
                  },
                  {
                    label: 'Pass Rate',
                    target: `${targets.passRate}%`,
                    current: budgetSummary.passRate !== undefined ? `${budgetSummary.passRate.toFixed(0)}%` : '—',
                    ok: budgetSummary.passRate !== undefined && budgetSummary.passRate >= targets.passRate,
                  },
                  {
                    label: 'Memory Util',
                    target: `${targets.maxMemoryUtil}%`,
                    current: budgetSummary.memoryUtil !== undefined ? `${budgetSummary.memoryUtil.toFixed(0)}%` : '—',
                    ok: budgetSummary.memoryUtil !== undefined && budgetSummary.memoryUtil <= targets.maxMemoryUtil,
                  },
                ].map((item) => (
                  <div
                    key={item.label}
                    className={`p-4 rounded-lg border ${
                      item.ok ? 'border-accent-success/30 bg-accent-success/5' : 'border-accent-warning/30 bg-accent-warning/5'
                    }`}
                  >
                    <div className="text-sm text-white/60">{item.label}</div>
                    <div className="text-xl font-bold text-white">{item.current}</div>
                    <div className="text-xs text-white/40">Target {item.target}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <StatsCard
                title="Average Speedup"
                value={`${filteredSummary.avg_speedup?.toFixed(2) || 0}x`}
                subtitle="Across filtered benchmarks"
                icon={Zap}
                variant="success"
              />
              <StatsCard
                title="Max Speedup"
                value={`${filteredSummary.max_speedup?.toFixed(2) || 0}x`}
                subtitle="Best optimization (filtered)"
                icon={TrendingUp}
                variant="default"
              />
              <StatsCard
                title="Success Rate"
                value={`${filteredSummary.total ? ((filteredSummary.succeeded / filteredSummary.total) * 100).toFixed(0) : 0}%`}
                subtitle={`${filteredSummary.succeeded}/${filteredSummary.total || 0} passed`}
                icon={CheckCircle}
                variant={filteredSummary.succeeded === filteredSummary.total ? 'success' : 'warning'}
              />
              <StatsCard
                title="Total Benchmarks"
                value={filteredSummary.total || 0}
                subtitle={benchmarkData.timestamp ? new Date(benchmarkData.timestamp).toLocaleDateString() : 'N/A'}
                icon={Clock}
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <SpeedupChart benchmarks={filteredBenchmarks} />
              </div>
              <div>
                <StatusChart summary={filteredSummary} />
              </div>
            </div>

            {gpuInfo && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <GpuCard gpu={gpuInfo} />
                <GpuControlPanel gpuInfo={gpuInfo} />
              </div>
            )}

            {/* Performance Insights */}
            <InsightsCard />

            {/* Live GPU Monitoring */}
            <LiveGpuMonitor />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <CudaEnvCard />
              <GpuThermalMonitor />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <PreflightChecklist benchmarks={benchmarkData.benchmarks || []} gpuName={gpuInfo?.name} />
              <VarianceAnalysis benchmarks={benchmarkData.benchmarks || []} />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <SystemContextCard />
              <DiagnosticsCard />
              <AvailableBenchmarksCard />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="p-4 rounded-lg border border-white/10 bg-white/5">
                <h3 className="text-white font-semibold mb-2">Compare Runs</h3>
                <div className="space-y-2">
                  <input
                    className="w-full rounded bg-white/10 px-3 py-2 text-white text-sm"
                    value={compareInputs.baseline}
                    onChange={(e) => setCompareInputs((p) => ({ ...p, baseline: e.target.value }))}
                    placeholder="baseline benchmark_test_results.json"
                  />
                  <input
                    className="w-full rounded bg-white/10 px-3 py-2 text-white text-sm"
                    value={compareInputs.candidate}
                    onChange={(e) => setCompareInputs((p) => ({ ...p, candidate: e.target.value }))}
                    placeholder="candidate benchmark_test_results.json"
                  />
                  <div className="flex items-center gap-2">
                    <label className="text-xs text-white/60">Top</label>
                    <input
                      type="number"
                      className="w-16 rounded bg-white/10 px-2 py-1 text-white text-sm"
                      value={compareInputs.top}
                      onChange={(e) => setCompareInputs((p) => ({ ...p, top: Number(e.target.value) || 0 }))}
                    />
                    <button
                      onClick={handleCompareRuns}
                      className="ml-auto px-3 py-1.5 bg-accent-primary/20 text-accent-primary rounded text-sm"
                    >
                      Diff
                    </button>
                  </div>
                  {compareResult && (
                    <div className="text-xs text-white/80 space-y-1 max-h-32 overflow-y-auto">
                      <div className="font-semibold text-accent-warning">Regressions</div>
                      {compareResult.regressions?.map((r, idx) => (
                        <div key={`reg-${idx}`}>{r.name}: {r.baseline?.toFixed?.(2)}x → {r.candidate?.toFixed?.(2)}x</div>
                      ))}
                      <div className="font-semibold text-accent-success pt-2">Improvements</div>
                      {compareResult.improvements?.map((r, idx) => (
                        <div key={`imp-${idx}`}>{r.name}: {r.baseline?.toFixed?.(2)}x → {r.candidate?.toFixed?.(2)}x</div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div className="p-4 rounded-lg border border-white/10 bg-white/5">
                <h3 className="text-white font-semibold mb-2">Launch Plan</h3>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {(["model_params","nodes","gpus","tp","pp","dp","batch_size"] as const).map((key) => (
                    <div key={key} className="flex flex-col gap-1">
                      <label className="text-white/60">{key.replace('_',' ').toUpperCase()}</label>
                      <input
                        type="number"
                        className="rounded bg-white/10 px-2 py-1 text-white text-sm"
                        value={launchPlanInputs[key]}
                        onChange={(e) => setLaunchPlanInputs((p) => ({ ...p, [key]: Number(e.target.value) || 0 }))}
                      />
                    </div>
                  ))}
                  <div className="col-span-2 flex flex-col gap-1">
                    <label className="text-white/60">Script</label>
                    <input
                      className="rounded bg-white/10 px-2 py-1 text-white text-sm"
                      value={launchPlanInputs.script}
                      onChange={(e) => setLaunchPlanInputs((p) => ({ ...p, script: e.target.value }))}
                    />
                  </div>
                  <div className="col-span-2 flex flex-col gap-1">
                    <label className="text-white/60">Extra Args</label>
                    <input
                      className="rounded bg-white/10 px-2 py-1 text-white text-sm"
                      value={launchPlanInputs.extra_args}
                      onChange={(e) => setLaunchPlanInputs((p) => ({ ...p, extra_args: e.target.value }))}
                    />
                  </div>
                  <button
                    onClick={handleLaunchPlan}
                    className="col-span-2 px-3 py-2 bg-accent-primary/20 text-accent-primary rounded text-sm"
                  >
                    Build Command
                  </button>
                  {launchPlan && (
                    <div className="col-span-2 text-white/80 text-xs bg-black/40 rounded p-2">
                      <div className="font-semibold mb-1">Command</div>
                      <code className="whitespace-pre-wrap break-words">{launchPlan.command}</code>
                    </div>
                  )}
                </div>
              </div>

              <div className="p-4 rounded-lg border border-white/10 bg-white/5">
                <h3 className="text-white font-semibold mb-2">Memory Roofline Sweep</h3>
                <div className="flex items-center gap-2 mb-2">
                  <label className="text-xs text-white/60">Size MB</label>
                  <input
                    type="number"
                    className="w-20 rounded bg-white/10 px-2 py-1 text-white text-sm"
                    value={rooflineSweep.size_mb}
                    onChange={(e) => setRooflineSweep((p) => ({ ...p, size_mb: Number(e.target.value) || 0 }))}
                  />
                  <button
                    onClick={handleRooflineSweep}
                    className="ml-auto px-3 py-1.5 bg-accent-primary/20 text-accent-primary rounded text-sm"
                  >
                    Sweep
                  </button>
                </div>
                <div className="text-xs text-white/80 space-y-1 max-h-40 overflow-y-auto">
                  {rooflineSweep.rows.map((row, idx) => (
                    <div key={idx} className="flex items-center gap-2">
                      <span className="w-16 text-white/60">{row.stride}B</span>
                      <div className="flex-1 h-2 bg-white/5 rounded">
                        <div
                          className="h-2 bg-accent-primary rounded"
                          style={{
                            width: `${Math.min(100, (row.bandwidth_gbps || 0) * 3)}%`,
                          }}
                        />
                      </div>
                      <span className="w-16 text-right">{row.bandwidth_gbps?.toFixed(3) ?? '0.000'} GB/s</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <BenchmarkTable
              benchmarks={filteredBenchmarks}
              pinnedBenchmarks={pinnedBenchmarks}
              favorites={favorites}
              onTogglePin={handleTogglePin}
              onToggleFavorite={handleToggleFavorite}
              onFocusBenchmark={(b) => openFocus(b)}
              onShowCodeDiff={(b) => openCodeDiff(b)}
            />
          </div>
        )}

        {activeTab === 'compare' && benchmarkData && (
          <CompareTab benchmarks={filteredBenchmarks} pinnedBenchmarks={pinnedBenchmarks} />
        )}

        {activeTab === 'insights' && <LLMInsightsTab />}

        {activeTab === 'roofline' && <RooflineTab />}

        {activeTab === 'profiler' && <ProfilerTab />}

        {activeTab === 'memory' && <MemoryTab />}

        {activeTab === 'compile' && <CompileTab />}

        {activeTab === 'deepprofile' && <DeepProfileTab />}

        {activeTab === 'liveopt' && <LiveOptimizerTab />}

        {activeTab === 'analysis' && benchmarkData && (
          <AnalysisTab benchmarks={benchmarkData.benchmarks || []} />
        )}

        {activeTab === 'reports' && <ReportsTab />}

        {activeTab === 'advanced' && <AdvancedTab />}

        {activeTab === 'multigpu' && <MultiGpuTab />}

        {activeTab === 'distributed' && <DistributedTab />}

        {activeTab === 'rlhf' && <RLHFTab />}

        {activeTab === 'inference' && <InferenceTab />}

        {activeTab === 'history' && <HistoryTab />}

        {activeTab === 'batchopt' && <BatchSizeOptimizationTab />}

        {activeTab === 'webhooks' && <WebhooksTab />}

        {activeTab === 'hw' && <HwbenchTab />}

        {activeTab === 'themes' && <ThemesTab />}

        {activeTab === 'aiassistant' && <AIAssistantTab />}

      </main>

      {/* Global modals/overlays */}
      <ShortcutsModal isOpen={showShortcuts} onClose={() => setShowShortcuts(false)} />
      <PerformanceTargetsModal
        isOpen={showTargets}
        onClose={() => setShowTargets(false)}
        targets={targets}
        onSave={setTargets}
        currentSummary={budgetSummary}
      />
      <RunBenchmarkModal
        isOpen={showRunModal}
        onClose={() => setShowRunModal(false)}
        benchmarks={benchmarkData?.benchmarks || []}
        onRunComplete={loadData}
      />
      <ComparisonMatrixModal
        isOpen={showMatrix}
        onClose={() => setShowMatrix(false)}
        benchmarks={filteredBenchmarks}
      />
      <FocusOverlay
        isOpen={showFocus}
        benchmark={focusBenchmark}
        onClose={() => setShowFocus(false)}
      />
      {codeDiffTarget && (
        <CodeDiffModal
          isOpen={!!codeDiffTarget}
          onClose={() => setCodeDiffTarget(null)}
          chapter={codeDiffTarget.chapter}
          name={codeDiffTarget.name}
        />
      )}
    </div>
  );
}
