'use client';

import { useState, useMemo } from 'react';
import { Search, ArrowUpDown, ArrowUp, ArrowDown, Pin, Lightbulb, X, Target, FileCode2, Star } from 'lucide-react';
import { Benchmark } from '@/types';
import { formatMs, getSpeedupColor, cn } from '@/lib/utils';
import { ExplainModal } from './ExplainModal';

interface BenchmarkTableProps {
  benchmarks: Benchmark[];
  pinnedBenchmarks?: Set<string>;
  favorites?: Set<string>;
  speedupCap?: number;
  onTogglePin?: (key: string) => void;
  onToggleFavorite?: (key: string) => void;
  onFocusBenchmark?: (benchmark: Benchmark) => void;
  onShowCodeDiff?: (benchmark: Benchmark) => void;
}

type SortField = 'name' | 'chapter' | 'speedup' | 'baseline_time_ms' | 'optimized_time_ms' | 'status';
type SortDir = 'asc' | 'desc';

export function BenchmarkTable({
  benchmarks,
  pinnedBenchmarks = new Set(),
  favorites = new Set(),
  speedupCap,
  onTogglePin,
  onToggleFavorite,
  onFocusBenchmark,
  onShowCodeDiff,
}: BenchmarkTableProps) {
  const [search, setSearch] = useState('');
  const [sortField, setSortField] = useState<SortField>('speedup');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [chapterFilter, setChapterFilter] = useState<string>('all');
  
  // Explain modal state
  const [explainModal, setExplainModal] = useState<{
    isOpen: boolean;
    technique: string;
    speedup: number;
    benchmarkName: string;
    chapter?: string;
  }>({
    isOpen: false,
    technique: '',
    speedup: 0,
    benchmarkName: '',
  });

  const chapters = useMemo(() => {
    return Array.from(new Set(benchmarks.map((b) => b.chapter))).sort();
  }, [benchmarks]);

  const filtered = useMemo(() => {
    return benchmarks
      .filter((b) => {
        const matchesSearch =
          b.name.toLowerCase().includes(search.toLowerCase()) ||
          b.chapter.toLowerCase().includes(search.toLowerCase());
        const matchesStatus = statusFilter === 'all' || b.status === statusFilter;
        const matchesChapter = chapterFilter === 'all' || b.chapter === chapterFilter;
        return matchesSearch && matchesStatus && matchesChapter;
      })
      .sort((a, b) => {
        const aKey = `${a.chapter}:${a.name}`;
        const bKey = `${b.chapter}:${b.name}`;
        const aPinned = pinnedBenchmarks.has(aKey);
        const bPinned = pinnedBenchmarks.has(bKey);
        const aFav = favorites.has(aKey);
        const bFav = favorites.has(bKey);
        // Pinned, then favorites
        if (aPinned && !bPinned) return -1;
        if (!aPinned && bPinned) return 1;
        if (aFav && !bFav) return -1;
        if (!aFav && bFav) return 1;

        let aVal: any = a[sortField];
        let bVal: any = b[sortField];

        if (sortField === 'speedup' || sortField === 'baseline_time_ms' || sortField === 'optimized_time_ms') {
          aVal = aVal || 0;
          bVal = bVal || 0;
        }

        if (aVal < bVal) return sortDir === 'asc' ? -1 : 1;
        if (aVal > bVal) return sortDir === 'asc' ? 1 : -1;
        return 0;
      });
  }, [benchmarks, search, sortField, sortDir, statusFilter, chapterFilter, pinnedBenchmarks, favorites]);

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDir('desc');
    }
  };

  const getSortIcon = (field: SortField) => {
    if (field !== sortField) return <ArrowUpDown className="w-4 h-4 opacity-30" />;
    return sortDir === 'asc' ? (
      <ArrowUp className="w-4 h-4 text-accent-primary" />
    ) : (
      <ArrowDown className="w-4 h-4 text-accent-primary" />
    );
  };

  const handleExplain = (benchmark: Benchmark) => {
    setExplainModal({
      isOpen: true,
      technique: benchmark.name,
      speedup: benchmark.speedup || 0,
      benchmarkName: benchmark.name,
      chapter: benchmark.chapter,
    });
  };

  const getBenchmarkKey = (b: Benchmark) => `${b.chapter}:${b.name}`;

  return (
    <>
      <div className="card">
        <div className="card-header flex-col sm:flex-row gap-4">
          <h2 className="text-lg font-semibold text-white">Benchmarks</h2>
          <div className="flex flex-wrap items-center gap-3">
            {/* Pinned count */}
            {pinnedBenchmarks.size > 0 && (
              <div className="flex items-center gap-2 px-3 py-1.5 bg-accent-primary/10 border border-accent-primary/30 rounded-full">
                <Pin className="w-4 h-4 text-accent-primary" />
                <span className="text-sm text-accent-primary">{pinnedBenchmarks.size} pinned</span>
              </div>
            )}

            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
              <input
                type="text"
                placeholder="Search..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder:text-white/40 focus:outline-none focus:border-accent-primary/50 w-48"
              />
            </div>

            {/* Chapter filter */}
            <select
              value={chapterFilter}
              onChange={(e) => setChapterFilter(e.target.value)}
              className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
            >
              <option value="all">All Chapters</option>
              {chapters.map((ch) => (
                <option key={ch} value={ch}>
                  {ch}
                </option>
              ))}
            </select>

            {/* Status filter */}
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
            >
              <option value="all">All Status</option>
              <option value="succeeded">Succeeded</option>
              <option value="failed">Failed</option>
              <option value="skipped">Skipped</option>
            </select>

            <span className="text-sm text-white/40">{filtered.length} results</span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/5">
                {(onTogglePin || onToggleFavorite) && (
                  <th className="px-3 py-3 w-16"></th>
                )}
                <th
                  className="px-5 py-3 text-left text-xs font-medium text-white/50 uppercase cursor-pointer hover:text-white"
                  onClick={() => handleSort('name')}
                >
                  <div className="flex items-center gap-2">
                    Name {getSortIcon('name')}
                  </div>
                </th>
                <th
                  className="px-5 py-3 text-left text-xs font-medium text-white/50 uppercase cursor-pointer hover:text-white"
                  onClick={() => handleSort('chapter')}
                >
                  <div className="flex items-center gap-2">
                    Chapter {getSortIcon('chapter')}
                  </div>
                </th>
                <th
                  className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase cursor-pointer hover:text-white"
                  onClick={() => handleSort('baseline_time_ms')}
                >
                  <div className="flex items-center justify-end gap-2">
                    Baseline {getSortIcon('baseline_time_ms')}
                  </div>
                </th>
                <th
                  className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase cursor-pointer hover:text-white"
                  onClick={() => handleSort('optimized_time_ms')}
                >
                  <div className="flex items-center justify-end gap-2">
                    Optimized {getSortIcon('optimized_time_ms')}
                  </div>
                </th>
                <th
                  className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase cursor-pointer hover:text-white"
                  onClick={() => handleSort('speedup')}
                >
                  <div className="flex items-center justify-end gap-2">
                    Speedup {getSortIcon('speedup')}
                  </div>
                </th>
                <th
                  className="px-5 py-3 text-center text-xs font-medium text-white/50 uppercase cursor-pointer hover:text-white"
                  onClick={() => handleSort('status')}
                >
                  <div className="flex items-center justify-center gap-2">
                    Status {getSortIcon('status')}
                  </div>
                </th>
                <th className="px-5 py-3 text-center text-xs font-medium text-white/50 uppercase">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {filtered.slice(0, 50).map((b, i) => {
                const key = getBenchmarkKey(b);
                const isPinned = pinnedBenchmarks.has(key);
                const displaySpeedup = b.speedup || 0;
                const rawSpeedup = b.raw_speedup ?? displaySpeedup;
                const isCapped = b.speedup_capped && rawSpeedup !== displaySpeedup;
                
                return (
                  <tr
                    key={`${b.chapter}-${b.name}-${i}`}
                    className={cn(
                      'hover:bg-white/[0.02] transition-colors',
                      isPinned && 'bg-accent-primary/5 border-l-2 border-accent-primary'
                    )}
                  >
                {(onTogglePin || onToggleFavorite) && (
                  <td className="px-3 py-4">
                    <div className="flex items-center gap-2">
                      {onTogglePin && (
                        <button
                          onClick={() => onTogglePin(key)}
                          className={cn(
                            'p-1 rounded transition-all',
                            isPinned
                              ? 'text-accent-primary'
                              : 'text-white/20 hover:text-white/60'
                          )}
                          title={isPinned ? 'Unpin' : 'Pin'}
                        >
                          <Pin className="w-4 h-4" />
                        </button>
                      )}
                      {onToggleFavorite && (
                        <button
                          onClick={() => onToggleFavorite(key)}
                          className={cn(
                            'p-1 rounded transition-all',
                            favorites.has(key)
                              ? 'text-accent-secondary'
                              : 'text-white/20 hover:text-white/60'
                          )}
                          title={favorites.has(key) ? 'Unfavorite' : 'Favorite'}
                        >
                          <Star className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </td>
                )}
                    <td className="px-5 py-4">
                      <span className="font-medium text-white">{b.name}</span>
                    </td>
                    <td className="px-5 py-4">
                      <span className="px-2 py-1 bg-accent-secondary/10 text-accent-secondary rounded text-xs">
                        {b.chapter}
                      </span>
                    </td>
                    <td className="px-5 py-4 text-right font-mono text-sm text-accent-tertiary">
                      {b.status === 'succeeded' ? formatMs(b.baseline_time_ms) : '-'}
                    </td>
                    <td className="px-5 py-4 text-right font-mono text-sm text-accent-success">
                      {b.status === 'succeeded' ? formatMs(b.optimized_time_ms) : '-'}
                    </td>
                    <td className="px-5 py-4 text-right">
                      {b.status === 'succeeded' && displaySpeedup ? (
                        <div className="flex items-center justify-end gap-2">
                          <span
                            className="font-bold text-lg"
                            style={{ color: getSpeedupColor(displaySpeedup) }}
                          >
                            {displaySpeedup.toFixed(2)}x
                          </span>
                          {isCapped && (
                            <span
                              className="text-[10px] text-white/50 bg-white/5 px-1.5 py-0.5 rounded"
                              title={`Raw ${rawSpeedup.toFixed(2)}x${speedupCap ? ` • capped at ${speedupCap.toFixed(0)}x for display` : ''}`}
                            >
                              capped
                            </span>
                          )}
                        </div>
                      ) : (
                        '-'
                      )}
                    </td>
                    <td className="px-5 py-4 text-center">
                      <span
                        className={cn(
                          'px-2 py-1 rounded-full text-xs font-medium',
                          b.status === 'succeeded' && 'bg-accent-success/20 text-accent-success',
                          b.status === 'failed' && 'bg-accent-danger/20 text-accent-danger',
                          b.status === 'skipped' && 'bg-white/10 text-white/50'
                        )}
                      >
                        {b.status}
                      </span>
                    </td>
                    <td className="px-5 py-4 text-center">
                      <div className="flex items-center justify-center gap-2">
                        {onFocusBenchmark && (
                          <button
                            onClick={() => onFocusBenchmark(b)}
                            className="p-2 bg-white/5 hover:bg-white/10 text-white rounded-lg transition-colors"
                            title="Focus mode"
                          >
                            <Target className="w-4 h-4" />
                          </button>
                        )}
                        {onShowCodeDiff && (
                          <button
                            onClick={() => onShowCodeDiff(b)}
                            className="p-2 bg-white/5 hover:bg-white/10 text-white rounded-lg transition-colors"
                            title="View code diff"
                          >
                            <FileCode2 className="w-4 h-4" />
                          </button>
                        )}
                        <button
                          onClick={() => handleExplain(b)}
                          className="p-2 bg-accent-primary/10 hover:bg-accent-primary/20 text-accent-primary rounded-lg transition-colors"
                          title="Explain this technique"
                        >
                          <Lightbulb className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {filtered.length > 50 && (
          <div className="px-5 py-3 text-sm text-white/40 border-t border-white/5">
            Showing 50 of {filtered.length} results
          </div>
        )}
      </div>

      {/* Explain Modal */}
      <ExplainModal
        isOpen={explainModal.isOpen}
        onClose={() => setExplainModal((prev) => ({ ...prev, isOpen: false }))}
        technique={explainModal.technique}
        speedup={explainModal.speedup}
        benchmarkName={explainModal.benchmarkName}
        chapter={explainModal.chapter}
      />
    </>
  );
}
