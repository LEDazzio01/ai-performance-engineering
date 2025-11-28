'use client';

import { useMemo, useState } from 'react';
import { Search, SlidersHorizontal, Save, Trash2, Share2, Download } from 'lucide-react';
import { Benchmark } from '@/types';
import { useToast } from './Toast';

export type FilterState = {
  search: string;
  chapter: string;
  type: string;
  status: string;
  goal: string;
  speedup: string;
};

interface FilterBarProps {
  benchmarks: Benchmark[];
  filters: FilterState;
  onChange: (next: FilterState) => void;
  presets: Record<string, FilterState>;
  onSavePreset: (name: string) => void;
  onLoadPreset: (name: string) => void;
  onDeletePreset: (name: string) => void;
  onClear: () => void;
  onExportJson: () => void;
  onExportCsv: () => void;
}

export function FilterBar({
  benchmarks,
  filters,
  onChange,
  presets,
  onSavePreset,
  onLoadPreset,
  onDeletePreset,
  onClear,
  onExportJson,
  onExportCsv,
}: FilterBarProps) {
  const [showPresets, setShowPresets] = useState(false);
  const { showToast } = useToast();

  const chapters = useMemo(() => Array.from(new Set(benchmarks.map((b) => b.chapter))).sort(), [benchmarks]);
  const types = useMemo(() => Array.from(new Set(benchmarks.map((b) => b.type).filter(Boolean))).sort(), [benchmarks]);

  const handleChange = (field: keyof FilterState, value: string) => {
    onChange({ ...filters, [field]: value });
  };

  const savePreset = () => {
    const name = prompt('Preset name?');
    if (!name) return;
    onSavePreset(name);
    showToast(`Saved preset "${name}"`, 'success');
    setShowPresets(false);
  };

  const copyShareLink = () => {
    try {
      navigator.clipboard.writeText(window.location.href);
      showToast('Shareable link copied', 'success');
    } catch {
      showToast('Failed to copy link', 'error');
    }
  };

  return (
    <div className="card mb-4">
      <div className="card-header flex-col sm:flex-row gap-3">
        <div className="flex items-center gap-2 text-white">
          <SlidersHorizontal className="w-4 h-4 text-accent-primary" />
          <span className="font-medium">Global Filters</span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={savePreset}
            className="px-3 py-1.5 text-sm bg-white/5 hover:bg-white/10 rounded-lg text-white flex items-center gap-2"
          >
            <Save className="w-4 h-4" /> Save Preset
          </button>
          <div className="relative">
            <button
              onClick={() => setShowPresets(!showPresets)}
              className="px-3 py-1.5 text-sm bg-white/5 hover:bg-white/10 rounded-lg text-white flex items-center gap-2"
            >
              Presets ({Object.keys(presets).length})
            </button>
            {showPresets && (
              <div className="absolute right-0 mt-2 w-56 bg-brand-card border border-white/10 rounded-xl shadow-xl z-50 p-2 space-y-1">
                {Object.keys(presets).length === 0 && (
                  <div className="text-sm text-white/50 px-2 py-1">No presets saved</div>
                )}
                {Object.keys(presets).map((name) => (
                  <div key={name} className="flex items-center justify-between px-2 py-1 hover:bg-white/5 rounded-lg">
                    <button
                      className="text-left text-white text-sm flex-1"
                      onClick={() => {
                        onLoadPreset(name);
                        setShowPresets(false);
                      }}
                    >
                      {name}
                    </button>
                    <button
                      className="text-white/40 hover:text-accent-danger"
                      onClick={() => onDeletePreset(name)}
                      title="Delete preset"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
          <button
            onClick={onClear}
            className="px-3 py-1.5 text-sm bg-white/5 hover:bg-white/10 rounded-lg text-white flex items-center gap-2"
          >
            Clear
          </button>
          <button
            onClick={copyShareLink}
            className="px-3 py-1.5 text-sm bg-white/5 hover:bg-white/10 rounded-lg text-white flex items-center gap-2"
          >
            <Share2 className="w-4 h-4" /> Share
          </button>
          <div className="flex items-center gap-1">
            <button
              onClick={onExportJson}
              className="px-3 py-1.5 text-sm bg-white/5 hover:bg-white/10 rounded-lg text-white flex items-center gap-2"
            >
              <Download className="w-4 h-4" /> JSON
            </button>
            <button
              onClick={onExportCsv}
              className="px-3 py-1.5 text-sm bg-white/5 hover:bg-white/10 rounded-lg text-white flex items-center gap-2"
            >
              <Download className="w-4 h-4" /> CSV
            </button>
          </div>
        </div>
      </div>
      <div className="card-body grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3">
        <div className="col-span-2">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
            <input
              id="globalFilterSearch"
              type="text"
              value={filters.search}
              onChange={(e) => handleChange('search', e.target.value)}
              placeholder="Search name or chapter..."
              className="w-full pl-9 pr-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder:text-white/40 focus:outline-none"
            />
          </div>
        </div>
        <select
          value={filters.chapter}
          onChange={(e) => handleChange('chapter', e.target.value)}
          className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
        >
          <option value="">All Chapters</option>
          {chapters.map((ch) => (
            <option key={ch} value={ch}>
              {ch}
            </option>
          ))}
        </select>
        <select
          value={filters.type}
          onChange={(e) => handleChange('type', e.target.value)}
          className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
        >
          <option value="">All Types</option>
          {types.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <select
          value={filters.status}
          onChange={(e) => handleChange('status', e.target.value)}
          className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
        >
          <option value="">All Status</option>
          <option value="succeeded">Succeeded</option>
          <option value="failed">Failed</option>
          <option value="skipped">Skipped</option>
        </select>
        <select
          value={filters.goal}
          onChange={(e) => handleChange('goal', e.target.value)}
          className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
        >
          <option value="">All Goals</option>
          <option value="speed">Speed</option>
          <option value="memory">Memory</option>
        </select>
        <select
          value={filters.speedup}
          onChange={(e) => handleChange('speedup', e.target.value)}
          className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
        >
          <option value="">Any Speedup</option>
          <option value="regression">Regression (&lt;1x)</option>
          <option value="1.5">≥1.5x</option>
          <option value="2">≥2x</option>
          <option value="5">≥5x</option>
        </select>
      </div>
    </div>
  );
}
