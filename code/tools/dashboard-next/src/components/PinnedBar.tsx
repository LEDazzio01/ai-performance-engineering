'use client';

import { Pin, X, GitCompare } from 'lucide-react';
import { cn } from '@/lib/utils';

interface PinnedBarProps {
  pinnedBenchmarks: Set<string>;
  onRemove: (key: string) => void;
  onClear: () => void;
  onCompare: () => void;
}

export function PinnedBar({ pinnedBenchmarks, onRemove, onClear, onCompare }: PinnedBarProps) {
  if (pinnedBenchmarks.size === 0) return null;

  const pinnedArray = Array.from(pinnedBenchmarks);

  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-accent-primary/5 border border-accent-primary/20 rounded-lg">
      <div className="flex items-center gap-2 text-sm text-accent-primary">
        <Pin className="w-4 h-4" />
        <span className="font-medium">{pinnedBenchmarks.size} pinned</span>
      </div>

      <div className="h-4 w-px bg-accent-primary/20" />

      <div className="flex-1 flex items-center gap-2 overflow-x-auto hide-scrollbar">
        {pinnedArray.slice(0, 5).map((key) => {
          const [chapter, name] = key.split(':');
          return (
            <div
              key={key}
              className="flex items-center gap-2 px-2 py-1 bg-accent-primary/10 border border-accent-primary/30 rounded-full text-sm"
            >
              <span className="text-accent-primary whitespace-nowrap">
                {name}
              </span>
              <button
                onClick={() => onRemove(key)}
                className="hover:bg-accent-primary/20 rounded-full p-0.5"
              >
                <X className="w-3 h-3 text-accent-primary" />
              </button>
            </div>
          );
        })}
        {pinnedArray.length > 5 && (
          <span className="text-sm text-accent-primary">+{pinnedArray.length - 5} more</span>
        )}
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onCompare}
          className="flex items-center gap-2 px-3 py-1.5 bg-accent-primary text-black rounded-lg text-sm font-medium hover:opacity-90"
        >
          <GitCompare className="w-4 h-4" />
          Compare
        </button>
        <button
          onClick={onClear}
          className="px-3 py-1.5 text-white/50 hover:text-white text-sm"
        >
          Clear all
        </button>
      </div>
    </div>
  );
}


