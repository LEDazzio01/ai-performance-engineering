'use client';

import { useEffect, useState } from 'react';
import { getCodeDiff } from '@/lib/api';
import { Loader2, FileCode, AlertTriangle } from 'lucide-react';

interface CodeDiffModalProps {
  isOpen: boolean;
  onClose: () => void;
  chapter: string;
  name: string;
}

export function CodeDiffModal({ isOpen, onClose, chapter, name }: CodeDiffModalProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    if (!isOpen) return;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const res = await getCodeDiff(chapter, name);
        setData(res);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load code diff');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [isOpen, chapter, name]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[9998] bg-black/70 backdrop-blur-sm flex items-start justify-center pt-[8vh]"
      onClick={onClose}
    >
      <div
        className="w-[1100px] max-w-[96vw] bg-brand-card border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[84vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
          <div className="flex items-center gap-2">
            <FileCode className="w-5 h-5 text-accent-info" />
            <div>
              <div className="text-xs uppercase tracking-wide text-white/40">Code Comparison</div>
              <div className="text-lg font-semibold text-white">{chapter}: {name}</div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white/70"
          >
            Close
          </button>
        </div>

        <div className="flex-1 overflow-auto">
          {loading ? (
            <div className="flex items-center justify-center py-20 text-white/60">
              <Loader2 className="w-5 h-5 animate-spin mr-2" />
              Loading baseline vs optimized code...
            </div>
          ) : error ? (
            <div className="flex items-center justify-center gap-2 py-12 text-accent-warning">
              <AlertTriangle className="w-5 h-5" />
              {error}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-0">
              <div className="border-r border-white/5">
                <div className="px-4 py-3 border-b border-white/5 bg-white/[0.03] text-sm text-accent-tertiary font-semibold">
                  Baseline
                </div>
                <pre className="p-4 text-sm text-white/80 font-mono whitespace-pre overflow-auto max-h-[70vh]">
{data?.baseline || 'No baseline code found'}
                </pre>
              </div>
              <div>
                <div className="px-4 py-3 border-b border-white/5 bg-white/[0.03] text-sm text-accent-success font-semibold">
                  Optimized
                </div>
                <pre className="p-4 text-sm text-white/80 font-mono whitespace-pre overflow-auto max-h-[70vh]">
{data?.optimized || 'No optimized code found'}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
