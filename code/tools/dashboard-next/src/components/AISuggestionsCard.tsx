'use client';

import { useEffect, useState } from 'react';
import { getAISuggestions, getAIContext } from '@/lib/api';
import { Sparkles, AlertTriangle, Loader2, RefreshCw } from 'lucide-react';

export function AISuggestionsCard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [context, setContext] = useState<any>(null);

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const [suggestData, ctx] = await Promise.all([
        getAISuggestions().catch(() => null),
        getAIContext().catch(() => null),
      ]);
      const list =
        (suggestData as any)?.suggestions ||
        (Array.isArray(suggestData) ? suggestData : []) ||
        [];
      setSuggestions(list);
      setContext(ctx);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load AI suggestions');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-accent-primary" />
          <h3 className="font-medium text-white">AI Suggestions</h3>
        </div>
        <button onClick={load} className="p-2 hover:bg-white/5 rounded-lg">
          <RefreshCw className="w-4 h-4 text-white/50" />
        </button>
      </div>
      <div className="card-body space-y-3">
        {loading ? (
          <div className="flex items-center gap-2 text-white/60">
            <Loader2 className="w-4 h-4 animate-spin" /> Syncing with AI engine...
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 text-accent-warning text-sm">
            <AlertTriangle className="w-4 h-4" /> {error}
          </div>
        ) : suggestions.length === 0 ? (
          <div className="text-sm text-white/50">No suggestions returned.</div>
        ) : (
          <div className="space-y-2">
            {suggestions.map((s, i) => (
              <div key={i} className="p-3 rounded-lg bg-white/5 border border-white/10 text-sm text-white/80">
                {s}
              </div>
            ))}
          </div>
        )}
        {context && (
          <div className="flex flex-wrap gap-2 text-xs text-white/60">
            {Object.entries(context).slice(0, 6).map(([k, v]) => (
              <span key={k} className="px-2 py-1 rounded-full bg-white/5 border border-white/10">
                {k}: {typeof v === 'object' ? JSON.stringify(v) : String(v)}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
