'use client';

import { useState, useEffect } from 'react';
import { CheckCircle, AlertTriangle, Loader2, RefreshCw } from 'lucide-react';
import { getDependencies, checkDependencyUpdates } from '@/lib/api';

export function DependenciesWidget() {
  const [deps, setDeps] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [updates, setUpdates] = useState<any>(null);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const data = await getDependencies();
        setDeps(data);
        setError(false);
      } catch (e) {
        setError(true);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const handleCheckUpdates = async () => {
    try {
      setChecking(true);
      const res = await checkDependencyUpdates();
      setUpdates(res);
      setError(false);
    } catch {
      setError(true);
    } finally {
      setChecking(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-full">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-sm text-white/50">Checking...</span>
      </div>
    );
  }

  if (error) {
    return null; // Hide if unavailable
  }

  const hasIssues = deps?.missing?.length > 0 || deps?.outdated?.length > 0;

  return (
    <div className={`flex items-center gap-3 px-4 py-2 rounded-full border ${
      hasIssues 
        ? 'bg-accent-warning/10 border-accent-warning/20' 
        : 'bg-accent-success/10 border-accent-success/20'
    }`}>
      {hasIssues ? (
        <>
          <AlertTriangle className="w-4 h-4 text-accent-warning" />
          <span className="text-sm text-accent-warning">
            {deps.missing?.length || 0} missing, {deps.outdated?.length || 0} outdated
          </span>
        </>
      ) : (
        <>
          <CheckCircle className="w-4 h-4 text-accent-success" />
          <span className="text-sm text-accent-success">All deps OK</span>
        </>
      )}
      {updates && (
        <span className="text-xs text-white/60">
          Updates: {updates.missing?.length || 0} missing · {updates.outdated?.length || 0} outdated
        </span>
      )}
      <button
        onClick={handleCheckUpdates}
        title="Check for dependency updates"
        className="p-1 rounded hover:bg-white/10"
      >
        {checking ? (
          <Loader2 className="w-3 h-3 animate-spin text-white/60" />
        ) : (
          <RefreshCw className="w-3 h-3 text-white/50" />
        )}
      </button>
    </div>
  );
}

