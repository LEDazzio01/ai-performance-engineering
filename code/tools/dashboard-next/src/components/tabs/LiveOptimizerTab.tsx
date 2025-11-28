'use client';

import { useEffect, useRef, useState } from 'react';
import { Rocket, Play, Square, Terminal, CheckCircle, XCircle, Clock, Loader2, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { getTargets, startOptimization, stopOptimization, getOptimizeJobs, subscribeToOptimization } from '@/lib/api';
import { useApiQuery, getErrorMessage } from '@/lib/useApi';
import { EmptyState, ErrorState, LoadingState } from '@/components/DataState';

interface LogEntry {
  timestamp: string;
  level: 'info' | 'success' | 'warning' | 'error';
  message: string;
}

export function LiveOptimizerTab() {
  const [running, setRunning] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [selectedTarget, setSelectedTarget] = useState('');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState('');
  const logsEndRef = useRef<HTMLDivElement>(null);
  const unsubscribeRef = useRef<(() => void) | null>(null);

  const targetsQuery = useApiQuery('optimize/targets', getTargets);
  const jobsQuery = useApiQuery('optimize/jobs', async () => {
    const jobs = await getOptimizeJobs().catch(() => ({ jobs: [] }));
    return (jobs as any)?.jobs || jobs || [];
  });

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
      }
    };
  }, []);

  const addLog = (level: LogEntry['level'], message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, { timestamp, level, message }]);
  };

  const handleStartOptimization = async () => {
    if (!selectedTarget) return;

    try {
      setRunning(true);
      setLogs([]);
      setProgress(0);
      setPhase('Starting...');
      addLog('info', `Starting optimization for ${selectedTarget}`);

      const result: any = await startOptimization(selectedTarget);
      setJobId(result.job_id);

      // Subscribe to SSE updates
      unsubscribeRef.current = subscribeToOptimization(result.job_id, (data: any) => {
        if (data.type === 'log') {
          addLog(data.level || 'info', data.message);
        } else if (data.type === 'progress') {
          setProgress(data.progress);
          setPhase(data.phase || '');
        } else if (data.type === 'complete') {
          setRunning(false);
          setPhase('Complete');
          addLog('success', `✅ Optimization complete! Speedup: ${data.speedup?.toFixed(2) || 'N/A'}x`);
          jobsQuery.mutate();
        } else if (data.type === 'error') {
          setRunning(false);
          addLog('error', data.message);
        }
      });
    } catch (e) {
      setRunning(false);
      addLog('error', e instanceof Error ? e.message : 'Failed to start optimization');
    }
  };

  const handleStopOptimization = async () => {
    if (!jobId) return;

    try {
      await stopOptimization(jobId);
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
        unsubscribeRef.current = null;
      }
      setRunning(false);
      addLog('warning', 'Optimization cancelled by user');
    } catch (e) {
      addLog('error', e instanceof Error ? e.message : 'Failed to stop optimization');
    }
  };

  const getLevelColor = (level: LogEntry['level']) => {
    switch (level) {
      case 'success':
        return 'text-accent-success';
      case 'warning':
        return 'text-accent-warning';
      case 'error':
        return 'text-accent-danger';
      default:
        return 'text-white/70';
    }
  };

  if (targetsQuery.error || jobsQuery.error) {
    return (
      <div className="card">
        <div className="card-body">
          <ErrorState
            message={getErrorMessage(targetsQuery.error || jobsQuery.error, 'Failed to load optimizer data')}
            onRetry={() => {
              targetsQuery.mutate();
              jobsQuery.mutate();
            }}
          />
        </div>
      </div>
    );
  }

  if (targetsQuery.isLoading || jobsQuery.isLoading) {
    return (
      <div className="card">
        <div className="card-body">
          <LoadingState message="Loading optimizer..." />
        </div>
      </div>
    );
  }

  const targets = targetsQuery.data || [];
  const recentJobs = jobsQuery.data || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Rocket className="w-5 h-5 text-accent-success" />
            <h2 className="text-lg font-semibold text-white">Live Optimizer</h2>
          </div>
          <div className="flex items-center gap-3">
            {running && (
              <div className="flex items-center gap-2 text-accent-primary">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-sm">{phase}</span>
              </div>
            )}
            <button
              onClick={() => {
                targetsQuery.mutate();
                jobsQuery.mutate();
              }}
              className="p-2 hover:bg-white/5 rounded-lg transition-colors text-white/70 flex items-center gap-2"
              aria-label="Refresh optimizer data"
            >
              <RefreshCw className="w-4 h-4" />
              {(targetsQuery.isValidating || jobsQuery.isValidating) && <span className="text-xs">Refreshing…</span>}
            </button>
          </div>
        </div>
        <div className="card-body">
          <div className="flex flex-col md:flex-row gap-4">
            {/* Target selector */}
            <div className="flex-1">
              <label className="block text-sm text-white/50 mb-2">Select Target</label>
              <select
                value={selectedTarget}
                onChange={(e) => setSelectedTarget(e.target.value)}
                disabled={running}
                className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-accent-primary/50 disabled:opacity-50"
              >
                <option value="">Choose a benchmark...</option>
                {targets.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>

            {/* Action buttons */}
            <div className="flex items-end gap-2">
              {running ? (
                <button
                  onClick={handleStopOptimization}
                  className="flex items-center gap-2 px-6 py-2.5 bg-accent-danger/20 text-accent-danger border border-accent-danger/30 rounded-lg font-medium hover:bg-accent-danger/30 transition-all"
                >
                  <Square className="w-4 h-4" />
                  Stop
                </button>
              ) : (
                <button
                  onClick={handleStartOptimization}
                  disabled={!selectedTarget}
                  className="flex items-center gap-2 px-6 py-2.5 bg-gradient-to-r from-accent-success to-accent-primary text-black rounded-lg font-medium hover:opacity-90 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Play className="w-4 h-4" />
                  Start Optimization
                </button>
              )}
            </div>
          </div>

          {/* Progress */}
          <div className="mt-6">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2 text-sm text-white/60">
                <Clock className="w-4 h-4" />
                <span>{phase || 'Idle'}</span>
              </div>
              <span className="text-sm text-white/60">{progress.toFixed(0)}%</span>
            </div>
            <div className="h-3 bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-accent-primary to-accent-success rounded-full transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Logs + jobs */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="card lg:col-span-2">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Terminal className="w-5 h-5 text-accent-secondary" />
              <h3 className="font-medium text-white">Optimization Logs</h3>
            </div>
          </div>
          <div className="card-body">
            <div className="h-72 overflow-y-auto space-y-2 bg-white/5 border border-white/10 rounded-lg p-3">
              {logs.length === 0 ? (
                <div className="text-sm text-white/50">No logs yet. Start an optimization to see live output.</div>
              ) : (
                logs.map((log, i) => (
                  <div key={i} className="flex items-start gap-2 text-sm">
                    <span className="text-white/50 font-mono">{log.timestamp}</span>
                    <span className={cn(getLevelColor(log.level), 'font-mono')}>[{log.level}]</span>
                    <span className="text-white/80">{log.message}</span>
                  </div>
                ))
              )}
              <div ref={logsEndRef} />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-accent-success" />
              <h3 className="font-medium text-white">Recent Jobs</h3>
            </div>
            <button
              onClick={() => jobsQuery.mutate()}
              className="p-2 rounded hover:bg-white/5 text-white/70"
              aria-label="Refresh recent jobs"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
          <div className="card-body space-y-2">
            {recentJobs.length === 0 ? (
              <EmptyState
                title="No jobs yet"
                description="Start an optimization to see recent runs."
                actionLabel="Refresh"
                onAction={() => jobsQuery.mutate()}
              />
            ) : (
              recentJobs.slice(0, 6).map((job: any, i: number) => (
                <div key={i} className="p-3 rounded-lg bg-white/5 border border-white/10">
                  <div className="flex items-center justify-between">
                    <span className="text-white font-semibold">{job.target || job.name || `Job ${i + 1}`}</span>
                    <span className="text-xs text-white/50">{job.status || 'running'}</span>
                  </div>
                  <div className="text-xs text-white/50 mt-1">
                    {job.job_id || job.id} · {job.speedup ? `${job.speedup.toFixed?.(2)}x` : '—'}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
