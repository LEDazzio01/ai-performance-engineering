'use client';

import { useEffect } from 'react';
import { Command, RefreshCw, Search, Zap, Target, Keyboard, Gauge, PanelTopOpen } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const shortcuts = [
  { icon: Command, title: 'Command Palette', combo: '⌘ / Ctrl + K', desc: 'Open quick navigation' },
  { icon: Search, title: 'Search Benchmarks', combo: '/', desc: 'Focus search input' },
  { icon: RefreshCw, title: 'Refresh Data', combo: 'R', desc: 'Reload benchmarks & GPU info' },
  { icon: Target, title: 'Focus Mode', combo: 'Z', desc: 'Highlight top benchmark' },
  { icon: Zap, title: 'Auto-Refresh', combo: 'A', desc: 'Toggle 10s background refresh' },
  { icon: Gauge, title: 'Run Benchmark', combo: 'B', desc: 'Open quick benchmark runner' },
  { icon: PanelTopOpen, title: 'Performance Targets', combo: 'T', desc: 'Set budgets & guardrails' },
  { icon: Keyboard, title: 'Tabs 1-9', combo: '1 … 9', desc: 'Jump between tabs quickly' },
];

export function ShortcutsModal({ isOpen, onClose }: ShortcutsModalProps) {
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[9998] bg-black/70 backdrop-blur-sm flex items-start justify-center pt-[12vh]"
      onClick={onClose}
    >
      <div
        className="w-[720px] max-w-[94vw] bg-brand-card border border-white/10 rounded-2xl shadow-2xl overflow-hidden animate-slide-in"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
          <div>
            <div className="text-sm text-white/40 uppercase tracking-wide">Keyboard Shortcuts</div>
            <div className="text-lg font-semibold text-white">Stay in flow without the mouse</div>
          </div>
          <button
            onClick={onClose}
            className="px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white/70"
          >
            Esc
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 p-6">
          {shortcuts.map((item) => {
            const Icon = item.icon;
            return (
              <div
                key={item.title}
                className={cn(
                  'flex items-center gap-3 p-3 rounded-xl border border-white/5 bg-white/[0.03]',
                  'hover:border-accent-primary/40 transition-colors'
                )}
              >
                <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center">
                  <Icon className="w-5 h-5 text-accent-primary" />
                </div>
                <div className="flex-1">
                  <div className="text-white font-medium">{item.title}</div>
                  <div className="text-sm text-white/50">{item.desc}</div>
                </div>
                <div className="px-3 py-1 bg-white/5 border border-white/10 rounded-md text-xs font-mono text-white/70">
                  {item.combo}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
