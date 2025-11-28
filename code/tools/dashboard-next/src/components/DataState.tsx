'use client';

import { AlertTriangle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface LoadingStateProps {
  message?: string;
  className?: string;
  inline?: boolean;
}

export function LoadingState({ message = 'Loading...', className, inline }: LoadingStateProps) {
  return (
    <div
      className={cn(
        inline ? 'inline-flex items-center gap-2 text-sm text-white/70' : 'flex items-center justify-center gap-3 py-8 text-white/70',
        className
      )}
      role="status"
      aria-live="polite"
    >
      <Loader2 className={cn('animate-spin text-accent-primary', inline ? 'w-4 h-4' : 'w-6 h-6')} />
      <span>{message}</span>
    </div>
  );
}

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
  className?: string;
}

export function ErrorState({ message, onRetry, className }: ErrorStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3 rounded-lg border border-accent-warning/30 bg-accent-warning/10 px-4 py-6 text-center text-white/80',
        className
      )}
      role="alert"
    >
      <div className="flex items-center gap-2 text-accent-warning">
        <AlertTriangle className="w-5 h-5" />
        <span className="font-medium">Something went wrong</span>
      </div>
      <div className="text-sm text-white/70">{message}</div>
      {onRetry && (
        <button
          className="px-3 py-1.5 rounded-lg bg-white/10 text-sm text-white hover:bg-white/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-warning"
          onClick={onRetry}
        >
          Try again
        </button>
      )}
    </div>
  );
}

interface EmptyStateProps {
  title: string;
  description?: string;
  actionLabel?: string;
  onAction?: () => void;
  className?: string;
}

export function EmptyState({ title, description, actionLabel, onAction, className }: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/5 px-4 py-6 text-center text-white/70',
        className
      )}
      role="status"
      aria-live="polite"
    >
      <div className="text-white font-semibold">{title}</div>
      {description && <div className="text-sm text-white/60 max-w-md">{description}</div>}
      {actionLabel && onAction && (
        <button
          className="mt-2 rounded-lg bg-accent-primary/20 px-3 py-1.5 text-sm text-accent-primary hover:bg-accent-primary/30 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary"
          onClick={onAction}
        >
          {actionLabel}
        </button>
      )}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-md bg-white/5', className)} aria-hidden="true" />;
}
