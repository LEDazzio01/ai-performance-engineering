'use client';

import { List, RefreshCw } from 'lucide-react';
import { getProfiles } from '@/lib/api';
import { EmptyState, ErrorState, LoadingState } from '@/components/DataState';
import { getErrorMessage, useApiQuery } from '@/lib/useApi';

export function ProfilesListCard() {
  const profilesQuery = useApiQuery('profiles', getProfiles);
  const profiles = (profilesQuery.data as any)?.profiles || (Array.isArray(profilesQuery.data) ? (profilesQuery.data as any[]) : []);

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <List className="w-5 h-5 text-accent-secondary" />
          <h3 className="font-medium text-white">Profile Artifacts</h3>
        </div>
        <button
          onClick={() => profilesQuery.mutate()}
          className="p-2 rounded hover:bg-white/5 text-white/70"
          aria-label="Refresh profile artifacts"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>
      <div className="card-body space-y-2">
        {profilesQuery.error ? (
          <ErrorState
            message={getErrorMessage(profilesQuery.error, 'Failed to load profiles')}
            onRetry={() => profilesQuery.mutate()}
          />
        ) : profilesQuery.isLoading ? (
          <LoadingState inline message="Loading profiles..." />
        ) : profiles.length === 0 ? (
          <EmptyState title="No profile artifacts found" description="Run a new profile to see saved artifacts here." />
        ) : (
          <div className="space-y-2">
            {profiles.slice(0, 8).map((p, i) => (
              <div key={i} className="p-3 rounded-lg bg-white/5 border border-white/10">
                <div className="text-white font-semibold">{p.name || p.id || `Profile ${i + 1}`}</div>
                {p.path && <div className="text-xs text-white/50">{p.path}</div>}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
