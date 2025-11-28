'use client';

import { GitBranch, ServerCog, Bell, Upload } from 'lucide-react';

export function CiCdIntegrationCard() {
  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-accent-primary" />
          <h3 className="font-medium text-white">CI/CD Integration</h3>
        </div>
      </div>
      <div className="card-body space-y-3 text-sm text-white/80">
        <div className="p-3 rounded-lg bg-white/5 border border-white/10">
          <div className="flex items-center gap-2 text-white font-semibold">
            <ServerCog className="w-4 h-4 text-accent-info" /> Run benchmarks in CI
          </div>
          <pre className="mt-2 text-xs bg-black/40 p-3 rounded-lg overflow-x-auto">
            {`python -m tools.dashboard.server --port 6970 &
npm run build --prefix tools/dashboard-next`}
          </pre>
        </div>
        <div className="p-3 rounded-lg bg-white/5 border border-white/10">
          <div className="flex items-center gap-2 text-white font-semibold">
            <Upload className="w-4 h-4 text-accent-success" /> Publish artifacts
          </div>
          <div className="text-xs text-white/60 mt-1">
            - Upload `/tmp/dashboard-backend.log` for backend issues.
            <br />
            - Export filtered CSV/JSON via UI or `/api/export/*` for trend diffs.
          </div>
        </div>
        <div className="p-3 rounded-lg bg-white/5 border border-white/10">
          <div className="flex items-center gap-2 text-white font-semibold">
            <Bell className="w-4 h-4 text-accent-warning" /> Notifications
          </div>
          <div className="text-xs text-white/60 mt-1">
            Configure webhooks in the Webhooks tab to push regression alerts to Slack/Teams.
          </div>
        </div>
      </div>
    </div>
  );
}
