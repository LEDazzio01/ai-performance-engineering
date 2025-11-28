'use client';

import { useState } from 'react';
import { Bell, Plus, Trash2, TestTube, CheckCircle, XCircle, Loader2, Send } from 'lucide-react';
import { testWebhook, sendWebhookNotification } from '@/lib/api';

interface Webhook {
  id: string;
  name: string;
  url: string;
  events: string[];
  enabled: boolean;
  platform?: string;
}

export function WebhooksTab() {
  const [webhooks, setWebhooks] = useState<Webhook[]>([
    {
      id: '1',
      name: 'Slack Notifications',
      url: 'https://hooks.slack.com/services/...',
      events: ['optimization_complete', 'regression_detected'],
      enabled: true,
      platform: 'slack',
    },
  ]);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newName, setNewName] = useState('');
  const [newUrl, setNewUrl] = useState('');
  const [newPlatform, setNewPlatform] = useState('slack');
  const [newEvents, setNewEvents] = useState<string[]>(['optimization_complete']);
  const [testing, setTesting] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<{ id: string; success: boolean } | null>(null);

  const eventTypes = [
    { id: 'optimization_complete', label: 'Optimization Complete' },
    { id: 'regression_detected', label: 'Regression Detected' },
    { id: 'benchmark_failed', label: 'Benchmark Failed' },
    { id: 'daily_summary', label: 'Daily Summary' },
  ];

  function addWebhook() {
    if (!newName || !newUrl) return;

    const webhook: Webhook = {
      id: Date.now().toString(),
      name: newName,
      url: newUrl,
      events: newEvents.length ? newEvents : ['optimization_complete'],
      enabled: true,
      platform: newPlatform as any,
    };

    setWebhooks([...webhooks, webhook]);
    setNewName('');
    setNewUrl('');
    setNewEvents(['optimization_complete']);
    setNewPlatform('slack');
    setShowAddForm(false);
  }

  function deleteWebhook(id: string) {
    setWebhooks(webhooks.filter((w) => w.id !== id));
  }

  function toggleWebhook(id: string) {
    setWebhooks(
      webhooks.map((w) => (w.id === id ? { ...w, enabled: !w.enabled } : w))
    );
  }

  async function testWebhook(id: string) {
    setTesting(id);
    setTestResult(null);

    const webhook = webhooks.find((w) => w.id === id);
    try {
      if (!webhook) throw new Error('Webhook not found');
      const res = await testWebhook({
        name: webhook.name,
        url: webhook.url,
        events: webhook.events,
        platform: webhook.platform,
      } as any);
      setTestResult({ id, success: (res as any).success !== false });
    } catch {
      setTestResult({ id, success: false });
    } finally {
      setTesting(null);
    }
  }

  async function sendReport(id: string) {
    setTesting(id);
    setTestResult(null);
    const webhook = webhooks.find((w) => w.id === id);
    try {
      if (!webhook) throw new Error('Webhook not found');
      const res = await sendWebhookNotification({
        url: webhook.url,
        type: webhook.platform || 'slack',
        message_type: 'summary',
      });
      setTestResult({ id, success: (res as any).success !== false });
    } catch {
      setTestResult({ id, success: false });
    } finally {
      setTesting(null);
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Bell className="w-5 h-5 text-accent-info" />
            <h2 className="text-lg font-semibold text-white">Webhooks & Notifications</h2>
          </div>
          <button
            onClick={() => setShowAddForm(true)}
            className="flex items-center gap-2 px-4 py-2 bg-accent-primary/20 text-accent-primary rounded-lg hover:bg-accent-primary/30"
          >
            <Plus className="w-4 h-4" />
            Add Webhook
          </button>
        </div>
      </div>

      {/* Add form */}
      {showAddForm && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Add New Webhook</h3>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm text-white/50 mb-2">Name</label>
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="e.g., Slack Notifications"
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-white/30"
                />
              </div>
              <div>
                <label className="block text-sm text-white/50 mb-2">Webhook URL</label>
                <input
                  type="url"
                  value={newUrl}
                  onChange={(e) => setNewUrl(e.target.value)}
                  placeholder="https://..."
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-white/30"
                />
              </div>
              <div>
                <label className="block text-sm text-white/50 mb-2">Platform</label>
                <select
                  value={newPlatform}
                  onChange={(e) => setNewPlatform(e.target.value)}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white"
                >
                  <option value="slack">Slack</option>
                  <option value="teams">Microsoft Teams</option>
                  <option value="discord">Discord</option>
                </select>
              </div>
            </div>
            <div className="mb-4">
              <div className="block text-sm text-white/50 mb-2">Events</div>
              <div className="flex flex-wrap gap-2">
                {eventTypes.map((event) => {
                  const checked = newEvents.includes(event.id);
                  return (
                    <label
                      key={event.id}
                      className="flex items-center gap-2 px-3 py-2 bg-white/5 rounded-lg text-white/80 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => {
                          if (checked) {
                            setNewEvents(newEvents.filter((e) => e !== event.id));
                          } else {
                            setNewEvents([...newEvents, event.id]);
                          }
                        }}
                        className="w-4 h-4 accent-accent-primary"
                      />
                      <span>{event.label}</span>
                    </label>
                  );
                })}
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={addWebhook}
                disabled={!newName || !newUrl}
                className="px-4 py-2 bg-accent-primary text-black rounded-lg font-medium disabled:opacity-50"
              >
                Add Webhook
              </button>
              <button
                onClick={() => setShowAddForm(false)}
                className="px-4 py-2 bg-white/5 text-white rounded-lg"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Webhooks list */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Configured Webhooks</h3>
        </div>
        <div className="card-body">
          {webhooks.length === 0 ? (
            <div className="text-center py-8 text-white/50">
              No webhooks configured. Add one to get started.
            </div>
          ) : (
            <div className="space-y-4">
              {webhooks.map((webhook) => (
                <div
                  key={webhook.id}
                  className={`p-4 rounded-lg border ${
                    webhook.enabled
                      ? 'bg-white/5 border-white/10'
                      : 'bg-white/[0.02] border-white/5 opacity-60'
                  }`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h4 className="font-medium text-white">{webhook.name}</h4>
                      <p className="text-sm text-white/50 font-mono truncate max-w-md">
                        {webhook.url}
                      </p>
                      {webhook.platform && (
                        <p className="text-xs text-white/40 mt-1">Platform: {webhook.platform}</p>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => testWebhook(webhook.id)}
                        disabled={testing === webhook.id}
                        className="p-2 hover:bg-white/5 rounded-lg"
                        title="Test webhook"
                      >
                        {testing === webhook.id ? (
                          <Loader2 className="w-4 h-4 animate-spin text-accent-info" />
                        ) : testResult?.id === webhook.id ? (
                          testResult.success ? (
                            <CheckCircle className="w-4 h-4 text-accent-success" />
                          ) : (
                            <XCircle className="w-4 h-4 text-accent-danger" />
                          )
                        ) : (
                          <TestTube className="w-4 h-4 text-white/50" />
                        )}
                      </button>
                      <button
                        onClick={() => sendReport(webhook.id)}
                        disabled={testing === webhook.id}
                        className="p-2 hover:bg-white/5 rounded-lg"
                        title="Send performance summary"
                      >
                        {testing === webhook.id ? (
                          <Loader2 className="w-4 h-4 animate-spin text-accent-primary" />
                        ) : (
                          <Send className="w-4 h-4 text-accent-primary" />
                        )}
                      </button>
                      <button
                        onClick={() => deleteWebhook(webhook.id)}
                        className="p-2 hover:bg-accent-danger/10 rounded-lg text-accent-danger"
                        title="Delete webhook"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => toggleWebhook(webhook.id)}
                        className={`px-3 py-1 rounded-full text-xs font-medium ${
                          webhook.enabled
                            ? 'bg-accent-success/20 text-accent-success'
                            : 'bg-white/10 text-white/50'
                        }`}
                      >
                        {webhook.enabled ? 'Enabled' : 'Disabled'}
                      </button>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {webhook.events.map((event) => (
                      <span
                        key={event}
                        className="px-2 py-1 bg-accent-info/20 text-accent-info text-xs rounded"
                      >
                        {eventTypes.find((e) => e.id === event)?.label || event}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Event types */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Available Event Types</h3>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {eventTypes.map((event) => (
              <div key={event.id} className="p-3 bg-white/5 rounded-lg">
                <span className="font-medium text-white">{event.label}</span>
                <p className="text-sm text-white/40">{event.id}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
