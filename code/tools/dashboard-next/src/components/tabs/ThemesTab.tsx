'use client';

import { useState, useEffect } from 'react';
import { Palette, Check, Moon, Sun, Monitor, Loader2, AlertTriangle, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { getThemes } from '@/lib/api';

interface Theme {
  id: string;
  name: string;
  description?: string;
  colors: {
    primary: string;
    secondary: string;
    bg: string;
    card: string;
  };
}

export function ThemesTab() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [themes, setThemes] = useState<Theme[]>([]);
  const [selectedTheme, setSelectedTheme] = useState('cyberpunk');
  const [colorMode, setColorMode] = useState<'dark' | 'light' | 'system'>('dark');

  async function loadData() {
    try {
      setLoading(true);
      setError(null);
      const themesData = await getThemes();
      const themeList = (themesData as any)?.themes || themesData || [];
      setThemes(themeList);
      if ((themesData as any)?.current) {
        setSelectedTheme((themesData as any).current);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load themes');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="card">
        <div className="card-body flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-accent-secondary" />
          <span className="ml-3 text-white/50">Loading themes...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-body text-center py-16">
          <AlertTriangle className="w-12 h-12 text-accent-danger mx-auto mb-4" />
          <p className="text-white/70 mb-4">{error}</p>
          <button
            onClick={loadData}
            className="flex items-center gap-2 px-4 py-2 bg-accent-primary/20 text-accent-primary rounded-lg hover:bg-accent-primary/30 mx-auto"
          >
            <RefreshCw className="w-4 h-4" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  const currentTheme = themes.find((t) => t.id === selectedTheme);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Palette className="w-5 h-5 text-accent-secondary" />
            <h2 className="text-lg font-semibold text-white">Theme Settings</h2>
          </div>
          <button onClick={loadData} className="p-2 hover:bg-white/5 rounded-lg">
            <RefreshCw className="w-4 h-4 text-white/50" />
          </button>
        </div>
        <div className="card-body">
          {/* Color mode toggle */}
          <div className="mb-6">
            <h3 className="text-sm text-white/50 mb-3">Color Mode</h3>
            <div className="flex gap-2">
              {[
                { id: 'dark', label: 'Dark', icon: Moon },
                { id: 'light', label: 'Light', icon: Sun },
                { id: 'system', label: 'System', icon: Monitor },
              ].map((mode) => {
                const Icon = mode.icon;
                return (
                  <button
                    key={mode.id}
                    onClick={() => setColorMode(mode.id as typeof colorMode)}
                    className={cn(
                      'flex items-center gap-2 px-4 py-2 rounded-lg transition-all',
                      colorMode === mode.id
                        ? 'bg-accent-primary/20 text-accent-primary border border-accent-primary/30'
                        : 'bg-white/5 text-white/60 hover:text-white hover:bg-white/10'
                    )}
                  >
                    <Icon className="w-4 h-4" />
                    {mode.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Theme grid */}
      {themes.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Color Themes</h3>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {themes.map((theme) => (
                <button
                  key={theme.id}
                  onClick={() => setSelectedTheme(theme.id)}
                  className={cn(
                    'relative p-4 rounded-xl border transition-all text-left',
                    selectedTheme === theme.id
                      ? 'border-accent-primary bg-accent-primary/10'
                      : 'border-white/10 bg-white/5 hover:bg-white/10'
                  )}
                >
                  {selectedTheme === theme.id && (
                    <div className="absolute top-3 right-3 w-6 h-6 bg-accent-primary rounded-full flex items-center justify-center">
                      <Check className="w-4 h-4 text-black" />
                    </div>
                  )}

                  {/* Color preview */}
                  <div className="flex gap-2 mb-3">
                    <div
                      className="w-8 h-8 rounded-lg"
                      style={{ backgroundColor: theme.colors?.primary || '#00f5d4' }}
                    />
                    <div
                      className="w-8 h-8 rounded-lg"
                      style={{ backgroundColor: theme.colors?.secondary || '#9d4edd' }}
                    />
                    <div
                      className="w-8 h-8 rounded-lg border border-white/20"
                      style={{ backgroundColor: theme.colors?.bg || '#06060a' }}
                    />
                    <div
                      className="w-8 h-8 rounded-lg border border-white/20"
                      style={{ backgroundColor: theme.colors?.card || '#10101a' }}
                    />
                  </div>

                  <h4 className="font-medium text-white mb-1">{theme.name}</h4>
                  {theme.description && (
                    <p className="text-sm text-white/50">{theme.description}</p>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Preview */}
      {currentTheme && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Preview</h3>
          </div>
          <div
            className="p-6 rounded-b-xl"
            style={{ backgroundColor: currentTheme.colors?.bg }}
          >
            <div
              className="p-6 rounded-xl border border-white/10"
              style={{ backgroundColor: currentTheme.colors?.card }}
            >
              <div className="flex items-center gap-4 mb-4">
                <div
                  className="w-12 h-12 rounded-lg"
                  style={{
                    background: `linear-gradient(135deg, ${currentTheme.colors?.primary}, ${currentTheme.colors?.secondary})`,
                  }}
                />
                <div>
                  <h4
                    className="text-lg font-bold"
                    style={{ color: currentTheme.colors?.primary }}
                  >
                    Sample Card Title
                  </h4>
                  <p className="text-sm" style={{ color: 'rgba(255,255,255,0.5)' }}>
                    This is how cards will look
                  </p>
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  className="px-4 py-2 rounded-lg font-medium"
                  style={{
                    backgroundColor: currentTheme.colors?.primary,
                    color: '#000',
                  }}
                >
                  Primary Button
                </button>
                <button
                  className="px-4 py-2 rounded-lg font-medium"
                  style={{
                    backgroundColor: `${currentTheme.colors?.secondary}30`,
                    color: currentTheme.colors?.secondary,
                  }}
                >
                  Secondary Button
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
