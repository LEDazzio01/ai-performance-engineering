import React from 'react';
import { render, screen } from '@testing-library/react';
import { vi } from 'vitest';
import { DeepProfileTab } from '../components/tabs/DeepProfileTab';

const mockUseApiQuery = vi.fn();

vi.mock('@/lib/useApi', () => ({
  useApiQuery: (...args: unknown[]) => mockUseApiQuery(...(args as any)),
  getErrorMessage: (err: any) => String(err),
}));

describe('DeepProfileTab', () => {
  beforeEach(() => {
    mockUseApiQuery.mockImplementation((key: any) => {
      if (key === 'deep-profile/base') {
        return {
          data: {
            profiles: [{ chapter: 'ch1', name: 'baseline.nsys' }],
            recommendations: { recommendations: [{ title: 'Tune kernels', description: 'Use occupancy' }] },
            ncuData: { metrics: { sm_efficiency: 0.8 } },
          },
          isLoading: false,
          error: null,
          mutate: vi.fn(),
          isValidating: false,
        };
      }
      if (Array.isArray(key) && key[0] === 'deep-profile/compare') {
        return {
          data: {
            ncu_comparison: {
              baseline_sources: [{ kernel: 'k1', file: 'file.cu', line: 10 }],
            },
          },
          isLoading: false,
          error: null,
          mutate: vi.fn(),
          isValidating: false,
        };
      }
      return { data: null, isLoading: false, error: null, mutate: vi.fn(), isValidating: false };
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('renders deep profile content', () => {
    render(<DeepProfileTab />);

    expect(screen.getByText(/Deep Profile Comparison/)).toBeInTheDocument();
    expect(screen.getByText(/Available Profiles/)).toBeInTheDocument();
    expect(screen.getByText(/ch1/)).toBeInTheDocument();
    expect(screen.getByText(/NCU Deep Dive/)).toBeInTheDocument();
  });

  it('shows error state on base failure', () => {
    mockUseApiQuery.mockReturnValueOnce({
      data: null,
      isLoading: false,
      error: new Error('deep profile failed'),
      mutate: vi.fn(),
      isValidating: false,
    });

    render(<DeepProfileTab />);
    expect(screen.getByText(/deep profile failed/)).toBeInTheDocument();
  });
});
