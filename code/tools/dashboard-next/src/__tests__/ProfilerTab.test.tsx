import React from 'react';
import { render, screen } from '@testing-library/react';
import { vi } from 'vitest';
import { ProfilerTab } from '../components/tabs/ProfilerTab';

const mockUseApiQuery = vi.fn();
const mockUseApiMutation = vi.fn();

vi.mock('@/lib/useApi', () => ({
  useApiQuery: (...args: unknown[]) => mockUseApiQuery(...(args as any)),
  useApiMutation: (...args: unknown[]) => mockUseApiMutation(...(args as any)),
  getErrorMessage: (err: any) => String(err),
}));

describe('ProfilerTab', () => {
  beforeEach(() => {
    mockUseApiQuery.mockReturnValue({
      data: {
        kernels: [{ name: 'myKernel', duration_ms: 12.3, category: 'GEMM' }],
        bottlenecks: { bottlenecks: [{ name: 'L2 bound', severity: 'high' }] },
        flame: { nodes: [] },
        score: { score: 90 },
        timeline: { cpu: [{ name: 'step', duration_ms: 1.2, start_ms: 0.1 }], gpu: [] },
      },
      isLoading: false,
      error: null,
      mutate: vi.fn(),
      isValidating: false,
    });
    mockUseApiMutation.mockReturnValue({
      trigger: vi.fn(),
      isMutating: false,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('renders profiler data', () => {
    render(<ProfilerTab />);

    expect(screen.getByText(/GPU Kernel Profiler/)).toBeInTheDocument();
    expect(screen.getByText(/myKernel/)).toBeInTheDocument();
    expect(screen.getByText(/Kernel Details/)).toBeInTheDocument();
    expect(screen.getByText(/Bottleneck Detective/)).toBeInTheDocument();
  });

  it('shows error state', () => {
    mockUseApiQuery.mockReturnValueOnce({
      data: null,
      isLoading: false,
      error: new Error('profiler failed'),
      mutate: vi.fn(),
      isValidating: false,
    });

    render(<ProfilerTab />);
    expect(screen.getByText(/profiler failed/)).toBeInTheDocument();
  });
});
