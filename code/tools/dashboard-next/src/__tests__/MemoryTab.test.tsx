import React, { type ReactNode } from 'react';
import { render, screen } from '@testing-library/react';
import { afterEach, beforeEach, vi } from 'vitest';
import { MemoryTab } from '../components/tabs/MemoryTab';

const mockUseApiQuery = vi.fn();

vi.mock('@/lib/useApi', () => ({
  useApiQuery: (...args: unknown[]) => mockUseApiQuery(...(args as any)),
  getErrorMessage: (err: any) => String(err),
}));

vi.mock('recharts', () => {
  const Wrapper = ({ children }: { children: ReactNode }) => <div>{children}</div>;
  const Component = ({ children }: { children: ReactNode }) => <div>{children}</div>;
  return {
    ResponsiveContainer: Wrapper,
    AreaChart: Component,
    Area: Component,
    XAxis: Component,
    YAxis: Component,
    CartesianGrid: Component,
    Tooltip: Component,
    PieChart: Component,
    Pie: Component,
    Cell: Component,
  };
});

describe('MemoryTab', () => {
  beforeEach(() => {
    mockUseApiQuery.mockReturnValue({
      data: {
        timeline: [
          { timestamp: 0, allocated: 256_000_000, reserved: 512_000_000 },
          { timestamp: 10, allocated: 512_000_000, reserved: 512_000_000 },
        ],
        allocations: [
          { name: 'Parameters', size: 512_000_000, category: 'Parameters' },
          { name: 'Activations', size: 256_000_000, category: 'Activations' },
        ],
        peak_memory: 1_024_000_000,
        total_memory: 2_000_000_000,
      },
      isLoading: false,
      error: null,
      mutate: vi.fn(),
      isValidating: false,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('renders memory stats and allocations', () => {
    render(<MemoryTab />);

    expect(screen.getByText(/Peak Memory/i)).toBeInTheDocument();
    expect(screen.getByText(/1.02 GB/)).toBeInTheDocument();
    expect(screen.getByText(/Current Allocated/i)).toBeInTheDocument();
    expect(screen.getByText(/Reserved/i)).toBeInTheDocument();
    expect(screen.getByText(/Parameters/)).toBeInTheDocument();
    expect(screen.getByText(/Activations/)).toBeInTheDocument();
  });

  it('shows error state when query fails', () => {
    mockUseApiQuery.mockReturnValueOnce({
      data: null,
      isLoading: false,
      error: new Error('fail'),
      mutate: vi.fn(),
      isValidating: false,
    });

    render(<MemoryTab />);
    expect(screen.getByText(/fail/)).toBeInTheDocument();
  });
});
