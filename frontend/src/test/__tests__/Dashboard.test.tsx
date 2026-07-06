import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

import { Dashboard } from '../../pages/Dashboard';
import * as dashboardApi from '../../api/dashboard';
import type { DashboardSummary } from '../../types';

vi.mock('../../api/dashboard');

const mockFetch = dashboardApi.fetchDashboard as ReturnType<typeof vi.fn>;

const mockData: DashboardSummary = {
  status: 'success',
  message: 'Dashboard data retrieved',
  data: {
    active_clients: 12,
    total_clients: 20,
    current_round: 47,
    total_rounds: 100,
    global_accuracy: 0.8734,
    global_loss: 0.2341,
    training_status: 'running',
    experiments_running: 3,
    uptime_hours: 127.5,
    last_updated: '2026-07-06T04:00:00Z',
  },
};

function renderDashboard() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, retryDelay: 0 },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('Dashboard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('shows loading skeleton while fetching', () => {
    mockFetch.mockReturnValue(new Promise(() => {}));
    renderDashboard();
    const skeletons = document.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBeGreaterThanOrEqual(4);
  });

  it('renders stat cards with real data on success', async () => {
    mockFetch.mockResolvedValue(mockData);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getAllByText('Active Clients').length).toBeGreaterThanOrEqual(1);
    });

    expect(screen.getByText('12 / 20')).toBeInTheDocument();
    expect(screen.getByText('47 / 100')).toBeInTheDocument();
    expect(screen.getByText('87.3%')).toBeInTheDocument();
    expect(screen.getByText('0.2341')).toBeInTheDocument();
  });

  it('renders system status section with backend data', async () => {
    mockFetch.mockResolvedValue(mockData);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('System Status')).toBeInTheDocument();
    });

    expect(screen.getByText('Training Status')).toBeInTheDocument();
    expect(screen.getByText('running')).toBeInTheDocument();
    expect(screen.getByText('Experiments Running')).toBeInTheDocument();
    expect(screen.getByText('127.5h')).toBeInTheDocument();
  });

  it('shows round progress section', async () => {
    mockFetch.mockResolvedValue(mockData);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Round Progress')).toBeInTheDocument();
    });

    expect(screen.getByText('47.0%')).toBeInTheDocument();
    expect(screen.getByText('12 / 20 clients active')).toBeInTheDocument();
    expect(screen.getByText('Training round 47 of 100')).toBeInTheDocument();
  });

  it('shows error state on failure', async () => {
    mockFetch.mockRejectedValue(new Error('Server error. Please try again later.'));
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Failed to load dashboard')).toBeInTheDocument();
    });

    expect(screen.getByText('Server error. Please try again later.')).toBeInTheDocument();
    expect(screen.getByText('Try Again')).toBeInTheDocument();
  });

  it('shows error state on backend unavailable', async () => {
    mockFetch.mockRejectedValue(
      new Error('Backend is unavailable. Please check your connection.'),
    );
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Failed to load dashboard')).toBeInTheDocument();
    });

    expect(
      screen.getByText('Backend is unavailable. Please check your connection.'),
    ).toBeInTheDocument();
  });

  it('shows empty state when response has no data', async () => {
    mockFetch.mockResolvedValue({
      status: 'success',
      message: 'ok',
      data: null,
    });
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('No dashboard data available')).toBeInTheDocument();
    });
    expect(screen.getByText('Retry')).toBeInTheDocument();
  });

  it('calls fetchDashboard on mount', async () => {
    mockFetch.mockResolvedValue(mockData);
    renderDashboard();

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });
  });

  it('retries on error when clicking Try Again', async () => {
    mockFetch.mockRejectedValue(new Error('Server error'));
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Failed to load dashboard')).toBeInTheDocument();
    });

    mockFetch.mockResolvedValue(mockData);

    const retryButton = screen.getByText('Try Again');
    await userEvent.click(retryButton);

    await waitFor(() => {
      expect(screen.getByText('12 / 20')).toBeInTheDocument();
    });
  });

  it('shows manual refresh button that refetches data', async () => {
    mockFetch.mockResolvedValue(mockData);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('12 / 20')).toBeInTheDocument();
    });

    const refreshButton = screen.getByText('Refresh');
    expect(refreshButton).toBeInTheDocument();

    await userEvent.click(refreshButton);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });
  });

  it('shows last updated timestamp in header', async () => {
    mockFetch.mockResolvedValue(mockData);
    renderDashboard();

    await waitFor(() => {
      const elements = screen.queryAllByText(/Last updated/);
      expect(elements.length).toBeGreaterThanOrEqual(1);
    });
  });
});
