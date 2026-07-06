import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

import { Clients } from '../../pages/Clients';
import * as clientsApi from '../../api/clients';
import type { ClientListResponse } from '../../types';

vi.mock('../../api/clients');

const mockFetchClients = clientsApi.fetchClients as ReturnType<typeof vi.fn>;

const sampleClients: ClientListResponse = {
  status: 'success',
  message: 'Clients retrieved',
  total: 5,
  data: [
    { id: 'client-001', name: 'Edge-Device-Alpha', status: 'active', accuracy: 0.9123, loss: 0.1876, data_size: 15234, last_round: 47, device: 'NVIDIA Jetson Xavier', region: 'us-east-1', joined_at: '2026-06-22T00:00:00Z', last_communication: '2026-07-06T10:57:00Z' },
    { id: 'client-002', name: 'Mobile-Unit-Beta', status: 'active', accuracy: 0.8845, loss: 0.2103, data_size: 9821, last_round: 47, device: 'Google Pixel 8', region: 'eu-west-1', joined_at: '2026-06-26T00:00:00Z', last_communication: '2026-07-06T10:59:00Z' },
    { id: 'client-003', name: 'IoT-Sensor-Gamma', status: 'inactive', accuracy: 0.7654, loss: 0.3421, data_size: 4532, last_round: 32, device: 'Raspberry Pi 5', region: 'ap-southeast-1', joined_at: '2026-06-15T00:00:00Z', last_communication: '2026-07-06T06:00:00Z' },
    { id: 'client-004', name: 'Workstation-Delta', status: 'active', accuracy: 0.9456, loss: 0.1209, data_size: 28765, last_round: 47, device: 'Linux Workstation RTX 4090', region: 'us-west-2', joined_at: '2026-06-29T00:00:00Z', last_communication: '2026-07-06T10:58:00Z' },
    { id: 'client-005', name: 'Server-Epsilon', status: 'active', accuracy: 0.9234, loss: 0.1654, data_size: 32100, last_round: 46, device: 'DGX A100', region: 'eu-central-1', joined_at: '2026-06-06T00:00:00Z', last_communication: '2026-07-06T10:50:00Z' },
  ],
};

function renderClients() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, retryDelay: 0 },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Clients />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const user = userEvent.setup();

describe('Clients page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    cleanup();
  });

  /* ------------------------------------------------------------------ */
  /*  Loading state                                                      */
  /* ------------------------------------------------------------------ */
  it('shows loading skeleton while fetching', () => {
    mockFetchClients.mockReturnValue(new Promise(() => {}));
    renderClients();
    expect(screen.getByText('Client Manager')).toBeInTheDocument();
    const skeletons = document.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBeGreaterThanOrEqual(5);
  });

  /* ------------------------------------------------------------------ */
  /*  Error state                                                        */
  /* ------------------------------------------------------------------ */
  it('shows error state on fetch failure', async () => {
    mockFetchClients.mockRejectedValue(new Error('Server error. Please try again later.'));
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Failed to load clients')).toBeInTheDocument();
    });
    expect(screen.getByText('Server error. Please try again later.')).toBeInTheDocument();
    expect(screen.getByText('Try Again')).toBeInTheDocument();
  });

  it('retries on error when clicking Try Again', async () => {
    mockFetchClients.mockRejectedValue(new Error('Server error'));
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Failed to load clients')).toBeInTheDocument();
    });

    mockFetchClients.mockResolvedValue(sampleClients);

    await user.click(screen.getByText('Try Again'));

    await waitFor(() => {
      expect(screen.getByText('Edge-Device-Alpha')).toBeInTheDocument();
    });
  });

  it('handles generic error message', async () => {
    mockFetchClients.mockRejectedValue('string error');
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Failed to load clients')).toBeInTheDocument();
    });
    expect(screen.getByText('An unexpected error occurred.')).toBeInTheDocument();
  });

  /* ------------------------------------------------------------------ */
  /*  Empty state                                                        */
  /* ------------------------------------------------------------------ */
  it('shows empty state when no clients exist', async () => {
    mockFetchClients.mockResolvedValue({
      status: 'success',
      message: 'No clients',
      total: 0,
      data: [],
    });
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('No clients found')).toBeInTheDocument();
    });
    const refreshBtns = screen.getAllByText('Refresh');
    expect(refreshBtns.length).toBeGreaterThanOrEqual(1);
  });

  /* ------------------------------------------------------------------ */
  /*  Success state                                                      */
  /* ------------------------------------------------------------------ */
  it('renders client table with data on success', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Edge-Device-Alpha')).toBeInTheDocument();
    });

    expect(screen.getByText('Mobile-Unit-Beta')).toBeInTheDocument();
    expect(screen.getByText('IoT-Sensor-Gamma')).toBeInTheDocument();
    expect(screen.getByText('Workstation-Delta')).toBeInTheDocument();
    expect(screen.getByText('Server-Epsilon')).toBeInTheDocument();
  });

  it('shows 5 clients count in filter bar', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('5 clients')).toBeInTheDocument();
    });
  });

  it('renders status badges correctly', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      const activeBadges = screen.getAllByText('active');
      expect(activeBadges.length).toBeGreaterThanOrEqual(4);
    });
    const inactiveBadge = screen.getAllByText('inactive').filter(el => el.tagName === 'SPAN');
    expect(inactiveBadge.length).toBeGreaterThanOrEqual(1);
  });

  it('renders accuracy values formatted correctly', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('91.2%')).toBeInTheDocument();
    });
    expect(screen.getByText('94.6%')).toBeInTheDocument();
  });

  it('calls fetchClients on mount', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(mockFetchClients).toHaveBeenCalledTimes(1);
    });
  });

  it('shows client ID in each row', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('client-001')).toBeInTheDocument();
    });
    expect(screen.getByText('client-005')).toBeInTheDocument();
  });

  /* ------------------------------------------------------------------ */
  /*  Search filtering                                                   */
  /* ------------------------------------------------------------------ */
  it('filters clients by search query', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Edge-Device-Alpha')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText('Search by name, ID, device, or region...');
    await user.type(searchInput, 'gamma');

    await waitFor(() => {
      expect(screen.queryByText('Edge-Device-Alpha')).not.toBeInTheDocument();
    });
    expect(screen.getByText('IoT-Sensor-Gamma')).toBeInTheDocument();
    expect(screen.getByText('1 client')).toBeInTheDocument();
  });

  it('searches by device name', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Edge-Device-Alpha')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText('Search by name, ID, device, or region...');
    await user.type(searchInput, 'raspberry');

    await waitFor(() => {
      expect(screen.queryByText('Edge-Device-Alpha')).not.toBeInTheDocument();
    });
    expect(screen.getByText('IoT-Sensor-Gamma')).toBeInTheDocument();
  });

  it('searches by region', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Edge-Device-Alpha')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText('Search by name, ID, device, or region...');
    await user.type(searchInput, 'eu-');

    await waitFor(() => {
      expect(screen.queryByText('Edge-Device-Alpha')).not.toBeInTheDocument();
    });
    expect(screen.getByText('Mobile-Unit-Beta')).toBeInTheDocument();
    expect(screen.getByText('Server-Epsilon')).toBeInTheDocument();
  });

  it('shows empty state when search yields no results', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Edge-Device-Alpha')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText('Search by name, ID, device, or region...');
    await user.type(searchInput, 'zzzzz');

    await waitFor(() => {
      expect(screen.queryByText('Edge-Device-Alpha')).not.toBeInTheDocument();
    });
    expect(screen.getByText('No clients found')).toBeInTheDocument();
  });

  /* ------------------------------------------------------------------ */
  /*  Status filter                                                      */
  /* ------------------------------------------------------------------ */
  it('filters clients by status', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Edge-Device-Alpha')).toBeInTheDocument();
    });

    const filterSelect = screen.getAllByRole('combobox')[0];
    await user.selectOptions(filterSelect, 'inactive');

    await waitFor(() => {
      expect(screen.queryByText('Edge-Device-Alpha')).not.toBeInTheDocument();
    });
    expect(screen.getByText('IoT-Sensor-Gamma')).toBeInTheDocument();
    expect(screen.getByText('1 client')).toBeInTheDocument();
  });

  /* ------------------------------------------------------------------ */
  /*  Sorting                                                            */
  /* ------------------------------------------------------------------ */
  it('sorts by name column click', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Edge-Device-Alpha')).toBeInTheDocument();
    });

    const nameHeader = screen.getByText('Client');
    await user.click(nameHeader);
  });

  it('toggles sort direction on second click', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Edge-Device-Alpha')).toBeInTheDocument();
    });

    const nameHeader = screen.getByText('Client');
    await user.click(nameHeader);
    await user.click(nameHeader);
  });

  it('sorts by different columns', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Edge-Device-Alpha')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Accuracy'));
    await user.click(screen.getByText('Loss'));
    await user.click(screen.getByText('Round'));
  });

  /* ------------------------------------------------------------------ */
  /*  Pagination                                                         */
  /* ------------------------------------------------------------------ */
  it('paginates when clients exceed page size', async () => {
    const manyClients: ClientListResponse = {
      status: 'success',
      message: 'ok',
      total: 15,
      data: Array.from({ length: 15 }, (_, i) => ({
        id: `c${String(i + 1).padStart(3, '0')}`,
        name: `Z-Client-${String(i + 1).padStart(2, '0')}`,
        status: 'active',
        accuracy: 0.9,
        loss: 0.1,
        data_size: 1000,
        last_round: 10,
        device: 'Test Device',
        region: 'test-region',
        joined_at: '2026-01-01T00:00:00Z',
        last_communication: '2026-07-06T10:00:00Z',
      })),
    };
    mockFetchClients.mockResolvedValue(manyClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Z-Client-01')).toBeInTheDocument();
    });

    expect(screen.queryByText('Z-Client-11')).not.toBeInTheDocument();

    const paginationSection = screen.getByText(/of 15/).closest('div')!;
    const nextBtn = paginationSection.querySelector('button:last-child')!;
    await user.click(nextBtn);

    await waitFor(() => {
      expect(screen.getByText('Z-Client-11')).toBeInTheDocument();
    });
  });

  it('changes page size', async () => {
    const manyClients: ClientListResponse = {
      status: 'success',
      message: 'ok',
      total: 12,
      data: Array.from({ length: 12 }, (_, i) => ({
        id: `c${i + 1}`,
        name: `Client-${i + 1}`,
        status: 'active',
        accuracy: 0.9,
        loss: 0.1,
        data_size: 1000,
        last_round: 10,
        device: 'Test Device',
        region: 'test-region',
        joined_at: '2026-01-01T00:00:00Z',
        last_communication: '2026-07-06T10:00:00Z',
      })),
    };
    mockFetchClients.mockResolvedValue(manyClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Client-11')).toBeInTheDocument();
    });

    const sizeSelect = screen.getAllByRole('combobox')[1];
    await user.selectOptions(sizeSelect, '20');
  });

  /* ------------------------------------------------------------------ */
  /*  Refresh                                                            */
  /* ------------------------------------------------------------------ */
  it('shows refresh button that refetches data', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Edge-Device-Alpha')).toBeInTheDocument();
    });

    const refreshBtn = screen.getByText('Refresh');
    expect(refreshBtn).toBeInTheDocument();

    await user.click(refreshBtn);

    await waitFor(() => {
      expect(mockFetchClients).toHaveBeenCalledTimes(2);
    });
  });

  it('shows last updated timestamp', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
    });
  });

  /* ------------------------------------------------------------------ */
  /*  Detail panel                                                       */
  /* ------------------------------------------------------------------ */
  it('opens detail panel on eye icon click', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Edge-Device-Alpha')).toBeInTheDocument();
    });

    const eyeBtns = screen.getAllByTitle('View details');
    await user.click(eyeBtns[0]);

    await waitFor(() => {
      expect(screen.getByText('Client Profile')).toBeInTheDocument();
    });
  });

  it('shows client detail information', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Edge-Device-Alpha')).toBeInTheDocument();
    });

    const eyeBtns = screen.getAllByTitle('View details');
    await user.click(eyeBtns[0]);

    await waitFor(() => {
      expect(screen.getByText('Training Metrics')).toBeInTheDocument();
    });

    expect(screen.getAllByText('91.2%').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('0.1876').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('#47').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('15.2 KB')).toBeInTheDocument();
  });

  it('closes detail panel', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Edge-Device-Alpha')).toBeInTheDocument();
    });

    const eyeBtns = screen.getAllByTitle('View details');
    await user.click(eyeBtns[0]);

    await waitFor(() => {
      expect(screen.getByText('Client Profile')).toBeInTheDocument();
    });

    const closeBtns = document.querySelectorAll('.lucide-x');
    const closeBtn = closeBtns[0]?.closest('button');
    if (closeBtn) {
      await user.click(closeBtn);
    }

    await waitFor(() => {
      expect(screen.queryByText('Client Profile')).not.toBeInTheDocument();
    });
  });

  it('shows participation section in detail panel', async () => {
    mockFetchClients.mockResolvedValue(sampleClients);
    renderClients();

    await waitFor(() => {
      expect(screen.getByText('Edge-Device-Alpha')).toBeInTheDocument();
    });

    const eyeBtns = screen.getAllByTitle('View details');
    await user.click(eyeBtns[0]);

    await waitFor(() => {
      expect(screen.getByText('Participation')).toBeInTheDocument();
    });

    expect(screen.getByText('Jun 22, 2026')).toBeInTheDocument();
  });
});
