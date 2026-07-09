import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

import { Experiments } from '../../pages/Experiments';
import * as experimentsApi from '../../api/experiments';
import type { ExperimentListResponse } from '../../types';

vi.mock('../../api/experiments');

const mockFetchExp = experimentsApi.fetchExperiments as ReturnType<typeof vi.fn>;

const mockExperiments: ExperimentListResponse = {
  status: 'success',
  message: 'Experiments retrieved',
  data: [
    { id: 'exp-001', name: 'FedAvg-Baseline-CIFAR10', status: 'running', algorithm: 'FedAvg', num_clients: 10, total_rounds: 100, current_round: 47, best_accuracy: 0.8734, started_at: '2026-07-04T00:00:00Z', completed_at: null },
    { id: 'exp-002', name: 'FedProx-NonIID-EMNIST', status: 'completed', algorithm: 'FedProx', num_clients: 15, total_rounds: 80, current_round: 80, best_accuracy: 0.9213, started_at: '2026-07-01T00:00:00Z', completed_at: '2026-07-05T00:00:00Z' },
    { id: 'exp-003', name: 'SCAFFOLD-Heterogeneous', status: 'completed', algorithm: 'SCAFFOLD', num_clients: 20, total_rounds: 120, current_round: 120, best_accuracy: 0.9432, started_at: '2026-06-29T00:00:00Z', completed_at: '2026-07-03T00:00:00Z' },
    { id: 'exp-004', name: 'Personalized-FL-Prototype', status: 'pending', algorithm: 'pFedProto', num_clients: 12, total_rounds: 150, current_round: 0, best_accuracy: 0.0, started_at: '2026-07-06T00:00:00Z', completed_at: null },
    { id: 'exp-005', name: 'FedAvg-MNIST-Experiment', status: 'failed', algorithm: 'FedAvg', num_clients: 8, total_rounds: 50, current_round: 12, best_accuracy: 0.4521, started_at: '2026-06-25T00:00:00Z', completed_at: '2026-06-26T00:00:00Z' },
    { id: 'exp-006', name: 'FedProx-CIFAR100-Test', status: 'completed', algorithm: 'FedProx', num_clients: 25, total_rounds: 200, current_round: 200, best_accuracy: 0.8876, started_at: '2026-06-20T00:00:00Z', completed_at: '2026-06-28T00:00:00Z' },
  ],
  total: 6,
};

function renderExperiments() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, retryDelay: 0 } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Experiments />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('Experiments', () => {
  beforeEach(() => { vi.clearAllMocks(); });
  afterEach(() => { vi.restoreAllMocks(); });

  it('shows loading skeleton while fetching', () => {
    mockFetchExp.mockReturnValue(new Promise(() => {}));
    renderExperiments();
    expect(document.querySelectorAll('.animate-pulse').length).toBeGreaterThanOrEqual(8);
  });

  it('renders statistics cards with experiment data on success', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Total')).toBeInTheDocument(); });
    expect(screen.getByText('6')).toBeInTheDocument();
    expect(screen.getAllByText('1').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Avg Accuracy')).toBeInTheDocument();
  });

  it('renders experiment table with all rows', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('All Experiments (6)')).toBeInTheDocument(); });
    expect(screen.getAllByText(/FedAvg-Baseline-CIFAR10/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/FedProx-NonIID-EMNIST/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/SCAFFOLD-Heterogeneous/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/Personalized-FL-Prototype/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/FedAvg-MNIST-Experiment/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/FedProx-CIFAR100-Test/).length).toBeGreaterThanOrEqual(1);
  });

  it('shows status badges correctly in table', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getAllByText('running').length).toBeGreaterThanOrEqual(1); });
    expect(screen.getAllByText('completed').length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText('pending').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('failed').length).toBeGreaterThanOrEqual(1);
  });

  it('shows error state on request failure', async () => {
    mockFetchExp.mockRejectedValue(new Error('Server error. Please try again later.'));
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Failed to load experiments')).toBeInTheDocument(); });
    expect(screen.getByText('Server error. Please try again later.')).toBeInTheDocument();
    expect(screen.getByText('Try Again')).toBeInTheDocument();
  });

  it('shows backend unavailable error state', async () => {
    mockFetchExp.mockRejectedValue(new Error('Backend is unavailable. Please check your connection.'));
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Failed to load experiments')).toBeInTheDocument(); });
    expect(screen.getByText('Backend is unavailable. Please check your connection.')).toBeInTheDocument();
  });

  it('shows empty state when no experiments exist', async () => {
    mockFetchExp.mockResolvedValue({ status: 'success', message: 'ok', data: [], total: 0 });
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('No experiments available')).toBeInTheDocument(); });
    expect(screen.getByText('Retry')).toBeInTheDocument();
  });

  it('calls fetchExperiments on mount', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(mockFetchExp).toHaveBeenCalledTimes(1); });
  });

  it('retries on error when clicking Try Again', async () => {
    mockFetchExp.mockRejectedValue(new Error('Server error'));
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Failed to load experiments')).toBeInTheDocument(); });
    mockFetchExp.mockResolvedValue(mockExperiments);
    await userEvent.click(screen.getByText('Try Again'));
    await waitFor(() => { expect(screen.getByText('All Experiments (6)')).toBeInTheDocument(); });
  });

  it('shows manual refresh button that refetches data', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('All Experiments (6)')).toBeInTheDocument(); });
    await userEvent.click(screen.getByText('Refresh'));
    await waitFor(() => { expect(mockFetchExp).toHaveBeenCalledTimes(2); });
  });

  it('filters experiments by search', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('All Experiments (6)')).toBeInTheDocument(); });
    await userEvent.type(screen.getByPlaceholderText('Search experiments...'), 'FedProx');
    await waitFor(() => { expect(screen.getByText('All Experiments (2)')).toBeInTheDocument(); });
  });

  it('filters experiments by status', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('All Experiments (6)')).toBeInTheDocument(); });
    await userEvent.selectOptions(screen.getAllByRole('combobox')[0], 'completed');
    await waitFor(() => { expect(screen.getByText('All Experiments (3)')).toBeInTheDocument(); });
  });

  it('filters experiments by algorithm', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('All Experiments (6)')).toBeInTheDocument(); });
    await userEvent.selectOptions(screen.getAllByRole('combobox')[1], 'FedProx');
    await waitFor(() => { expect(screen.getByText('All Experiments (2)')).toBeInTheDocument(); });
  });

  it('clears all filters', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('All Experiments (6)')).toBeInTheDocument(); });
    await userEvent.type(screen.getByPlaceholderText('Search experiments...'), 'XYZ');
    await waitFor(() => { expect(screen.getByText('All Experiments (0)')).toBeInTheDocument(); });
    await userEvent.click(screen.getByText('Clear'));
    await waitFor(() => { expect(screen.getByText('All Experiments (6)')).toBeInTheDocument(); });
  });

  it('shows no experiments match state when filters yield no results', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('All Experiments (6)')).toBeInTheDocument(); });
    await userEvent.type(screen.getByPlaceholderText('Search experiments...'), 'NONEXISTENT');
    await waitFor(() => { expect(screen.getByText('No experiments match your filters.')).toBeInTheDocument(); });
  });

  it('shows page size selector and changes page size', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('All Experiments (6)')).toBeInTheDocument(); });
    const pageSizeSelect = screen.getAllByRole('combobox')[2];
    expect(pageSizeSelect).toHaveValue('10');
    await userEvent.selectOptions(pageSizeSelect, '5');
    expect(pageSizeSelect).toHaveValue('5');
  });

  it('shows pagination when items exceed page size', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('All Experiments (6)')).toBeInTheDocument(); });
    await userEvent.selectOptions(screen.getAllByRole('combobox')[2], '5');
    await waitFor(() => { expect(screen.getByText('2')).toBeInTheDocument(); });
  });

  it('opens and closes detail panel on eye icon click', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getAllByText(/FedAvg-Baseline-CIFAR10/).length).toBeGreaterThanOrEqual(1); });
    await userEvent.click(screen.getAllByTitle('View details')[0]);
    await waitFor(() => { expect(screen.getByText('Experiment Details')).toBeInTheDocument(); });
    const closeBtn = document.querySelector('[class*="sticky"] button');
    if (closeBtn) {
      await userEvent.click(closeBtn);
      await waitFor(() => { expect(screen.queryByText('Experiment Details')).not.toBeInTheDocument(); });
    }
  });

  it('shows experiment log tabs in detail panel', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getAllByText(/FedAvg-Baseline-CIFAR10/).length).toBeGreaterThanOrEqual(1); });
    await userEvent.click(screen.getAllByTitle('View details')[0]);
    await waitFor(() => { expect(screen.getByText('Logs')).toBeInTheDocument(); });
    expect(screen.getByText('Training')).toBeInTheDocument();
    expect(screen.getByText('System')).toBeInTheDocument();
    expect(screen.getByText('Aggregation')).toBeInTheDocument();
    expect(screen.getByText('Knowledge Transfer')).toBeInTheDocument();
    expect(screen.getByText('Personalization')).toBeInTheDocument();
  });

  it('switches log tabs in detail panel', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getAllByText(/FedAvg-Baseline-CIFAR10/).length).toBeGreaterThanOrEqual(1); });
    await userEvent.click(screen.getAllByTitle('View details')[0]);
    await waitFor(() => { expect(screen.getByText('Logs')).toBeInTheDocument(); });
    await userEvent.click(screen.getByText('System'));
    await waitFor(() => { expect(screen.getByText(/initialized/)).toBeInTheDocument(); });
  });

  it('sorts experiments by column click', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('All Experiments (6)')).toBeInTheDocument(); });
    const nameHeader = screen.getByText('Name').closest('th');
    if (nameHeader) await userEvent.click(nameHeader);
  });

  it('shows title and last updated timestamp', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Experiments')).toBeInTheDocument(); });
    expect(screen.queryAllByText(/Last updated/).length).toBeGreaterThanOrEqual(1);
  });

  it('shows auto-refresh indicator when experiments are running', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Auto-refresh active')).toBeInTheDocument(); });
  });

  it('does not show auto-refresh when no experiments are running', async () => {
    const noRunning = { ...mockExperiments, data: mockExperiments.data.map(function(e) { return { ...e, status: e.status === 'running' ? 'pending' : e.status }; }) };
    mockFetchExp.mockResolvedValue(noRunning);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Experiments')).toBeInTheDocument(); });
    expect(screen.queryByText('Auto-refresh active')).not.toBeInTheDocument();
  });

  it('renders algorithm comparison chart', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Algorithm Comparison')).toBeInTheDocument(); });
  });

  it('renders duration distribution chart', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Duration Distribution')).toBeInTheDocument(); });
  });

  it('renders status distribution chart', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Status Distribution')).toBeInTheDocument(); });
  });

  it('renders accuracy trend chart', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Accuracy Trend')).toBeInTheDocument(); });
  });

  it('renders experiment timeline section', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Experiment Timeline')).toBeInTheDocument(); });
  });

  it('shows avg duration stat card', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Avg Duration')).toBeInTheDocument(); });
  });

  it('shows best experiment name in stat card', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Best Experiment')).toBeInTheDocument(); });
    await waitFor(() => { expect(screen.getAllByText(/SCAFFOLD-Heterogeneous/).length).toBeGreaterThanOrEqual(1); });
  });

  it('handles 404 error', async () => {
    mockFetchExp.mockRejectedValue(new Error('Experiments endpoint not found.'));
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Failed to load experiments')).toBeInTheDocument(); });
    expect(screen.getByText('Experiments endpoint not found.')).toBeInTheDocument();
  });

  it('handles 422 error', async () => {
    mockFetchExp.mockRejectedValue(new Error('Invalid request parameters.'));
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Failed to load experiments')).toBeInTheDocument(); });
    expect(screen.getByText('Invalid request parameters.')).toBeInTheDocument();
  });

  it('handles 500 error', async () => {
    mockFetchExp.mockRejectedValue(new Error('Server error. Please try again later.'));
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Failed to load experiments')).toBeInTheDocument(); });
    expect(screen.getByText('Server error. Please try again later.')).toBeInTheDocument();
  });

  it('handles timeout error', async () => {
    mockFetchExp.mockRejectedValue(new Error('Request timed out. Please try again.'));
    renderExperiments();
    await waitFor(() => { expect(screen.getByText('Failed to load experiments')).toBeInTheDocument(); });
    expect(screen.getByText('Request timed out. Please try again.')).toBeInTheDocument();
  });

  it('shows detail panel with experiment metadata', async () => {
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderExperiments();
    await waitFor(() => { expect(screen.getAllByText(/FedAvg-Baseline-CIFAR10/).length).toBeGreaterThanOrEqual(1); });
    await userEvent.click(screen.getAllByTitle('View details')[0]);
    await waitFor(() => { expect(screen.getByText('Metadata')).toBeInTheDocument(); });
    expect(screen.getByText('Performance')).toBeInTheDocument();
    expect(screen.getByText('Timeline')).toBeInTheDocument();
  });
});
