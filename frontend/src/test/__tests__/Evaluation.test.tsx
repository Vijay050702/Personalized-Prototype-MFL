import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

import { Evaluation } from '../../pages/Evaluation';
import * as evaluationApi from '../../api/evaluation';
import type { EvaluationSummary, ExperimentListResponse } from '../../types';

vi.mock('../../api/evaluation');

const mockFetchEval = evaluationApi.fetchEvaluation as ReturnType<typeof vi.fn>;
const mockFetchExp = evaluationApi.fetchExperiments as ReturnType<typeof vi.fn>;

const mockEvalData: EvaluationSummary = {
  status: 'success',
  message: 'Evaluation results retrieved',
  data: {
    accuracy: 0.8734,
    precision: 0.8654,
    recall: 0.8812,
    f1_score: 0.8732,
    auc_roc: 0.9245,
    client_id: 'global',
    round: 47,
    samples_evaluated: 10000,
  },
};

const mockExperiments: ExperimentListResponse = {
  status: 'success',
  message: 'Experiments retrieved',
  data: [
    {
      id: 'exp-001',
      name: 'FedAvg-Baseline-CIFAR10',
      status: 'running',
      algorithm: 'FedAvg',
      num_clients: 10,
      total_rounds: 100,
      current_round: 47,
      best_accuracy: 0.8734,
      started_at: '2026-07-04T00:00:00Z',
      completed_at: null,
    },
    {
      id: 'exp-002',
      name: 'FedProx-NonIID-EMNIST',
      status: 'completed',
      algorithm: 'FedProx',
      num_clients: 15,
      total_rounds: 80,
      current_round: 80,
      best_accuracy: 0.9213,
      started_at: '2026-07-01T00:00:00Z',
      completed_at: '2026-07-05T00:00:00Z',
    },
    {
      id: 'exp-003',
      name: 'SCAFFOLD-Heterogeneous',
      status: 'completed',
      algorithm: 'SCAFFOLD',
      num_clients: 20,
      total_rounds: 120,
      current_round: 120,
      best_accuracy: 0.9432,
      started_at: '2026-06-29T00:00:00Z',
      completed_at: '2026-07-03T00:00:00Z',
    },
    {
      id: 'exp-004',
      name: 'Personalized-FL-Prototype',
      status: 'pending',
      algorithm: 'pFedProto',
      num_clients: 12,
      total_rounds: 150,
      current_round: 0,
      best_accuracy: 0.0,
      started_at: '2026-07-06T00:00:00Z',
      completed_at: null,
    },
  ],
  total: 4,
};

function renderEvaluation() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, retryDelay: 0 },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Evaluation />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('Evaluation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('shows loading skeleton while fetching', () => {
    mockFetchEval.mockReturnValue(new Promise(() => {}));
    mockFetchExp.mockReturnValue(new Promise(() => {}));
    renderEvaluation();
    const skeletons = document.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBeGreaterThanOrEqual(5);
  });

  it('renders stat cards with real evaluation data on success', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getAllByText('Accuracy').length).toBeGreaterThanOrEqual(1);
    });

    expect(screen.getAllByText('87.34%').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('86.54%').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('88.12%').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('87.32%').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('92.45%').length).toBeGreaterThanOrEqual(1);
  });

  it('renders evaluation metadata section', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Evaluation Metadata')).toBeInTheDocument();
    });

    expect(screen.getAllByText('global').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('R-47').length).toBeGreaterThanOrEqual(1);
  });

  it('shows comm round and samples stat cards', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getAllByText('R-47').length).toBeGreaterThanOrEqual(1);
    });

    expect(screen.getAllByText('10,000').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Comm Round').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Samples').length).toBeGreaterThanOrEqual(1);
  });

  it('renders charts section', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Precision / Recall / F1')).toBeInTheDocument();
    });

    expect(screen.getByText('Performance Radar')).toBeInTheDocument();
  });

  it('renders baseline comparison with best model', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Baseline Comparison')).toBeInTheDocument();
    });

    expect(screen.getByText('Best Model')).toBeInTheDocument();
    expect(screen.getByText('SCAFFOLD-Heterogeneous')).toBeInTheDocument();
    expect(screen.getAllByText('94.32%').length).toBeGreaterThanOrEqual(1);
  });

  it('renders experiment table with all experiment rows', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Experiment Runs')).toBeInTheDocument();
    });

    expect(screen.getByText('FedAvg-Baseline-CIFAR10')).toBeInTheDocument();
    expect(screen.getByText('FedProx-NonIID-EMNIST')).toBeInTheDocument();
    expect(screen.getByText('SCAFFOLD-Heterogeneous')).toBeInTheDocument();
    expect(screen.getByText('Personalized-FL-Prototype')).toBeInTheDocument();
  });

  it('shows statuses correctly in experiment table', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getAllByText('running').length).toBeGreaterThanOrEqual(1);
    });

    expect(screen.getAllByText('completed').length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText('pending').length).toBeGreaterThanOrEqual(1);
  });

  it('shows error state on evaluation failure', async () => {
    mockFetchEval.mockRejectedValue(new Error('Server error. Please try again later.'));
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Failed to load evaluation data')).toBeInTheDocument();
    });

    expect(screen.getByText('Server error. Please try again later.')).toBeInTheDocument();
    expect(screen.getByText('Try Again')).toBeInTheDocument();
  });

  it('shows error state on backend unavailable', async () => {
    mockFetchEval.mockRejectedValue(
      new Error('Backend is unavailable. Please check your connection.'),
    );
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Failed to load evaluation data')).toBeInTheDocument();
    });

    expect(
      screen.getByText('Backend is unavailable. Please check your connection.'),
    ).toBeInTheDocument();
  });

  it('shows empty state when evaluation response has no data', async () => {
    mockFetchEval.mockResolvedValue({
      status: 'success',
      message: 'ok',
      data: null,
    });
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('No evaluation data available')).toBeInTheDocument();
    });
    expect(screen.getByText('Retry')).toBeInTheDocument();
  });

  it('shows empty state on experiments error', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockRejectedValue(new Error('Experiments unavailable'));
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Failed to load evaluation data')).toBeInTheDocument();
    });
  });

  it('calls fetchEvaluation and fetchExperiments on mount', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(mockFetchEval).toHaveBeenCalledTimes(1);
    });
    expect(mockFetchExp).toHaveBeenCalledTimes(1);
  });

  it('retries on error when clicking Try Again', async () => {
    mockFetchEval.mockRejectedValue(new Error('Server error'));
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Failed to load evaluation data')).toBeInTheDocument();
    });

    mockFetchEval.mockResolvedValue(mockEvalData);

    const retryButton = screen.getByText('Try Again');
    await userEvent.click(retryButton);

    await waitFor(() => {
      expect(screen.getAllByText('87.34%').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('shows manual refresh button that refetches data', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getAllByText('87.34%').length).toBeGreaterThanOrEqual(1);
    });

    const refreshButton = screen.getByText('Refresh');
    expect(refreshButton).toBeInTheDocument();

    await userEvent.click(refreshButton);

    await waitFor(() => {
      expect(mockFetchEval).toHaveBeenCalledTimes(2);
    });
  });

  it('filters experiments by search', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('FedAvg-Baseline-CIFAR10')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText('Search experiments...');
    await userEvent.type(searchInput, 'FedProx');

    await waitFor(() => {
      expect(screen.queryByText('FedAvg-Baseline-CIFAR10')).not.toBeInTheDocument();
    });

    expect(screen.getByText('FedProx-NonIID-EMNIST')).toBeInTheDocument();
  });

  it('filters experiments by status', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('FedAvg-Baseline-CIFAR10')).toBeInTheDocument();
    });

    const statusSelect = screen.getAllByRole('combobox')[0];
    await userEvent.selectOptions(statusSelect, 'completed');

    await waitFor(() => {
      expect(screen.queryByText('FedAvg-Baseline-CIFAR10')).not.toBeInTheDocument();
    });

    expect(screen.getByText('FedProx-NonIID-EMNIST')).toBeInTheDocument();
    expect(screen.getByText('SCAFFOLD-Heterogeneous')).toBeInTheDocument();
  });

  it('filters experiments by algorithm', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('FedAvg-Baseline-CIFAR10')).toBeInTheDocument();
    });

    const selects = screen.getAllByRole('combobox');
    const algoSelect = selects[1];
    await userEvent.selectOptions(algoSelect, 'FedProx');

    await waitFor(() => {
      expect(screen.queryByText('FedAvg-Baseline-CIFAR10')).not.toBeInTheDocument();
    });

    expect(screen.getByText('FedProx-NonIID-EMNIST')).toBeInTheDocument();
  });

  it('clears all filters', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('FedAvg-Baseline-CIFAR10')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText('Search experiments...');
    await userEvent.type(searchInput, 'XYZ');

    await waitFor(() => {
      expect(screen.queryByText('FedAvg-Baseline-CIFAR10')).not.toBeInTheDocument();
    });

    const clearButton = screen.getByText('Clear');
    await userEvent.click(clearButton);

    await waitFor(() => {
      expect(screen.getByText('FedAvg-Baseline-CIFAR10')).toBeInTheDocument();
    });
  });

  it('shows no experiments match state when filters yield no results', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Experiment Runs')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText('Search experiments...');
    await userEvent.type(searchInput, 'NONEXISTENT');

    await waitFor(() => {
      expect(screen.getByText('No experiments match your filters.')).toBeInTheDocument();
    });
  });

  it('opens and closes detail panel on eye icon click', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('FedAvg-Baseline-CIFAR10')).toBeInTheDocument();
    });

    const viewButtons = screen.getAllByTitle('View details');
    await userEvent.click(viewButtons[0]);

    await waitFor(() => {
      expect(screen.getByText('Experiment Details')).toBeInTheDocument();
    });

    expect(screen.getByText('exp-003')).toBeInTheDocument();
    expect(screen.getAllByText('SCAFFOLD').length).toBeGreaterThanOrEqual(1);

    const closeButton = document.querySelector('[class*="sticky"] button');
    if (closeButton) {
      await userEvent.click(closeButton);
      await waitFor(() => {
        expect(screen.queryByText('Experiment Details')).not.toBeInTheDocument();
      });
    }
  });

  it('sorts experiments by column click', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Experiment Runs')).toBeInTheDocument();
    });

    const nameHeader = screen.getByText('Name').closest('th');
    expect(nameHeader).toBeInTheDocument();

    if (nameHeader) {
      await userEvent.click(nameHeader);
    }
  });

  it('shows title and last updated timestamp', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Model Evaluation')).toBeInTheDocument();
    });

    const lastUpdatedElements = screen.queryAllByText(/Last updated/);
    expect(lastUpdatedElements.length).toBeGreaterThanOrEqual(1);
  });

  it('handles errors from both queries', async () => {
    mockFetchEval.mockRejectedValue(new Error('Backend is unavailable. Please check your connection.'));
    mockFetchExp.mockRejectedValue(new Error('Backend is unavailable. Please check your connection.'));
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Failed to load evaluation data')).toBeInTheDocument();
    });

    expect(
      screen.getByText('Backend is unavailable. Please check your connection.'),
    ).toBeInTheDocument();
  });

  it('handles 404 error gracefully', async () => {
    mockFetchEval.mockRejectedValue(new Error('Resource not found.'));
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Failed to load evaluation data')).toBeInTheDocument();
    });

    expect(screen.getByText('Resource not found.')).toBeInTheDocument();
  });

  it('handles 422 error gracefully', async () => {
    mockFetchEval.mockRejectedValue(new Error('Invalid request parameters.'));
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Failed to load evaluation data')).toBeInTheDocument();
    });

    expect(screen.getByText('Invalid request parameters.')).toBeInTheDocument();
  });

  it('shows evaluation endpoint not found error', async () => {
    mockFetchEval.mockRejectedValue(new Error('Evaluation endpoint not found.'));
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Failed to load evaluation data')).toBeInTheDocument();
    });

    expect(screen.getByText('Evaluation endpoint not found.')).toBeInTheDocument();
  });

  it('shows evaluation time in a stat card', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Evaluation Time')).toBeInTheDocument();
    });
  });

  it('shows Avg Accuracy in evaluation metadata section', async () => {
    mockFetchEval.mockResolvedValue(mockEvalData);
    mockFetchExp.mockResolvedValue(mockExperiments);
    renderEvaluation();

    await waitFor(() => {
      expect(screen.getByText('Evaluation Metadata')).toBeInTheDocument();
    });

    expect(screen.getByText('Avg Accuracy')).toBeInTheDocument();
  });
});
