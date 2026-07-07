import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

import { Training } from '../../pages/Training';
import * as trainingApi from '../../api/training';
import type {
  TrainingStatusSummary,
  TrainingConfigSummary,
  TrainingControlResponse,
} from '../../types';

vi.mock('../../api/training');

const mockFetchStatus = trainingApi.fetchTrainingStatus as ReturnType<typeof vi.fn>;
const mockFetchConfig = trainingApi.fetchTrainingConfig as ReturnType<typeof vi.fn>;
const mockStart = trainingApi.startTraining as ReturnType<typeof vi.fn>;
const mockPause = trainingApi.pauseTraining as ReturnType<typeof vi.fn>;
const mockResume = trainingApi.resumeTraining as ReturnType<typeof vi.fn>;
const mockStop = trainingApi.stopTraining as ReturnType<typeof vi.fn>;
const mockCheckpoint = trainingApi.saveCheckpoint as ReturnType<typeof vi.fn>;
const mockUpdateConfig = trainingApi.updateTrainingConfig as ReturnType<typeof vi.fn>;

const idleStatusData: TrainingStatusSummary = {
  status: 'success',
  message: 'Training status retrieved',
  data: {
    status: 'idle',
    current_round: 0,
    total_rounds: 100,
    epochs_completed: 0,
    total_epochs: 5,
    current_loss: 0,
    current_accuracy: 0,
    learning_rate: 0.001,
    clients_participating: 0,
    aggregation_algorithm: 'FedAvg',
    time_elapsed_seconds: 0,
    estimated_time_remaining: 0,
  },
};

const runningStatusData: TrainingStatusSummary = {
  status: 'success',
  message: 'Training status retrieved',
  data: {
    status: 'running',
    current_round: 47,
    total_rounds: 100,
    epochs_completed: 3,
    total_epochs: 5,
    current_loss: 0.2341,
    current_accuracy: 0.8734,
    learning_rate: 0.001,
    clients_participating: 12,
    aggregation_algorithm: 'FedAvg',
    time_elapsed_seconds: 8423.5,
    estimated_time_remaining: 9567.3,
  },
};

const pausedStatusData: TrainingStatusSummary = {
  status: 'success',
  message: 'Training status retrieved',
  data: {
    ...runningStatusData.data,
    status: 'paused',
  },
};

const completedStatusData: TrainingStatusSummary = {
  status: 'success',
  message: 'Training status retrieved',
  data: {
    ...idleStatusData.data,
    status: 'completed',
    current_round: 100,
    current_accuracy: 0.9421,
    current_loss: 0.0452,
    time_elapsed_seconds: 36000,
  },
};

const mockConfigData: TrainingConfigSummary = {
  status: 'success',
  message: 'Configuration retrieved',
  data: {
    dataset: 'cifar10',
    client_count: 10,
    communication_rounds: 100,
    local_epochs: 5,
    batch_size: 32,
    learning_rate: 0.001,
    optimizer: 'adam',
    scheduler: 'cosine_annealing',
    aggregation_strategy: 'FedAvg',
    knowledge_transfer_enabled: true,
    personalization_enabled: true,
  },
};

const successResponse: TrainingControlResponse = {
  status: 'success',
  message: 'Operation completed',
};

function renderTraining() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, retryDelay: 0 },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Training />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

function findButtonByExactText(text: string): HTMLButtonElement | null {
  const buttons = screen.getAllByRole('button');
  return buttons.find((b) => {
    const t = b.textContent?.trim() || '';
    return t === text || t.startsWith(text + ' ') || t.includes(text);
  }) as HTMLButtonElement ?? null;
}

describe('Training', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchConfig.mockResolvedValue(mockConfigData);
    mockStart.mockResolvedValue(successResponse);
    mockPause.mockResolvedValue(successResponse);
    mockResume.mockResolvedValue(successResponse);
    mockStop.mockResolvedValue(successResponse);
    mockCheckpoint.mockResolvedValue(successResponse);
    mockUpdateConfig.mockResolvedValue(mockConfigData);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Loading State', () => {
    it('shows loading skeleton while fetching status', () => {
      mockFetchStatus.mockReturnValue(new Promise(() => {}));
      renderTraining();
      const skeletons = document.querySelectorAll('.animate-pulse');
      expect(skeletons.length).toBeGreaterThanOrEqual(4);
    });
  });

  describe('Error State', () => {
    it('shows error state on API failure', async () => {
      mockFetchStatus.mockRejectedValue(new Error('Server error. Please try again later.'));
      renderTraining();

      await waitFor(() => {
        expect(screen.getByText('Failed to load training data')).toBeInTheDocument();
      });
      expect(screen.getByText('Server error. Please try again later.')).toBeInTheDocument();
      expect(screen.getByText('Try Again')).toBeInTheDocument();
    });

    it('shows error state when backend is unavailable', async () => {
      mockFetchStatus.mockRejectedValue(
        new Error('Backend is unavailable. Please check your connection.'),
      );
      renderTraining();

      await waitFor(() => {
        expect(screen.getByText('Failed to load training data')).toBeInTheDocument();
      });
      expect(
        screen.getByText('Backend is unavailable. Please check your connection.'),
      ).toBeInTheDocument();
    });

    it('retries on error when clicking Try Again', async () => {
      mockFetchStatus.mockRejectedValue(new Error('Server error'));
      renderTraining();

      await waitFor(() => {
        expect(screen.getByText('Failed to load training data')).toBeInTheDocument();
      });

      mockFetchStatus.mockResolvedValue(runningStatusData);

      const retryButton = screen.getByText('Try Again');
      await userEvent.click(retryButton);

      await waitFor(() => {
        const elems = screen.getAllByText('47 / 100');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });
    });
  });

  describe('Empty State', () => {
    it('shows empty state when response has null data', async () => {
      mockFetchStatus.mockResolvedValue({
        status: 'success',
        message: 'ok',
        data: null,
      });
      renderTraining();

      await waitFor(() => {
        expect(screen.getByText('No training data available')).toBeInTheDocument();
      });
      expect(screen.getByText('Retry')).toBeInTheDocument();
    });
  });

  describe('Success State - Status Cards', () => {
    it('renders stat cards with running data', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('47 / 100');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const lossElems = screen.getAllByText('0.2341');
      expect(lossElems.length).toBeGreaterThanOrEqual(1);

      const accElems = screen.getAllByText('87.34%');
      expect(accElems.length).toBeGreaterThanOrEqual(1);
    });

    it('renders stat cards with idle data', async () => {
      mockFetchStatus.mockResolvedValue(idleStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('Idle');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const rounds = screen.getAllByText('0 / 100');
      expect(rounds.length).toBeGreaterThanOrEqual(1);

      const epochs = screen.getAllByText('0 / 5');
      expect(epochs.length).toBeGreaterThanOrEqual(1);
    });

    it('renders status cards with completed data', async () => {
      mockFetchStatus.mockResolvedValue(completedStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('Completed');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const rounds = screen.getAllByText('100 / 100');
      expect(rounds.length).toBeGreaterThanOrEqual(1);

      const accElems = screen.getAllByText('94.21%');
      expect(accElems.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Current Status Section', () => {
    it('renders current status card with backend data', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      renderTraining();

      await waitFor(() => {
        expect(screen.getByText('Current Status')).toBeInTheDocument();
      });

      const fedAvgElems = screen.getAllByText('FedAvg');
      expect(fedAvgElems.length).toBeGreaterThanOrEqual(1);

      expect(screen.getByText('2h 20m 23s')).toBeInTheDocument();
    });
  });

  describe('Live Progress Section', () => {
    it('renders progress bars and metric boxes', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      renderTraining();

      await waitFor(() => {
        expect(screen.getByText('Communication Round')).toBeInTheDocument();
      });

      expect(screen.getByText('Round Progress')).toBeInTheDocument();
      expect(screen.getByText('Epoch Progress')).toBeInTheDocument();
    });
  });

  describe('Manual Refresh', () => {
    it('refetches data when Refresh button is clicked', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('47 / 100');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(mockFetchStatus).toHaveBeenCalledTimes(1);

      const refreshButton = screen.getByText('Refresh');
      await userEvent.click(refreshButton);

      await waitFor(() => {
        expect(mockFetchStatus).toHaveBeenCalledTimes(2);
      });
    });
  });

  describe('Training Controls - Button States', () => {
    it('disables Start, enables Stop/Pause/Resume/Checkpoint correctly when idle', async () => {
      mockFetchStatus.mockResolvedValue(idleStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('Idle');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const buttons = screen.getAllByRole('button');
      const startBtn = buttons.find((b) => b.textContent?.includes('Start'));
      const pauseBtn = buttons.find((b) => b.textContent?.includes('Pause'));
      const resumeBtn = buttons.find((b) => b.textContent?.includes('Resume'));
      const stopBtn = buttons.find((b) => b.textContent?.includes('Stop'));
      const checkpointBtn = buttons.find((b) => b.textContent?.includes('Checkpoint'));

      expect(startBtn).not.toBeDisabled();
      expect(pauseBtn).toBeDisabled();
      expect(resumeBtn).toBeDisabled();
      expect(stopBtn).toBeDisabled();
      expect(checkpointBtn).toBeDisabled();
    });

    it('enables Pause/Stop/Checkpoint, disables Start/Resume when running', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('Running');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const buttons = screen.getAllByRole('button');
      const startBtn = buttons.find((b) => b.textContent?.includes('Start'));
      const pauseBtn = buttons.find((b) => b.textContent?.includes('Pause'));
      const resumeBtn = buttons.find((b) => b.textContent?.includes('Resume'));
      const stopBtn = buttons.find((b) => b.textContent?.includes('Stop'));
      const checkpointBtn = buttons.find((b) => b.textContent?.includes('Checkpoint'));

      expect(startBtn).toBeDisabled();
      expect(pauseBtn).not.toBeDisabled();
      expect(resumeBtn).toBeDisabled();
      expect(stopBtn).not.toBeDisabled();
      expect(checkpointBtn).not.toBeDisabled();
    });

    it('enables Resume/Stop, disables Start/Pause/Checkpoint when paused', async () => {
      mockFetchStatus.mockResolvedValue(pausedStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('Paused');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const buttons = screen.getAllByRole('button');
      const startBtn = buttons.find((b) => b.textContent?.includes('Start'));
      const pauseBtn = buttons.find((b) => b.textContent?.includes('Pause'));
      const resumeBtn = buttons.find((b) => b.textContent?.includes('Resume'));
      const stopBtn = buttons.find((b) => b.textContent?.includes('Stop'));
      const checkpointBtn = buttons.find((b) => b.textContent?.includes('Checkpoint'));

      expect(startBtn).toBeDisabled();
      expect(pauseBtn).toBeDisabled();
      expect(resumeBtn).not.toBeDisabled();
      expect(stopBtn).not.toBeDisabled();
      expect(checkpointBtn).toBeDisabled();
    });

    it('enables Start, disables Pause/Resume/Stop/Checkpoint when completed', async () => {
      mockFetchStatus.mockResolvedValue(completedStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('Completed');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const buttons = screen.getAllByRole('button');
      const startBtn = buttons.find((b) => b.textContent?.includes('Start'));
      const pauseBtn = buttons.find((b) => b.textContent?.includes('Pause'));
      const resumeBtn = buttons.find((b) => b.textContent?.includes('Resume'));
      const stopBtn = buttons.find((b) => b.textContent?.includes('Stop'));
      const checkpointBtn = buttons.find((b) => b.textContent?.includes('Checkpoint'));

      expect(startBtn).not.toBeDisabled();
      expect(pauseBtn).toBeDisabled();
      expect(resumeBtn).toBeDisabled();
      expect(stopBtn).toBeDisabled();
      expect(checkpointBtn).toBeDisabled();
    });
  });

  describe('Training Controls - Mutations', () => {
    it('calls startTraining when Start is clicked on idle state', async () => {
      mockFetchStatus.mockResolvedValue(idleStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('Idle');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const buttons = screen.getAllByRole('button');
      const startBtn = buttons.find((b) => b.textContent?.includes('Start'));
      expect(startBtn).not.toBeDisabled();
      await userEvent.click(startBtn!);

      await waitFor(() => {
        expect(mockStart).toHaveBeenCalledTimes(1);
      });
    });

    it('calls pauseTraining when Pause is clicked on running state', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('47 / 100');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const buttons = screen.getAllByRole('button');
      const pauseBtn = buttons.find((b) => b.textContent?.includes('Pause'));
      expect(pauseBtn).not.toBeDisabled();
      await userEvent.click(pauseBtn!);

      await waitFor(() => {
        expect(mockPause).toHaveBeenCalledTimes(1);
      });
    });

    it('calls resumeTraining when Resume is clicked on paused state', async () => {
      mockFetchStatus.mockResolvedValue(pausedStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('Paused');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const buttons = screen.getAllByRole('button');
      const resumeBtn = buttons.find((b) => b.textContent?.includes('Resume'));
      expect(resumeBtn).not.toBeDisabled();
      await userEvent.click(resumeBtn!);

      await waitFor(() => {
        expect(mockResume).toHaveBeenCalledTimes(1);
      });
    });

    it('calls stopTraining when Stop is clicked on running state', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('47 / 100');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const buttons = screen.getAllByRole('button');
      const stopBtn = buttons.find((b) => b.textContent?.includes('Stop'));
      expect(stopBtn).not.toBeDisabled();
      await userEvent.click(stopBtn!);

      await waitFor(() => {
        expect(mockStop).toHaveBeenCalledTimes(1);
      });
    });

    it('calls saveCheckpoint when Checkpoint is clicked on running state', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('47 / 100');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const buttons = screen.getAllByRole('button');
      const cpBtn = buttons.find((b) => b.textContent?.includes('Checkpoint'));
      expect(cpBtn).not.toBeDisabled();
      await userEvent.click(cpBtn!);

      await waitFor(() => {
        expect(mockCheckpoint).toHaveBeenCalledTimes(1);
      });
    });

    it('shows notification on successful start', async () => {
      mockFetchStatus.mockResolvedValue(idleStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('Idle');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const buttons = screen.getAllByRole('button');
      const startBtn = buttons.find((b) => b.textContent?.includes('Start'));
      await userEvent.click(startBtn!);

      await waitFor(() => {
        expect(screen.getByText('Training started successfully')).toBeInTheDocument();
      });
    });

    it('shows notification on mutation error', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      mockPause.mockRejectedValue(new Error('Cannot pause while not running'));
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('47 / 100');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const buttons = screen.getAllByRole('button');
      const pauseBtn = buttons.find((b) => b.textContent?.includes('Pause'));
      await userEvent.click(pauseBtn!);

      await waitFor(() => {
        expect(screen.getByText('Cannot pause while not running')).toBeInTheDocument();
      });
    });
  });

  describe('Configuration Panel', () => {
    it('toggles configuration panel when Config button is clicked', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('47 / 100');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.queryByText('Training Configuration')).not.toBeInTheDocument();

      const buttons = screen.getAllByRole('button');
      const configBtn = buttons.find((b) => {
        const t = b.textContent?.trim() || '';
        return (t === 'Config' || t === 'Hide Config') ||
          (t.includes('Config') && !t.includes('Reload') && !t.includes('Save'));
      });
      expect(configBtn).toBeTruthy();
      await userEvent.click(configBtn!);

      await waitFor(() => {
        expect(screen.getByText('Training Configuration')).toBeInTheDocument();
      });
    });

    it('loads and displays configuration data', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('47 / 100');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const buttons = screen.getAllByRole('button');
      const configBtn = buttons.find((b) => {
        const t = b.textContent?.trim() || '';
        return (t === 'Config' || t === 'Hide Config') ||
          (t.includes('Config') && !t.includes('Reload') && !t.includes('Save'));
      });
      expect(configBtn).toBeTruthy();
      await userEvent.click(configBtn!);

      await waitFor(() => {
        expect(mockFetchConfig).toHaveBeenCalled();
      });

      await waitFor(() => {
        const datasetInput = document.querySelector<HTMLInputElement>('input[value="cifar10"]');
        expect(datasetInput).toBeInTheDocument();
      });

      const lrInput = document.querySelector<HTMLInputElement>('input[value="0.001"]');
      expect(lrInput).toBeInTheDocument();
    });

    it('saves configuration when Save is clicked', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('47 / 100');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const buttons = screen.getAllByRole('button');
      const configBtn = buttons.find((b) => {
        const t = b.textContent?.trim() || '';
        return (t === 'Config' || t === 'Hide Config') ||
          (t.includes('Config') && !t.includes('Reload') && !t.includes('Save'));
      });
      expect(configBtn).toBeTruthy();
      await userEvent.click(configBtn!);

      await waitFor(() => {
        expect(screen.getByText('Training Configuration')).toBeInTheDocument();
      });

      await waitFor(() => {
        const datasetInput = document.querySelector<HTMLInputElement>('input[value="cifar10"]');
        expect(datasetInput).toBeInTheDocument();
      });

      await waitFor(() => {
        const currentButtons = screen.getAllByRole('button');
        const saveBtn = currentButtons.find((b) => b.textContent?.includes('Save Configuration'));
        expect(saveBtn).toBeTruthy();
      });

      const currentButtons = screen.getAllByRole('button');
      const saveBtn = currentButtons.find((b) => b.textContent?.includes('Save Configuration'));
      await userEvent.click(saveBtn!);

      await waitFor(() => {
        expect(mockUpdateConfig).toHaveBeenCalledTimes(1);
      });
    });

    it('validates learning rate before saving', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('47 / 100');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const buttons = screen.getAllByRole('button');
      const configBtn = buttons.find((b) => {
        const t = b.textContent?.trim() || '';
        return (t === 'Config' || t === 'Hide Config') ||
          (t.includes('Config') && !t.includes('Reload') && !t.includes('Save'));
      });
      expect(configBtn).toBeTruthy();
      await userEvent.click(configBtn!);

      await waitFor(() => {
        expect(screen.getByText('Training Configuration')).toBeInTheDocument();
      });

      await waitFor(() => {
        const datasetInput = document.querySelector<HTMLInputElement>('input[value="cifar10"]');
        expect(datasetInput).toBeInTheDocument();
      });

      const numberInputs = document.querySelectorAll<HTMLInputElement>('input[type="number"]');
      const lrInput = Array.from(numberInputs).find(
        (el) => parseFloat(el.value) >= 0.0001 && parseFloat(el.value) < 1,
      );
      expect(lrInput).toBeTruthy();
      if (lrInput) {
        fireEvent.change(lrInput, { target: { value: '0' } });
      }

      const currentButtons = screen.getAllByRole('button');
      const saveBtn = currentButtons.find((b) => b.textContent?.includes('Save Configuration'));
      expect(saveBtn).toBeTruthy();
      await userEvent.click(saveBtn!);

      await waitFor(() => {
        expect(
          screen.getByText('Learning rate must be greater than 0'),
        ).toBeInTheDocument();
      });

      expect(mockUpdateConfig).not.toHaveBeenCalled();
    });
  });

  describe('Convergence Data Accumulation', () => {
    it('initializes convergence data from status on mount', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      renderTraining();

      await waitFor(() => {
        expect(screen.getByText('R-47')).toBeInTheDocument();
      });
    });

    it('accumulates new data points when round changes', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      renderTraining();

      await waitFor(() => {
        expect(screen.getByText('R-47')).toBeInTheDocument();
      });

      const rows47 = screen.getAllByText('R-47');
      expect(rows47.length).toBeGreaterThanOrEqual(1);

      const updatedStatus: TrainingStatusSummary = {
        ...runningStatusData,
        data: {
          ...runningStatusData.data,
          current_round: 48,
          current_accuracy: 0.88,
          current_loss: 0.2,
        },
      };
      mockFetchStatus.mockResolvedValue(updatedStatus);

      const refreshButton = screen.getByText('Refresh');
      await userEvent.click(refreshButton);

      await waitFor(() => {
        const elems = screen.getAllByText('48 / 100');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText('R-48')).toBeInTheDocument();
    });
  });

  describe('Round Execution History', () => {
    it('displays empty state when no rounds exist', async () => {
      mockFetchStatus.mockResolvedValue(idleStatusData);
      renderTraining();

      await waitFor(() => {
        const elems = screen.getAllByText('Idle');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText(/No round history yet/)).toBeInTheDocument();
    });

    it('shows round history table with data after accumulation', async () => {
      mockFetchStatus.mockResolvedValue(runningStatusData);
      renderTraining();

      await waitFor(() => {
        expect(screen.getByText('R-47')).toBeInTheDocument();
      });

      expect(screen.getByText('Round Execution History')).toBeInTheDocument();
    });
  });
});
