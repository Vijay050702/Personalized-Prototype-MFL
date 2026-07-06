import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

import { Datasets } from '../../pages/Datasets';
import * as datasetsApi from '../../api/datasets';
import type {
  DatasetListResponse,
  DatasetDetailResponse,
  OperationResponse,
  PartitionResponse,
} from '../../types';

vi.mock('../../api/datasets');

const mockFetchDatasets = datasetsApi.fetchDatasets as ReturnType<typeof vi.fn>;
const mockFetchDatasetDetail = datasetsApi.fetchDatasetDetail as ReturnType<typeof vi.fn>;
const mockRegisterDataset = datasetsApi.registerDataset as ReturnType<typeof vi.fn>;
const mockDownloadDataset = datasetsApi.downloadDataset as ReturnType<typeof vi.fn>;
const mockPreprocessDataset = datasetsApi.preprocessDataset as ReturnType<typeof vi.fn>;
const mockPartitionDataset = datasetsApi.partitionDataset as ReturnType<typeof vi.fn>;
const mockSimulateMissingModality = datasetsApi.simulateMissingModality as ReturnType<typeof vi.fn>;
const mockDeleteDataset = datasetsApi.deleteDataset as ReturnType<typeof vi.fn>;

const sampleDatasets: DatasetListResponse = {
  status: 'success',
  message: 'Datasets retrieved',
  total: 4,
  data: [
    { id: 'd1', name: 'CIFAR-10', type: 'classification', modality: 'image', size_mb: 168.0, samples: 60000, classes: 10, client_id: 'client-1', distribution: 'iid' },
    { id: 'd2', name: 'IMDB-Reviews', type: 'sentiment', modality: 'text', size_mb: 85.3, samples: 50000, classes: 2, client_id: 'client-2', distribution: 'dirichlet' },
    { id: 'd3', name: 'Sensor-Data', type: 'regression', modality: 'tabular', size_mb: 1200.0, samples: 1000000, classes: 0, client_id: 'client-3', distribution: 'shard' },
    { id: 'd4', name: 'AudioMNIST', type: 'classification', modality: 'audio', size_mb: 45.2, samples: 30000, classes: 10, client_id: 'client-1', distribution: 'iid' },
  ],
};

const sampleDetail: DatasetDetailResponse = {
  status: 'success',
  message: 'Dataset detail retrieved',
  data: {
    dataset_name: 'CIFAR-10',
    modalities: ['image'],
    classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    num_classes: 10,
    input_shapes: { image: [3, 32, 32] },
    num_samples: 60000,
    client_count: 3,
    missing_modality_ratio: 0.0,
    download_status: 'completed',
    preprocessing_status: 'completed',
    partition_status: 'not_partitioned',
  },
};

function renderDatasets() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, retryDelay: 0 },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Datasets />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const user = userEvent.setup();

describe('Datasets page', () => {
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
    mockFetchDatasets.mockReturnValue(new Promise(() => {}));
    renderDatasets();
    expect(screen.getByText('Dataset Manager')).toBeInTheDocument();
    const skeletons = document.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBeGreaterThanOrEqual(5);
  });

  /* ------------------------------------------------------------------ */
  /*  Error state                                                        */
  /* ------------------------------------------------------------------ */
  it('shows error state on fetch failure', async () => {
    mockFetchDatasets.mockRejectedValue(new Error('Server error. Please try again later.'));
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('Failed to load datasets')).toBeInTheDocument();
    });
    expect(screen.getByText('Server error. Please try again later.')).toBeInTheDocument();
    expect(screen.getByText('Try Again')).toBeInTheDocument();
  });

  it('retries on error when clicking Try Again', async () => {
    mockFetchDatasets.mockRejectedValue(new Error('Server error'));
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('Failed to load datasets')).toBeInTheDocument();
    });

    mockFetchDatasets.mockResolvedValue(sampleDatasets);

    await user.click(screen.getByText('Try Again'));

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });
  });

  /* ------------------------------------------------------------------ */
  /*  Empty state                                                        */
  /* ------------------------------------------------------------------ */
  it('shows empty state when no datasets exist', async () => {
    mockFetchDatasets.mockResolvedValue({
      status: 'success',
      message: 'No datasets',
      total: 0,
      data: [],
    });
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('No datasets found')).toBeInTheDocument();
    });
    const registerBtns = screen.getAllByText('Register Dataset');
    expect(registerBtns.length).toBeGreaterThanOrEqual(1);
  });

  /* ------------------------------------------------------------------ */
  /*  Table view with data                                               */
  /* ------------------------------------------------------------------ */
  it('renders dataset table with data on success', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    expect(screen.getByText('IMDB-Reviews')).toBeInTheDocument();
    expect(screen.getByText('Sensor-Data')).toBeInTheDocument();
    expect(screen.getByText('AudioMNIST')).toBeInTheDocument();
    expect(screen.getByText('168.0 MB')).toBeInTheDocument();
    expect(screen.getByText('1.2 GB')).toBeInTheDocument();
    expect(screen.getByText('60.0K')).toBeInTheDocument();
    expect(screen.getByText('1.0M')).toBeInTheDocument();
  });

  it('shows 4 datasets count in filter bar', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('4 datasets')).toBeInTheDocument();
    });
  });

  /* ------------------------------------------------------------------ */
  /*  Grid view                                                          */
  /* ------------------------------------------------------------------ */
  it('switches to grid view', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    const gridBtn = screen.getByTitle('Grid view');
    await user.click(gridBtn);

    expect(screen.getAllByText('10 classes').length).toBeGreaterThanOrEqual(1);
  });

  it('switches back to table view', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    await user.click(screen.getByTitle('Grid view'));

    await waitFor(() => {
      expect(screen.getAllByText('10 classes').length).toBeGreaterThanOrEqual(1);
    });

    await user.click(screen.getByTitle('Table view'));
  });

  /* ------------------------------------------------------------------ */
  /*  Search filtering                                                   */
  /* ------------------------------------------------------------------ */
  it('filters datasets by search query', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText('Search datasets...');
    await user.type(searchInput, 'audio');

    await waitFor(() => {
      expect(screen.queryByText('CIFAR-10')).not.toBeInTheDocument();
    });
    expect(screen.getByText('AudioMNIST')).toBeInTheDocument();
    expect(screen.getByText('1 dataset')).toBeInTheDocument();
  });

  it('shows empty state when search yields no results', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText('Search datasets...');
    await user.type(searchInput, 'zzzzz');

    await waitFor(() => {
      expect(screen.queryByText('CIFAR-10')).not.toBeInTheDocument();
    });
    expect(screen.getByText('No datasets found')).toBeInTheDocument();
  });

  /* ------------------------------------------------------------------ */
  /*  Type filter                                                        */
  /* ------------------------------------------------------------------ */
  it('filters datasets by type', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    const typeSelect = screen.getAllByRole('combobox')[0];
    await user.selectOptions(typeSelect, 'sentiment');

    await waitFor(() => {
      expect(screen.queryByText('CIFAR-10')).not.toBeInTheDocument();
    });
    expect(screen.getByText('IMDB-Reviews')).toBeInTheDocument();
  });

  /* ------------------------------------------------------------------ */
  /*  Sorting                                                            */
  /* ------------------------------------------------------------------ */
  it('sorts by name column click', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    const nameHeader = screen.getByText('Name');
    await user.click(nameHeader);
  });

  it('toggles sort direction on second click', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    const nameHeader = screen.getByText('Name');
    await user.click(nameHeader);
    await user.click(nameHeader);
  });

  it('sorts by different columns', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Samples'));
    await user.click(screen.getByText('Size'));
    await user.click(screen.getByText('Classes'));
  });

  /* ------------------------------------------------------------------ */
  /*  Pagination                                                         */
  /* ------------------------------------------------------------------ */
  it('paginates when datasets exceed page size', async () => {
    const manyDatasets: DatasetListResponse = {
      status: 'success',
      message: 'ok',
      total: 15,
      data: Array.from({ length: 15 }, (_, i) => ({
        id: `d${i}`,
        name: `Z-Dataset-${String(i + 1).padStart(2, '0')}`,
        type: 'test',
        modality: 'image',
        size_mb: 10,
        samples: 1000,
        classes: 5,
        client_id: 'client-1',
        distribution: 'iid',
      })),
    };
    mockFetchDatasets.mockResolvedValue(manyDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('Z-Dataset-01')).toBeInTheDocument();
    });

    expect(screen.queryByText('Z-Dataset-11')).not.toBeInTheDocument();

    const paginationSection = screen.getByText(/of 15/).closest('div')!;
    const nextBtn = paginationSection.querySelector('button:last-child')!;
    await user.click(nextBtn);

    await waitFor(() => {
      expect(screen.getByText('Z-Dataset-11')).toBeInTheDocument();
    });
  });

  it('changes page size', async () => {
    const manyDatasets: DatasetListResponse = {
      status: 'success',
      message: 'ok',
      total: 12,
      data: Array.from({ length: 12 }, (_, i) => ({
        id: `d${i}`,
        name: `Dataset-${i + 1}`,
        type: 'test',
        modality: 'image',
        size_mb: 10,
        samples: 1000,
        classes: 5,
        client_id: 'client-1',
        distribution: 'iid',
      })),
    };
    mockFetchDatasets.mockResolvedValue(manyDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('Dataset-11')).toBeInTheDocument();
    });

    const sizeSelect = screen.getAllByRole('combobox')[1];
    await user.selectOptions(sizeSelect, '20');
  });

  /* ------------------------------------------------------------------ */
  /*  Register modal                                                     */
  /* ------------------------------------------------------------------ */
  it('opens register modal from header button', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('Dataset Manager')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Register'));

    expect(screen.getByRole('heading', { name: 'Register Dataset' })).toBeInTheDocument();
    expect(screen.getByPlaceholderText('e.g. CIFAR-10')).toBeInTheDocument();
  });

  it('opens register modal from empty state', async () => {
    mockFetchDatasets.mockResolvedValue({
      status: 'success',
      message: 'No datasets',
      total: 0,
      data: [],
    });
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('No datasets found')).toBeInTheDocument();
    });

    const btns = screen.getAllByText('Register Dataset');
    await user.click(btns[0]);

    expect(screen.getByRole('heading', { name: 'Register Dataset' })).toBeInTheDocument();
  });

  it('closes register modal on cancel', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('Dataset Manager')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Register'));
    expect(screen.getByRole('heading', { name: 'Register Dataset' })).toBeInTheDocument();

    await user.click(screen.getByText('Cancel'));
    expect(screen.queryByRole('heading', { name: 'Register Dataset' })).not.toBeInTheDocument();
  });

  it('submits register modal', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockRegisterDataset.mockResolvedValue({
      status: 'success',
      message: 'Dataset registered',
      dataset_name: 'New-Dataset',
      operation: 'register',
    } as OperationResponse);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('Dataset Manager')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Register'));

    await user.type(screen.getByPlaceholderText('e.g. CIFAR-10'), 'New-Dataset');

    const allRegister = screen.getAllByText('Register');
    const submitRegister = allRegister[allRegister.length - 1];
    await user.click(submitRegister);

    await waitFor(() => {
      expect(mockRegisterDataset).toHaveBeenCalled();
      const call = mockRegisterDataset.mock.calls[0][0];
      expect(call.name).toBe('New-Dataset');
      expect(call.modality).toBe('image');
      expect(call.path).toBeUndefined();
    });
  });

  it('shows validation error when name is empty in register modal', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('Dataset Manager')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Register'));

    const allRegister = screen.getAllByText('Register');
    const submitRegister = allRegister[allRegister.length - 1];
    expect(submitRegister).toBeDisabled();
  });

  /* ------------------------------------------------------------------ */
  /*  Detail panel                                                       */
  /* ------------------------------------------------------------------ */
  it('opens detail panel on row click', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockFetchDatasetDetail.mockResolvedValue(sampleDetail);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    await user.click(screen.getByText('CIFAR-10'));

    await waitFor(() => {
      expect(screen.getByText('Processing Status')).toBeInTheDocument();
    });
    const completedBadges = screen.getAllByText('completed');
    expect(completedBadges.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('not_partitioned')).toBeInTheDocument();
  });

  it('closes detail panel', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockFetchDatasetDetail.mockResolvedValue(sampleDetail);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    await user.click(screen.getByText('CIFAR-10'));

    await waitFor(() => {
      expect(screen.getByText('Processing Status')).toBeInTheDocument();
    });

    const closeBtns = screen.getAllByRole('button', { name: '' });
    const xButtons = document.querySelectorAll('button');
    for (const btn of xButtons) {
      if (btn.querySelector('.lucide-x') && btn.closest('.fixed')) {
        await user.click(btn);
        break;
      }
    }

    await waitFor(() => {
      expect(screen.queryByText('Processing Status')).not.toBeInTheDocument();
    });
  });

  it('opens detail via eye icon in table', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockFetchDatasetDetail.mockResolvedValue(sampleDetail);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    const eyeBtns = screen.getAllByTitle('View details');
    await user.click(eyeBtns[0]);

    await waitFor(() => {
      expect(screen.getByText('Processing Status')).toBeInTheDocument();
    });
  });

  /* ------------------------------------------------------------------ */
  /*  Download action                                                    */
  /* ------------------------------------------------------------------ */
  it('triggers download from detail panel', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockFetchDatasetDetail.mockResolvedValue(sampleDetail);
    mockDownloadDataset.mockResolvedValue({
      status: 'success',
      message: 'Download started',
      dataset_name: 'CIFAR-10',
      operation: 'download',
    } as OperationResponse);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    await user.click(screen.getByText('CIFAR-10'));

    await waitFor(() => {
      expect(screen.getByText('Processing Status')).toBeInTheDocument();
    });

    const downloadBtns = screen.getAllByText('Download');
    const downloadBtn = downloadBtns[downloadBtns.length - 1];
    await user.click(downloadBtn);

    await waitFor(() => {
      expect(mockDownloadDataset).toHaveBeenCalled();
      expect(mockDownloadDataset.mock.calls[0][0]).toEqual({ dataset_name: 'CIFAR-10' });
    });
  });

  it('triggers download directly from table icon', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockDownloadDataset.mockResolvedValue({
      status: 'success',
      message: 'Download started',
      dataset_name: 'AudioMNIST',
      operation: 'download',
    } as OperationResponse);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    const downloadBtns = screen.getAllByTitle('Download');
    await user.click(downloadBtns[0]);

    await waitFor(() => {
      expect(mockDownloadDataset).toHaveBeenCalled();
      expect(mockDownloadDataset.mock.calls[0][0]).toEqual({ dataset_name: 'AudioMNIST' });
    });
  });

  /* ------------------------------------------------------------------ */
  /*  Preprocess action                                                  */
  /* ------------------------------------------------------------------ */
  it('triggers preprocess from detail panel', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockFetchDatasetDetail.mockResolvedValue(sampleDetail);
    mockPreprocessDataset.mockResolvedValue({
      status: 'success',
      message: 'Preprocessing started',
      dataset_name: 'CIFAR-10',
      operation: 'preprocess',
    } as OperationResponse);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    await user.click(screen.getByText('CIFAR-10'));

    await waitFor(() => {
      expect(screen.getByText('Processing Status')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Preprocess'));

    await waitFor(() => {
      expect(mockPreprocessDataset).toHaveBeenCalled();
      expect(mockPreprocessDataset.mock.calls[0][0]).toEqual({ dataset_name: 'CIFAR-10' });
    });
  });

  /* ------------------------------------------------------------------ */
  /*  Partition modal                                                    */
  /* ------------------------------------------------------------------ */
  it('opens partition modal from detail panel', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockFetchDatasetDetail.mockResolvedValue(sampleDetail);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    await user.click(screen.getByText('CIFAR-10'));

    await waitFor(() => {
      expect(screen.getByText('Processing Status')).toBeInTheDocument();
    });

    const partitionBtns = screen.getAllByText('Partition');
    await user.click(partitionBtns[partitionBtns.length - 1]);

    await waitFor(() => {
      expect(screen.getByText('Partition Dataset')).toBeInTheDocument();
    });
  });

  it('submits partition with IID strategy', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockFetchDatasetDetail.mockResolvedValue(sampleDetail);
    mockPartitionDataset.mockResolvedValue({
      status: 'success',
      dataset_name: 'CIFAR-10',
      strategy: 'iid',
      num_clients: 10,
      client_distributions: [],
      seed: 42,
    } as unknown as PartitionResponse);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    await user.click(screen.getByText('CIFAR-10'));

    await waitFor(() => {
      expect(screen.getByText('Processing Status')).toBeInTheDocument();
    });

    const partitionBtns = screen.getAllByText('Partition');
    await user.click(partitionBtns[partitionBtns.length - 1]);

    await waitFor(() => {
      expect(screen.getByText('Partition Dataset')).toBeInTheDocument();
    });

    const submitBtns = screen.getAllByText('Partition');
    await user.click(submitBtns[submitBtns.length - 1]);

    await waitFor(() => {
      expect(mockPartitionDataset).toHaveBeenCalled();
      const call = mockPartitionDataset.mock.calls[0][0];
      expect(call.dataset_name).toBe('CIFAR-10');
      expect(call.strategy).toBe('iid');
      expect(call.num_clients).toBe(10);
      expect(call.seed).toBe(42);
      expect(call.balanced).toBe(true);
      expect(call.shards_per_client).toBeUndefined();
    });
  });

  it('submits partition with Dirichlet strategy', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockFetchDatasetDetail.mockResolvedValue(sampleDetail);
    mockPartitionDataset.mockResolvedValue({
      status: 'success',
      dataset_name: 'CIFAR-10',
      strategy: 'dirichlet',
      num_clients: 5,
      client_distributions: [],
      seed: 42,
    } as unknown as PartitionResponse);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    await user.click(screen.getByText('CIFAR-10'));

    await waitFor(() => {
      expect(screen.getByText('Processing Status')).toBeInTheDocument();
    });

    const partitionBtns = screen.getAllByText('Partition');
    await user.click(partitionBtns[partitionBtns.length - 1]);

    await waitFor(() => {
      expect(screen.getByText('Partition Dataset')).toBeInTheDocument();
    });

    const allSelects = screen.getAllByRole('combobox');
    const strategySelect = allSelects[allSelects.length - 1] as HTMLSelectElement;
    fireEvent.change(strategySelect, { target: { value: 'dirichlet' } });

    await waitFor(() => {
      expect(strategySelect).toHaveValue('dirichlet');
    });

    const forms = document.querySelectorAll('form');
    const partitionForm = forms[forms.length - 1];
    fireEvent.submit(partitionForm);

    await waitFor(() => {
      expect(mockPartitionDataset).toHaveBeenCalled();
      const call = mockPartitionDataset.mock.calls[0][0];
      expect(call.strategy).toBe('dirichlet');
      expect(call.alpha).toBe(0.5);
    });
  });

  it('opens partition modal from table icon', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    const partitionIcons = screen.getAllByTitle('Partition');
    await user.click(partitionIcons[0]);

    await waitFor(() => {
      expect(screen.getByText('Partition Dataset')).toBeInTheDocument();
    });
  });

  /* ------------------------------------------------------------------ */
  /*  Missing modality modal                                             */
  /* ------------------------------------------------------------------ */
  it('opens missing modality modal from detail panel', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockFetchDatasetDetail.mockResolvedValue(sampleDetail);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    await user.click(screen.getByText('CIFAR-10'));

    await waitFor(() => {
      expect(screen.getByText('Processing Status')).toBeInTheDocument();
    });

    const mmBtns = screen.getAllByText('Missing Modality');
    await user.click(mmBtns[mmBtns.length - 1]);

    await waitFor(() => {
      expect(screen.getByText('Simulate Missing Modality')).toBeInTheDocument();
    });
  });

  it('submits missing modality simulation', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockFetchDatasetDetail.mockResolvedValue(sampleDetail);
    mockSimulateMissingModality.mockResolvedValue({
      status: 'success',
      message: 'Missing modality simulated',
      dataset_name: 'CIFAR-10',
      operation: 'missing_modality',
    } as OperationResponse);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    await user.click(screen.getByText('CIFAR-10'));

    await waitFor(() => {
      expect(screen.getByText('Processing Status')).toBeInTheDocument();
    });

    const mmBtns = screen.getAllByText('Missing Modality');
    await user.click(mmBtns[mmBtns.length - 1]);

    await waitFor(() => {
      expect(screen.getByText('Simulate Missing Modality')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Simulate'));

    await waitFor(() => {
      expect(mockSimulateMissingModality).toHaveBeenCalled();
      const call = mockSimulateMissingModality.mock.calls[0][0];
      expect(call.dataset_name).toBe('CIFAR-10');
      expect(call.strategy).toBe('random');
      expect(call.missing_ratio).toBe(0.3);
      expect(call.seed).toBe(42);
    });
  });

  it('selects different missing modality strategy', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockFetchDatasetDetail.mockResolvedValue(sampleDetail);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    await user.click(screen.getByText('CIFAR-10'));

    await waitFor(() => {
      expect(screen.getByText('Processing Status')).toBeInTheDocument();
    });

    const mmBtns = screen.getAllByText('Missing Modality');
    await user.click(mmBtns[mmBtns.length - 1]);

    await waitFor(() => {
      expect(screen.getByText('Simulate Missing Modality')).toBeInTheDocument();
    });

    const strategyLabels = screen.getAllByText('Strategy');
    const strategyLabel = strategyLabels[strategyLabels.length - 1];
    const select = strategyLabel.parentElement?.querySelector('select')!;
    await user.selectOptions(select, 'client_wise');
  });

  /* ------------------------------------------------------------------ */
  /*  Delete confirmation                                                */
  /* ------------------------------------------------------------------ */
  it('opens delete confirmation from detail panel', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockFetchDatasetDetail.mockResolvedValue(sampleDetail);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    await user.click(screen.getByText('CIFAR-10'));

    await waitFor(() => {
      expect(screen.getByText('Processing Status')).toBeInTheDocument();
    });

    const deleteBtn = screen.getAllByText('Delete Dataset')[0];
    await user.click(deleteBtn);

    await waitFor(() => {
      const deleteTexts = screen.getAllByText('Delete Dataset');
      expect(deleteTexts.length).toBeGreaterThanOrEqual(1);
    });
    expect(screen.getByText(/are you sure/i)).toBeInTheDocument();
  });

  it('opens delete confirmation from table icon', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    const deleteBtns = screen.getAllByTitle('Delete');
    await user.click(deleteBtns[0]);

    await waitFor(() => {
      expect(screen.getByText('Delete Dataset')).toBeInTheDocument();
    });
  });

  it('cancels delete confirmation', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    const deleteBtns = screen.getAllByTitle('Delete');
    await user.click(deleteBtns[0]);

    await waitFor(() => {
      expect(screen.getByText('Delete Dataset')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Cancel'));

    expect(screen.queryByText(/are you sure/i)).not.toBeInTheDocument();
  });

  it('confirms deletion', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockDeleteDataset.mockResolvedValue({
      status: 'success',
      message: 'Dataset deleted',
      dataset_name: 'CIFAR-10',
      operation: 'delete',
    } as OperationResponse);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    const deleteIcons = screen.getAllByTitle('Delete');
    await user.click(deleteIcons[0]);

    await waitFor(() => {
      expect(screen.getByText('Delete Dataset')).toBeInTheDocument();
    });

    const allDelete = screen.getAllByText('Delete');
    const confirmDelete = allDelete[allDelete.length - 1];
    await user.click(confirmDelete);

    await waitFor(() => {
      expect(mockDeleteDataset).toHaveBeenCalled();
      expect(mockDeleteDataset.mock.calls[0][0]).toBe('AudioMNIST');
    });
  });

  /* ------------------------------------------------------------------ */
  /*  Toast notifications                                                */
  /* ------------------------------------------------------------------ */
  it('shows success toast after register', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockRegisterDataset.mockResolvedValue({
      status: 'success',
      message: 'Dataset New-Dataset registered',
      dataset_name: 'New-Dataset',
      operation: 'register',
    } as OperationResponse);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('Dataset Manager')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Register'));
    await user.type(screen.getByPlaceholderText('e.g. CIFAR-10'), 'New-Dataset');
    const allRegister = screen.getAllByText('Register');
    await user.click(allRegister[allRegister.length - 1]);

    await waitFor(() => {
      expect(screen.getByText('Dataset New-Dataset registered')).toBeInTheDocument();
    });
  });

  it('shows error toast on mutation failure', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockRegisterDataset.mockRejectedValue(new Error('Registration failed'));
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('Dataset Manager')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Register'));
    await user.type(screen.getByPlaceholderText('e.g. CIFAR-10'), 'Bad-Name');
    const allRegister = screen.getAllByText('Register');
    await user.click(allRegister[allRegister.length - 1]);

    await waitFor(() => {
      expect(screen.getByText('Registration failed')).toBeInTheDocument();
    });
  });

  it('shows partition success toast', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    mockFetchDatasetDetail.mockResolvedValue(sampleDetail);
    mockPartitionDataset.mockResolvedValue({
      status: 'success',
      dataset_name: 'CIFAR-10',
      strategy: 'iid',
      num_clients: 10,
      client_distributions: [],
      seed: 42,
    } as unknown as PartitionResponse);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('CIFAR-10')).toBeInTheDocument();
    });

    await user.click(screen.getByText('CIFAR-10'));

    await waitFor(() => {
      expect(screen.getByText('Processing Status')).toBeInTheDocument();
    });

    const partitionBtns = screen.getAllByText('Partition');
    await user.click(partitionBtns[partitionBtns.length - 1]);

    await waitFor(() => {
      expect(screen.getByText('Partition Dataset')).toBeInTheDocument();
    });

    const submitBtns = screen.getAllByText('Partition');
    await user.click(submitBtns[submitBtns.length - 1]);

    await waitFor(() => {
      expect(screen.getByText(/Partitioned/)).toBeInTheDocument();
    });
  });

  /* ------------------------------------------------------------------ */
  /*  Edge cases                                                         */
  /* ------------------------------------------------------------------ */
  it('handles generic error message', async () => {
    mockFetchDatasets.mockRejectedValue('string error');
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('Failed to load datasets')).toBeInTheDocument();
    });
    expect(screen.getByText('An unexpected error occurred.')).toBeInTheDocument();
  });

  it('calls fetchDatasets on mount', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(mockFetchDatasets).toHaveBeenCalledTimes(1);
    });
  });

  it('renders register modal with modality select', async () => {
    mockFetchDatasets.mockResolvedValue(sampleDatasets);
    renderDatasets();

    await waitFor(() => {
      expect(screen.getByText('Dataset Manager')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Register'));

    expect(screen.getByText('Image')).toBeInTheDocument();
    expect(screen.getByText('Text')).toBeInTheDocument();
  });
});
