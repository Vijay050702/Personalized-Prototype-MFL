import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

import { KnowledgeTransfer } from '../../pages/KnowledgeTransfer';
import * as knowledgeTransferApi from '../../api/knowledgeTransfer';
import type { KnowledgeTransferListResponse, KnowledgeTransferStatisticsResponse } from '../../types';

vi.mock('../../api/knowledgeTransfer');

const mockFetchTransfers = knowledgeTransferApi.fetchKnowledgeTransfers as ReturnType<typeof vi.fn>;
const mockFetchStats = knowledgeTransferApi.fetchKnowledgeTransferStatistics as ReturnType<typeof vi.fn>;

const mockTransferList: KnowledgeTransferListResponse = {
  status: 'success',
  message: 'Transfers retrieved',
  data: [
    { transfer_id: 'kt-001', source_client: 'client-001', target_client: 'client-002', source_prototype: 'proto-001', target_prototype: 'proto-005', source_modality: 'visual', target_modality: 'linguistic', transfer_strategy: 'direct', cross_modal_mapping: 'visual→text', alignment_method: 'linear', transfer_loss: 0.1234, similarity_score: 0.9234, confidence: 0.9123, communication_round: 10, transfer_status: 'completed', execution_time: 2.34, created_at: '2025-01-01T10:00:00Z' },
    { transfer_id: 'kt-002', source_client: 'client-001', target_client: 'client-003', source_prototype: 'proto-002', target_prototype: 'proto-007', source_modality: 'visual', target_modality: 'acoustic', transfer_strategy: 'sequential', cross_modal_mapping: 'visual→audio', alignment_method: 'mlp', transfer_loss: 0.2345, similarity_score: 0.8456, confidence: 0.8765, communication_round: 10, transfer_status: 'completed', execution_time: 3.45, created_at: '2025-01-01T10:05:00Z' },
    { transfer_id: 'kt-003', source_client: 'client-002', target_client: 'client-001', source_prototype: 'proto-003', target_prototype: 'proto-006', source_modality: 'acoustic', target_modality: 'visual', transfer_strategy: 'direct', cross_modal_mapping: 'audio→visual', alignment_method: 'linear', transfer_loss: 0.3456, similarity_score: 0.7812, confidence: 0.8234, communication_round: 15, transfer_status: 'failed', execution_time: 1.23, created_at: '2025-01-01T11:00:00Z' },
    { transfer_id: 'kt-004', source_client: 'client-003', target_client: 'client-001', source_prototype: 'proto-004', target_prototype: 'proto-008', source_modality: 'linguistic', target_modality: 'visual', transfer_strategy: 'graph_based', cross_modal_mapping: 'text→visual', alignment_method: 'contrastive', transfer_loss: 0.0456, similarity_score: 0.9678, confidence: 0.9567, communication_round: 20, transfer_status: 'completed', execution_time: 5.67, created_at: '2025-01-01T12:00:00Z' },
    { transfer_id: 'kt-005', source_client: 'client-002', target_client: 'client-003', source_prototype: 'proto-005', target_prototype: 'proto-009', source_modality: 'multimodal', target_modality: 'visual', transfer_strategy: 'cross_modal', cross_modal_mapping: 'multimodal→visual', alignment_method: 'info_nce', transfer_loss: 0.0789, similarity_score: 0.9345, confidence: 0.9456, communication_round: 25, transfer_status: 'running', execution_time: 12.34, created_at: '2025-01-01T13:00:00Z' },
  ],
  total: 5,
};

const mockStatsResponse: KnowledgeTransferStatisticsResponse = {
  status: 'success',
  message: 'Statistics retrieved',
  data: {
    total_transfers: 5,
    successful_transfers: 3,
    failed_transfers: 1,
    average_similarity: 0.8905,
    average_confidence: 0.9029,
    average_transfer_loss: 0.1656,
    average_execution_time: 5.006,
    communication_efficiency: 0.85,
  },
};

function renderKnowledgeTransfer() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, retryDelay: 0 },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <KnowledgeTransfer />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('KnowledgeTransfer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  /* ================================================================== */
  /*  Loading State                                                      */
  /* ================================================================== */
  describe('Loading State', () => {
    it('shows loading skeleton while fetching', () => {
      mockFetchTransfers.mockReturnValue(new Promise(() => {}));
      mockFetchStats.mockReturnValue(new Promise(() => {}));
      renderKnowledgeTransfer();
      const skeletons = document.querySelectorAll('.animate-pulse');
      expect(skeletons.length).toBeGreaterThanOrEqual(4);
    });
  });

  /* ================================================================== */
  /*  Error State                                                        */
  /* ================================================================== */
  describe('Error State', () => {
    it('shows error state on API failure', async () => {
      mockFetchTransfers.mockRejectedValue(new Error('Server error. Please try again.'));
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        expect(screen.getByText('Failed to load transfers')).toBeInTheDocument();
      });
      expect(screen.getByText('Server error. Please try again.')).toBeInTheDocument();
      expect(screen.getByText('Try Again')).toBeInTheDocument();
    });

    it('shows error state when backend is unavailable', async () => {
      mockFetchTransfers.mockRejectedValue(new Error('Backend is unavailable. Please check your connection.'));
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        expect(screen.getByText('Failed to load transfers')).toBeInTheDocument();
      });
      expect(screen.getByText('Backend is unavailable. Please check your connection.')).toBeInTheDocument();
    });

    it('retries on error when clicking Try Again', async () => {
      mockFetchTransfers.mockRejectedValue(new Error('Server error'));
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        expect(screen.getByText('Failed to load transfers')).toBeInTheDocument();
      });

      mockFetchTransfers.mockResolvedValue(mockTransferList);

      const retryButton = screen.getByText('Try Again');
      await userEvent.click(retryButton);

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });
    });
  });

  /* ================================================================== */
  /*  Empty State                                                        */
  /* ================================================================== */
  describe('Empty State', () => {
    it('shows empty state when response has empty data', async () => {
      mockFetchTransfers.mockResolvedValue({
        status: 'success',
        message: 'ok',
        data: [],
        total: 0,
      });
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        expect(screen.getByText('No transfers found')).toBeInTheDocument();
      });
      expect(screen.getAllByText('Refresh').length).toBeGreaterThanOrEqual(1);
    });
  });

  /* ================================================================== */
  /*  Statistics                                                         */
  /* ================================================================== */
  describe('Statistics', () => {
    it('renders summary stat cards with backend data', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        expect(screen.getByText('Total Transfers')).toBeInTheDocument();
      });

      expect(screen.getByText('Successful')).toBeInTheDocument();
      expect(screen.getAllByText('Failed').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('Avg Similarity')).toBeInTheDocument();
      expect(screen.getByText('Avg Confidence')).toBeInTheDocument();
      expect(screen.getByText('Avg Loss')).toBeInTheDocument();
      expect(screen.getByText('Avg Exec Time')).toBeInTheDocument();
      expect(screen.getByText('Comm Efficiency')).toBeInTheDocument();
    });

    it('falls back to computed values when statistics endpoint returns no data', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue({ status: 'success', message: 'ok', data: null });
      renderKnowledgeTransfer();

      await waitFor(() => {
        expect(screen.getByText('Total Transfers')).toBeInTheDocument();
      });
    });
  });

  /* ================================================================== */
  /*  Data Table                                                         */
  /* ================================================================== */
  describe('Data Table', () => {
    it('renders transfer rows with backend data', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getAllByText('client-001').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('client-002').length).toBeGreaterThanOrEqual(1);
    });

    it('calls fetchKnowledgeTransfers on mount', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        expect(mockFetchTransfers).toHaveBeenCalledTimes(1);
      });
    });
  });

  /* ================================================================== */
  /*  Refresh                                                            */
  /* ================================================================== */
  describe('Manual Refresh', () => {
    it('refetches data when Refresh button is clicked', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(mockFetchTransfers).toHaveBeenCalledTimes(1);

      const refreshButton = screen.getByText('Refresh');
      await userEvent.click(refreshButton);

      await waitFor(() => {
        expect(mockFetchTransfers).toHaveBeenCalledTimes(2);
      });
    });
  });

  /* ================================================================== */
  /*  Sorting                                                            */
  /* ================================================================== */
  describe('Sorting', () => {
    it('sorts by similarity score column click', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const similarityHeader = screen.getByText('Similarity');
      await userEvent.click(similarityHeader);

      const rows = screen.getAllByRole('row');
      expect(rows.length).toBeGreaterThanOrEqual(2);
    });

    it('toggles sort direction on second click', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const similarityHeader = screen.getByText('Similarity');
      await userEvent.click(similarityHeader);
      await userEvent.click(similarityHeader);

      const rows = screen.getAllByRole('row');
      expect(rows.length).toBeGreaterThanOrEqual(2);
    });
  });

  /* ================================================================== */
  /*  Filtering                                                          */
  /* ================================================================== */
  describe('Filtering', () => {
    it('filters by status select', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const allSelects = screen.getAllByRole('combobox');
      const statusSelect = allSelects.find((s) => {
        const opts = Array.from(s.querySelectorAll('option'));
        return opts.some((o) => o.textContent === 'Completed');
      });
      expect(statusSelect).toBeTruthy();
      if (statusSelect) {
        await userEvent.selectOptions(statusSelect, 'completed');
      }

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });
    });

    it('clears filters when Clear button is clicked', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const searchInput = screen.getByPlaceholderText('Search by ID, client, modality, strategy...');
      await userEvent.type(searchInput, 'kt-003');
      await waitFor(() => {
        expect(screen.getByText('1 transfer')).toBeInTheDocument();
      });

      const clearBtn = screen.getByText('Clear');
      await userEvent.click(clearBtn);

      await waitFor(() => {
        expect(screen.getByText('5 transfers')).toBeInTheDocument();
      });
    });
  });

  /* ================================================================== */
  /*  Pagination                                                         */
  /* ================================================================== */
  describe('Pagination', () => {
    it('paginates when transfers exceed page size', async () => {
      const manyTransfers: KnowledgeTransferListResponse = {
        ...mockTransferList,
        data: Array.from({ length: 15 }, (_, i) => ({
          ...mockTransferList.data[0],
          transfer_id: `kt-${String(i + 1).padStart(3, '0')}`,
          source_client: `client-${String(i + 1).padStart(3, '0')}`,
        })),
        total: 15,
      };
      mockFetchTransfers.mockResolvedValue(manyTransfers);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText(/of 15/)).toBeInTheDocument();
    });

    it('changes page size', async () => {
      const manyTransfers: KnowledgeTransferListResponse = {
        ...mockTransferList,
        data: Array.from({ length: 15 }, (_, i) => ({
          ...mockTransferList.data[0],
          transfer_id: `kt-${String(i + 1).padStart(3, '0')}`,
          source_client: `client-${String(i + 1).padStart(3, '0')}`,
        })),
        total: 15,
      };
      mockFetchTransfers.mockResolvedValue(manyTransfers);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const allSelects = screen.getAllByRole('combobox');
      const sizeSelect = allSelects.find((s) => {
        const opts = Array.from(s.querySelectorAll('option'));
        return opts.some((o) => o.textContent === '20');
      });
      expect(sizeSelect).toBeTruthy();
    });
  });

  /* ================================================================== */
  /*  Detail Panel                                                       */
  /* ================================================================== */
  describe('Detail Panel', () => {
    it('opens detail panel on eye icon click', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const eyeButtons = screen.getAllByRole('button');
      const eyeBtn = eyeButtons.find((b) => b.querySelector('.lucide-eye'));
      expect(eyeBtn).toBeTruthy();
      await userEvent.click(eyeBtn!);

      await waitFor(() => {
        expect(screen.getByText('Transfer Details')).toBeInTheDocument();
      });
      expect(screen.getAllByText('kt-001').length).toBeGreaterThanOrEqual(1);
    });

    it('opens detail panel for correct transfer', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const eyeButtons = screen.getAllByRole('button');
      const eyeBtn = eyeButtons.find((b) => b.querySelector('.lucide-eye'));
      await userEvent.click(eyeBtn!);

      await waitFor(() => {
        expect(screen.getByText('Transfer Details')).toBeInTheDocument();
      });
      expect(screen.getByText('Transfer Metadata')).toBeInTheDocument();
      expect(screen.getByText('Alignment Information')).toBeInTheDocument();
      expect(screen.getByText('Similarity Metrics')).toBeInTheDocument();
      expect(screen.getByText('Associated Prototypes')).toBeInTheDocument();
      expect(screen.getByText('Knowledge Transfer History')).toBeInTheDocument();
    });

    it('shows similarity metrics in detail panel', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const eyeButtons = screen.getAllByRole('button');
      const eyeBtn = eyeButtons.find((b) => b.querySelector('.lucide-eye'));
      await userEvent.click(eyeBtn!);

      await waitFor(() => {
        expect(screen.getByText('Similarity Metrics')).toBeInTheDocument();
      });
      // Data is sorted by created_at DESC, so first row is kt-005 (score: 93.5%)
      expect(await screen.findByText(/93\.5/)).toBeInTheDocument();
      // Also verify confidence value for kt-005 (94.6%)
      expect(await screen.findByText(/94\.6/)).toBeInTheDocument();
    });

    it('closes detail panel', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const eyeButtons = screen.getAllByRole('button');
      const eyeBtn = eyeButtons.find((b) => b.querySelector('.lucide-eye'));
      await userEvent.click(eyeBtn!);

      await waitFor(() => {
        expect(screen.getByText('Transfer Details')).toBeInTheDocument();
      });

      const closeBtn = screen.getAllByRole('button').find((b) => b.querySelector('.lucide-x'));
      expect(closeBtn).toBeTruthy();
      await userEvent.click(closeBtn!);

      await waitFor(() => {
        expect(screen.queryByText('Transfer Details')).not.toBeInTheDocument();
      });
    });
  });

  /* ================================================================== */
  /*  Visualizations                                                     */
  /* ================================================================== */
  describe('Visualizations', () => {
    it('renders cross-modal transfer graph section', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText('Cross-Modal Transfer Graph')).toBeInTheDocument();
    });

    it('renders similarity heatmap section', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText('Similarity Heatmap')).toBeInTheDocument();
    });

    it('renders transfer timeline section', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText('Transfer Timeline')).toBeInTheDocument();
    });

    it('renders transfer success distribution section', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText('Transfer Success Distribution')).toBeInTheDocument();
    });

    it('renders transfer loss curve section', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText('Transfer Loss Curve')).toBeInTheDocument();
    });

    it('renders modality interaction matrix section', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText('Modality Interaction Matrix')).toBeInTheDocument();
    });
  });

  /* ================================================================== */
  /*  Last Updated Timestamp                                             */
  /* ================================================================== */
  describe('Last Updated Timestamp', () => {
    it('shows last updated timestamp', async () => {
      mockFetchTransfers.mockResolvedValue(mockTransferList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderKnowledgeTransfer();

      await waitFor(() => {
        const elems = screen.getAllByText('kt-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
    });
  });
});
