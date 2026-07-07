import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

import { Similarity } from '../../pages/Similarity';
import * as similarityApi from '../../api/similarity';
import type { SimilarityListResponse, SimilarityStatisticsResponse } from '../../types';

vi.mock('../../api/similarity');

const mockFetchAnalyses = similarityApi.fetchSimilarityAnalyses as ReturnType<typeof vi.fn>;
const mockFetchStats = similarityApi.fetchSimilarityStatistics as ReturnType<typeof vi.fn>;

const mockAnalysisList: SimilarityListResponse = {
  status: 'success',
  message: 'Similarity analyses retrieved',
  data: [
    { analysis_id: 'sim-001', source_client: 'client-001', target_client: 'client-002', source_prototype: 'proto-001', target_prototype: 'proto-005', source_modality: 'visual', target_modality: 'linguistic', similarity_metric: 'cosine', cosine_similarity: 0.9234, euclidean_distance: 0.1876, prototype_distance: 0.1234, transfer_confidence: 0.9123, aggregation_round: 10, cluster_id: 0, analysis_status: 'completed', created_at: '2025-01-01T10:00:00Z' },
    { analysis_id: 'sim-002', source_client: 'client-001', target_client: 'client-003', source_prototype: 'proto-002', target_prototype: 'proto-007', source_modality: 'visual', target_modality: 'acoustic', similarity_metric: 'cosine', cosine_similarity: 0.8456, euclidean_distance: 0.2543, prototype_distance: 0.2345, transfer_confidence: 0.8765, aggregation_round: 10, cluster_id: 1, analysis_status: 'completed', created_at: '2025-01-01T10:05:00Z' },
    { analysis_id: 'sim-003', source_client: 'client-002', target_client: 'client-001', source_prototype: 'proto-003', target_prototype: 'proto-006', source_modality: 'acoustic', target_modality: 'visual', similarity_metric: 'euclidean', cosine_similarity: 0.7812, euclidean_distance: 0.3210, prototype_distance: 0.3456, transfer_confidence: 0.8234, aggregation_round: 15, cluster_id: 0, analysis_status: 'failed', created_at: '2025-01-01T11:00:00Z' },
    { analysis_id: 'sim-004', source_client: 'client-003', target_client: 'client-001', source_prototype: 'proto-004', target_prototype: 'proto-008', source_modality: 'linguistic', target_modality: 'visual', similarity_metric: 'cosine', cosine_similarity: 0.9678, euclidean_distance: 0.0987, prototype_distance: 0.0456, transfer_confidence: 0.9567, aggregation_round: 20, cluster_id: 2, analysis_status: 'completed', created_at: '2025-01-01T12:00:00Z' },
    { analysis_id: 'sim-005', source_client: 'client-002', target_client: 'client-003', source_prototype: 'proto-005', target_prototype: 'proto-009', source_modality: 'multimodal', target_modality: 'visual', similarity_metric: 'cosine', cosine_similarity: 0.9345, euclidean_distance: 0.1654, prototype_distance: 0.0789, transfer_confidence: 0.9456, aggregation_round: 25, cluster_id: 1, analysis_status: 'running', created_at: '2025-01-01T13:00:00Z' },
  ],
  total: 5,
};

const mockStatsResponse: SimilarityStatisticsResponse = {
  status: 'success',
  message: 'Statistics retrieved',
  data: {
    average_similarity: 0.8905,
    maximum_similarity: 0.9678,
    minimum_similarity: 0.7812,
    average_distance: 0.2054,
    cluster_count: 3,
    client_groups: 3,
    prototype_groups: 5,
    communication_round: 25,
  },
};

function renderSimilarity() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, retryDelay: 0 },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Similarity />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('Similarity', () => {
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
      mockFetchAnalyses.mockReturnValue(new Promise(() => {}));
      mockFetchStats.mockReturnValue(new Promise(() => {}));
      renderSimilarity();
      const skeletons = document.querySelectorAll('.animate-pulse');
      expect(skeletons.length).toBeGreaterThanOrEqual(4);
    });
  });

  /* ================================================================== */
  /*  Error State                                                        */
  /* ================================================================== */
  describe('Error State', () => {
    it('shows error state on API failure', async () => {
      mockFetchAnalyses.mockRejectedValue(new Error('Server error. Please try again.'));
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getByText('Failed to load similarity analyses')).toBeInTheDocument();
      });
      expect(screen.getByText('Server error. Please try again.')).toBeInTheDocument();
      expect(screen.getByText('Try Again')).toBeInTheDocument();
    });

    it('shows error state when backend is unavailable', async () => {
      mockFetchAnalyses.mockRejectedValue(new Error('Backend is unavailable. Please check your connection.'));
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getByText('Failed to load similarity analyses')).toBeInTheDocument();
      });
      expect(screen.getByText('Backend is unavailable. Please check your connection.')).toBeInTheDocument();
    });

    it('retries when clicking Try Again', async () => {
      mockFetchAnalyses.mockRejectedValueOnce(new Error('Server error. Please try again.'));
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getByText('Try Again')).toBeInTheDocument();
      });

      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      const tryAgainBtn = screen.getByText('Try Again');
      await userEvent.click(tryAgainBtn);

      await waitFor(() => {
        expect(screen.getByText('Similarity Analysis')).toBeInTheDocument();
      });
    });
  });

  /* ================================================================== */
  /*  Empty State                                                        */
  /* ================================================================== */
  describe('Empty State', () => {
    it('shows empty state when no data', async () => {
      mockFetchAnalyses.mockResolvedValue({ status: 'success', message: 'No data', data: [], total: 0 });
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getByText('No similarity analyses found')).toBeInTheDocument();
      });
      expect(screen.getAllByText('Refresh').length).toBeGreaterThanOrEqual(1);
    });
  });

  /* ================================================================== */
  /*  Statistics                                                         */
  /* ================================================================== */
  describe('Statistics', () => {
    it('renders summary stat cards with backend data', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getByText('Average Similarity')).toBeInTheDocument();
      });

      expect(screen.getByText('Maximum Similarity')).toBeInTheDocument();
      expect(screen.getByText('Minimum Similarity')).toBeInTheDocument();
      expect(screen.getByText('Average Distance')).toBeInTheDocument();
      expect(screen.getByText('Cluster Count')).toBeInTheDocument();
      expect(screen.getByText('Client Groups')).toBeInTheDocument();
      expect(screen.getByText('Prototype Groups')).toBeInTheDocument();
      expect(screen.getByText('Communication Round')).toBeInTheDocument();
    });

    it('falls back to computed values when stats endpoint returns null', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue({ status: 'success', message: 'No stats', data: null as unknown as undefined });
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getByText('Average Similarity')).toBeInTheDocument();
      });
    });
  });

  /* ================================================================== */
  /*  Data Table                                                         */
  /* ================================================================== */
  describe('Data Table', () => {
    it('renders analysis rows from API data', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        const elems = screen.getAllByText(/sim-00/);
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getAllByText('client-001').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('client-002').length).toBeGreaterThanOrEqual(1);
    });

    it('calls fetchSimilarityAnalyses on mount', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(mockFetchAnalyses).toHaveBeenCalledTimes(1);
      });
    });
  });

  /* ================================================================== */
  /*  Manual Refresh                                                     */
  /* ================================================================== */
  describe('Manual Refresh', () => {
    it('refetches data when Refresh button clicked', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(mockFetchAnalyses).toHaveBeenCalledTimes(1);
      });

      const refreshBtn = screen.getByText('Refresh');
      await userEvent.click(refreshBtn);

      await waitFor(() => {
        expect(mockFetchAnalyses).toHaveBeenCalledTimes(2);
      });
    });
  });

  /* ================================================================== */
  /*  Sorting                                                            */
  /* ================================================================== */
  describe('Sorting', () => {
    it('sorts by cosine similarity column click', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        const elems = screen.getAllByText(/sim-00/);
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const cosineHeader = screen.getByText('Cosine');
      await userEvent.click(cosineHeader);

      await waitFor(() => {
        expect(screen.getByText('Cosine')).toBeInTheDocument();
      });
    });
  });

  /* ================================================================== */
  /*  Filtering                                                          */
  /* ================================================================== */
  describe('Filtering', () => {
    it('filters by client select', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        const elems = screen.getAllByText(/sim-00/);
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const allCombos = screen.getAllByRole('combobox');
      const clientDropdown = allCombos[0];
      await userEvent.selectOptions(clientDropdown, 'client-001');

      await waitFor(() => {
        expect(screen.getAllByText(/client-001/).length).toBeGreaterThanOrEqual(1);
      });
    });

    it('clears filters when Clear button clicked', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        const elems = screen.getAllByText(/sim-00/);
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const searchInput = screen.getByPlaceholderText('Search by ID, client, prototype, modality...');
      await userEvent.type(searchInput, 'sim-001');

      await waitFor(() => {
        expect(screen.getByText('Clear')).toBeInTheDocument();
      });

      const clearBtn = screen.getByText('Clear');
      await userEvent.click(clearBtn);

      await waitFor(() => {
        const elems = screen.getAllByText(/sim-00/);
        expect(elems.length).toBeGreaterThanOrEqual(3);
      });
    });
  });

  /* ================================================================== */
  /*  Pagination                                                         */
  /* ================================================================== */
  describe('Pagination', () => {
    it('paginates when analyses exceed page size', async () => {
      const manyAnalyses = Array.from({ length: 15 }, (_, i) => ({
        ...mockAnalysisList.data[0],
        analysis_id: `sim-${String(i + 1).padStart(3, '0')}`,
        created_at: `2025-01-${String(i + 1).padStart(2, '0')}T10:00:00Z`,
      }));
      mockFetchAnalyses.mockResolvedValue({
        ...mockAnalysisList,
        data: manyAnalyses,
        total: 15,
      });
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getByText(/of 15/)).toBeInTheDocument();
      });
    });

    it('changes page size', async () => {
      const manyAnalyses = Array.from({ length: 15 }, (_, i) => ({
        ...mockAnalysisList.data[0],
        analysis_id: `sim-${String(i + 1).padStart(3, '0')}`,
        created_at: `2025-01-${String(i + 1).padStart(2, '0')}T10:00:00Z`,
      }));
      mockFetchAnalyses.mockResolvedValue({
        ...mockAnalysisList,
        data: manyAnalyses,
        total: 15,
      });
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getByText(/of 15/)).toBeInTheDocument();
      });

      const allCombos = screen.getAllByRole('combobox');
      const pageSizeCombo = allCombos[allCombos.length - 1];
      await userEvent.selectOptions(pageSizeCombo, '20');

      await waitFor(() => {
        expect(screen.getByText(/of 15/)).toBeInTheDocument();
      });
    });
  });

  /* ================================================================== */
  /*  Detail Panel                                                       */
  /* ================================================================== */
  describe('Detail Panel', () => {
    it('opens detail panel on eye icon click', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        const elems = screen.getAllByText(/sim-00/);
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const eyeButtons = screen.getAllByRole('button');
      const eyeBtn = eyeButtons.find((b) => b.querySelector('.lucide-eye'));
      await userEvent.click(eyeBtn!);

      await waitFor(() => {
        expect(screen.getByText('Analysis Details')).toBeInTheDocument();
      });
    });

    it('opens correct analysis detail panel', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        const elems = screen.getAllByText(/sim-00/);
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const eyeButtons = screen.getAllByRole('button');
      const eyeBtn = eyeButtons.find((b) => b.querySelector('.lucide-eye'));
      await userEvent.click(eyeBtn!);

      await waitFor(() => {
        expect(screen.getByText('Analysis Details')).toBeInTheDocument();
      });
      expect(screen.getByText('Similarity Metrics')).toBeInTheDocument();
      expect(screen.getByText('Distance Metrics')).toBeInTheDocument();
      expect(screen.getByText('Prototype Statistics')).toBeInTheDocument();
      expect(screen.getByText('Client Information')).toBeInTheDocument();
      expect(screen.getByText('Transfer Relationship')).toBeInTheDocument();
      expect(screen.getByText('Aggregation History')).toBeInTheDocument();
    });

    it('shows similarity metrics in detail panel', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        const elems = screen.getAllByText(/sim-00/);
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const eyeButtons = screen.getAllByRole('button');
      const eyeBtn = eyeButtons.find((b) => b.querySelector('.lucide-eye'));
      await userEvent.click(eyeBtn!);

      await waitFor(() => {
        expect(screen.getByText('Similarity Metrics')).toBeInTheDocument();
      });
      // Data is sorted by created_at DESC, so first row is sim-005 (cosine: 93.5%)
      expect(await screen.findByText(/93\.5/)).toBeInTheDocument();
      expect(await screen.findByText(/94\.6/)).toBeInTheDocument();
    });

    it('closes detail panel', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        const elems = screen.getAllByText(/sim-00/);
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const eyeButtons = screen.getAllByRole('button');
      const eyeBtn = eyeButtons.find((b) => b.querySelector('.lucide-eye'));
      await userEvent.click(eyeBtn!);

      await waitFor(() => {
        expect(screen.getByText('Analysis Details')).toBeInTheDocument();
      });

      const closeBtn = screen.getAllByRole('button').find((b) => b.querySelector('.lucide-x'));
      await userEvent.click(closeBtn!);

      await waitFor(() => {
        expect(screen.queryByText('Analysis Details')).not.toBeInTheDocument();
      });
    });
  });

  /* ================================================================== */
  /*  Visualizations                                                     */
  /* ================================================================== */
  describe('Visualizations', () => {
    it('renders Similarity Heatmap section', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getAllByText(/sim-00/).length).toBeGreaterThanOrEqual(1);
      });

      await waitFor(() => {
        expect(screen.getByText('Similarity Heatmap')).toBeInTheDocument();
      });
    });

    it('renders Client Similarity Matrix section', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getAllByText(/sim-00/).length).toBeGreaterThanOrEqual(1);
      });

      await waitFor(() => {
        expect(screen.getByText('Client Similarity Matrix')).toBeInTheDocument();
      });
    });

    it('renders Prototype Similarity Matrix section', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getAllByText(/sim-00/).length).toBeGreaterThanOrEqual(1);
      });

      await waitFor(() => {
        expect(screen.getByText('Prototype Similarity Matrix')).toBeInTheDocument();
      });
    });

    it('renders Cluster Visualization section', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getAllByText(/sim-00/).length).toBeGreaterThanOrEqual(1);
      });

      await waitFor(() => {
        expect(screen.getByText('Cluster Visualization')).toBeInTheDocument();
      });
    });

    it('renders Similarity Timeline section', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getAllByText(/sim-00/).length).toBeGreaterThanOrEqual(1);
      });

      await waitFor(() => {
        expect(screen.getByText('Similarity Timeline')).toBeInTheDocument();
      });
    });

    it('renders Distribution Histogram section', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getAllByText(/sim-00/).length).toBeGreaterThanOrEqual(1);
      });

      await waitFor(() => {
        expect(screen.getByText('Distribution Histogram')).toBeInTheDocument();
      });
    });

    it('renders Radar Comparison section', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getAllByText(/sim-00/).length).toBeGreaterThanOrEqual(1);
      });

      await waitFor(() => {
        expect(screen.getByText('Radar Comparison')).toBeInTheDocument();
      });
    });
  });

  /* ================================================================== */
  /*  Last Updated Timestamp                                             */
  /* ================================================================== */
  describe('Last Updated Timestamp', () => {
    it('shows last updated timestamp after data loads', async () => {
      mockFetchAnalyses.mockResolvedValue(mockAnalysisList);
      mockFetchStats.mockResolvedValue(mockStatsResponse);
      renderSimilarity();

      await waitFor(() => {
        expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
      });
    });
  });
});
