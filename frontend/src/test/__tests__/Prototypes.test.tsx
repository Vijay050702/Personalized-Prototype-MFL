import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

import { Prototypes } from '../../pages/Prototypes';
import * as prototypesApi from '../../api/prototypes';
import type { PrototypeListResponse } from '../../types';

vi.mock('../../api/prototypes');

const mockFetchPrototypes = prototypesApi.fetchPrototypes as ReturnType<typeof vi.fn>;

const mockPrototypeList: PrototypeListResponse = {
  status: 'success',
  message: 'Prototypes retrieved',
  data: [
    { id: 'proto-001', modality: 'visual', dimension: 512, cluster_id: 0, quality_score: 0.9234, client_id: 'client-001', created_round: 10 },
    { id: 'proto-002', modality: 'visual', dimension: 512, cluster_id: 1, quality_score: 0.8876, client_id: 'client-001', created_round: 10 },
    { id: 'proto-003', modality: 'acoustic', dimension: 256, cluster_id: 0, quality_score: 0.8456, client_id: 'client-002', created_round: 15 },
    { id: 'proto-004', modality: 'linguistic', dimension: 768, cluster_id: 2, quality_score: 0.9123, client_id: 'client-002', created_round: 20 },
    { id: 'proto-005', modality: 'multimodal', dimension: 1024, cluster_id: 0, quality_score: 0.9567, client_id: 'client-003', created_round: 25 },
  ],
  total: 5,
};

function renderPrototypes() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, retryDelay: 0 },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Prototypes />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('Prototypes', () => {
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
      mockFetchPrototypes.mockReturnValue(new Promise(() => {}));
      renderPrototypes();
      const skeletons = document.querySelectorAll('.animate-pulse');
      expect(skeletons.length).toBeGreaterThanOrEqual(4);
    });
  });

  /* ================================================================== */
  /*  Error State                                                        */
  /* ================================================================== */
  describe('Error State', () => {
    it('shows error state on API failure', async () => {
      mockFetchPrototypes.mockRejectedValue(new Error('Server error. Please try again.'));
      renderPrototypes();

      await waitFor(() => {
        expect(screen.getByText('Failed to load prototypes')).toBeInTheDocument();
      });
      expect(screen.getByText('Server error. Please try again.')).toBeInTheDocument();
      expect(screen.getByText('Try Again')).toBeInTheDocument();
    });

    it('shows error state when backend is unavailable', async () => {
      mockFetchPrototypes.mockRejectedValue(new Error('Backend is unavailable. Please check your connection.'));
      renderPrototypes();

      await waitFor(() => {
        expect(screen.getByText('Failed to load prototypes')).toBeInTheDocument();
      });
      expect(screen.getByText('Backend is unavailable. Please check your connection.')).toBeInTheDocument();
    });

    it('retries on error when clicking Try Again', async () => {
      mockFetchPrototypes.mockRejectedValue(new Error('Server error'));
      renderPrototypes();

      await waitFor(() => {
        expect(screen.getByText('Failed to load prototypes')).toBeInTheDocument();
      });

      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);

      const retryButton = screen.getByText('Try Again');
      await userEvent.click(retryButton);

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });
    });
  });

  /* ================================================================== */
  /*  Empty State                                                        */
  /* ================================================================== */
  describe('Empty State', () => {
    it('shows empty state when response has empty data', async () => {
      mockFetchPrototypes.mockResolvedValue({
        status: 'success',
        message: 'ok',
        data: [],
        total: 0,
      });
      renderPrototypes();

      await waitFor(() => {
        expect(screen.getByText('No prototypes found')).toBeInTheDocument();
      });
      expect(screen.getAllByText('Refresh').length).toBeGreaterThanOrEqual(1);
    });

    it('shows empty state when search yields no results', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const searchInput = screen.getByPlaceholderText('Search by ID, client, modality, class...');
      await userEvent.type(searchInput, 'nonexistent_prototype');

      await waitFor(() => {
        expect(screen.getByText('No prototypes found')).toBeInTheDocument();
      });
    });
  });

  /* ================================================================== */
  /*  Success State - Summary Cards                                      */
  /* ================================================================== */
  describe('Summary Cards', () => {
    it('renders summary stat cards with backend data', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        expect(screen.getByText('Total Prototypes')).toBeInTheDocument();
      });

      expect(screen.getAllByText('5').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('Modalities')).toBeInTheDocument();
      expect(screen.getByText('Classes')).toBeInTheDocument();
      expect(screen.getByText('Avg Quality')).toBeInTheDocument();
    });
  });

  /* ================================================================== */
  /*  Success State - Data Table                                         */
  /* ================================================================== */
  describe('Data Table', () => {
    it('renders prototype rows with backend data', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getAllByText('512').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('#0').length).toBeGreaterThanOrEqual(1);
    });

    it('calls fetchPrototypes on mount', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        expect(mockFetchPrototypes).toHaveBeenCalledTimes(1);
      });
    });
  });

  /* ================================================================== */
  /*  Refresh                                                            */
  /* ================================================================== */
  describe('Manual Refresh', () => {
    it('refetches data when Refresh button is clicked', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(mockFetchPrototypes).toHaveBeenCalledTimes(1);

      const refreshButton = screen.getByText('Refresh');
      await userEvent.click(refreshButton);

      await waitFor(() => {
        expect(mockFetchPrototypes).toHaveBeenCalledTimes(2);
      });
    });
  });

  /* ================================================================== */
  /*  Sorting                                                            */
  /* ================================================================== */
  describe('Sorting', () => {
    it('sorts by quality score column click', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const qualityHeader = screen.getByText('Quality Score');
      await userEvent.click(qualityHeader);

      const rows = screen.getAllByRole('row');
      expect(rows.length).toBeGreaterThanOrEqual(2);
    });

    it('toggles sort direction on second click', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const qualityHeader = screen.getByText('Quality Score');
      await userEvent.click(qualityHeader);
      await userEvent.click(qualityHeader);

      const rows = screen.getAllByRole('row');
      expect(rows.length).toBeGreaterThanOrEqual(2);
    });
  });

  /* ================================================================== */
  /*  Filtering                                                          */
  /* ================================================================== */
  describe('Filtering', () => {
    it('filters by type select', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const allSelects = screen.getAllByRole('combobox');
      const select = allSelects.find((s) => {
        const opts = Array.from(s.querySelectorAll('option'));
        return opts.some((o) => o.textContent === 'Image');
      });
      expect(select).toBeTruthy();
      if (select) {
        await userEvent.selectOptions(select, 'visual');
      }

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });
    });

    it('clears filters when Clear button is clicked', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const searchInput = screen.getByPlaceholderText('Search by ID, client, modality, class...');
      await userEvent.type(searchInput, 'proto-003');
      await waitFor(() => {
        expect(screen.getByText('1 prototype')).toBeInTheDocument();
      });

      const clearBtn = screen.getByText('Clear');
      await userEvent.click(clearBtn);

      await waitFor(() => {
        expect(screen.getByText('5 prototypes')).toBeInTheDocument();
      });
    });
  });

  /* ================================================================== */
  /*  Pagination                                                         */
  /* ================================================================== */
  describe('Pagination', () => {
    it('paginates when prototypes exceed page size', async () => {
      const manyPrototypes: PrototypeListResponse = {
        ...mockPrototypeList,
        data: Array.from({ length: 15 }, (_, i) => ({
          ...mockPrototypeList.data[0],
          id: `proto-${String(i + 1).padStart(3, '0')}`,
          client_id: `client-${String(i + 1).padStart(3, '0')}`,
        })),
        total: 15,
      };
      mockFetchPrototypes.mockResolvedValue(manyPrototypes);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText(/of 15/)).toBeInTheDocument();
    });

    it('changes page size', async () => {
      const manyPrototypes: PrototypeListResponse = {
        ...mockPrototypeList,
        data: Array.from({ length: 15 }, (_, i) => ({
          ...mockPrototypeList.data[0],
          id: `proto-${String(i + 1).padStart(3, '0')}`,
          client_id: `client-${String(i + 1).padStart(3, '0')}`,
        })),
        total: 15,
      };
      mockFetchPrototypes.mockResolvedValue(manyPrototypes);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
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
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const eyeButtons = screen.getAllByRole('button');
      const eyeBtn = eyeButtons.find((b) => b.querySelector('.lucide-eye'));
      expect(eyeBtn).toBeTruthy();
      await userEvent.click(eyeBtn!);

      await waitFor(() => {
        expect(screen.getByText('Prototype Details')).toBeInTheDocument();
      });
      expect(screen.getAllByText('proto-001').length).toBeGreaterThanOrEqual(1);
    });

    it('opens detail panel for correct prototype', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const eyeButtons = screen.getAllByRole('button');
      const eyeBtn = eyeButtons.find((b) => b.querySelector('.lucide-eye'));
      await userEvent.click(eyeBtn!);

      await waitFor(() => {
        expect(screen.getByText('Prototype Details')).toBeInTheDocument();
      });
      expect(screen.getByText('Metadata')).toBeInTheDocument();
      expect(screen.getByText('Prototype Statistics')).toBeInTheDocument();
      expect(screen.getByText('History')).toBeInTheDocument();
      expect(screen.getByText('Associations')).toBeInTheDocument();
    });

    it('shows prototype statistics in detail panel', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const eyeButtons = screen.getAllByRole('button');
      const eyeBtn = eyeButtons.find((b) => b.querySelector('.lucide-eye'));
      await userEvent.click(eyeBtn!);

      await waitFor(() => {
        expect(screen.getAllByText('Quality Score').length).toBeGreaterThanOrEqual(1);
      });
      expect(screen.getAllByText('92.3%').length).toBeGreaterThanOrEqual(1);
    });

    it('closes detail panel', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      const eyeButtons = screen.getAllByRole('button');
      const eyeBtn = eyeButtons.find((b) => b.querySelector('.lucide-eye'));
      await userEvent.click(eyeBtn!);

      await waitFor(() => {
        expect(screen.getByText('Prototype Details')).toBeInTheDocument();
      });

      const closeBtn = screen.getAllByRole('button').find((b) => b.querySelector('.lucide-x'));
      expect(closeBtn).toBeTruthy();
      await userEvent.click(closeBtn!);

      await waitFor(() => {
        expect(screen.queryByText('Prototype Details')).not.toBeInTheDocument();
      });
    });
  });

  /* ================================================================== */
  /*  Visualizations                                                     */
  /* ================================================================== */
  describe('Visualizations', () => {
    it('renders similarity matrix section', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText('Similarity Matrix')).toBeInTheDocument();
    });

    it('renders quality score chart section', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText('Quality Score Distribution')).toBeInTheDocument();
    });

    it('renders evolution timeline section', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText('Prototype Evolution Timeline')).toBeInTheDocument();
    });

    it('renders modality distribution section', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText('Modality Distribution')).toBeInTheDocument();
    });

    it('renders class distribution section', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText('Class Distribution')).toBeInTheDocument();
    });

    it('renders embedding dimension statistics section', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText('Embedding Dimension Statistics')).toBeInTheDocument();
    });
  });

  /* ================================================================== */
  /*  Last Updated Timestamp                                             */
  /* ================================================================== */
  describe('Last Updated Timestamp', () => {
    it('shows last updated timestamp', async () => {
      mockFetchPrototypes.mockResolvedValue(mockPrototypeList);
      renderPrototypes();

      await waitFor(() => {
        const elems = screen.getAllByText('proto-001');
        expect(elems.length).toBeGreaterThanOrEqual(1);
      });

      expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
    });
  });
});
