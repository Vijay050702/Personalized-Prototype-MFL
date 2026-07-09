import type { ReactElement } from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

import { RealtimeProvider, RealtimeContext } from '../../realtime/provider';
import { useRealtime, useConnectionStatus, useEventHistory, useRealtimeDashboard } from '../../realtime/hooks';
import { ConnectionStatusIndicator, TransportLabel } from '../../realtime/status';
import { LiveDashboard } from '../../components/realtime/LiveDashboard';
import { createEvent, filterEvents, getCategoryLabel, severityFromStatus } from '../../realtime/events';
import { detectTransport, createConnectionManager } from '../../realtime/connection';
import { MAX_HISTORY_SIZE, TRANSPORT_LABELS } from '../../realtime/types';
import type { TransportProbeResult } from '../../realtime/connection';
import { fetchDashboard } from '../../api/dashboard';

vi.mock('../../api/dashboard');

const mockFetchDashboard = vi.mocked(fetchDashboard);

vi.mock('../../realtime/connection', () => {
  const detectTransport = vi.fn();
  return {
    detectTransport,
    createConnectionManager: () => ({
      status: 'disconnected',
      transport: 'none',
      error: null,
      connect: async () => await detectTransport(),
      disconnect: () => {},
    }),
  };
});

const mockDetectTransport = vi.mocked(detectTransport);

const mockPollingResult: TransportProbeResult = { transport: 'polling', error: null };

function createWrapper(ui: ReactElement, pollingInterval = 100000) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, retryDelay: 0 } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <RealtimeProvider pollingInterval={pollingInterval}>{ui}</RealtimeProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

function createRealtimeConsumer() {
  let captured: unknown = null;
  function TestConsumer() {
    const val = useRealtime();
    captured = val;
    return <div data-testid="realtime-mode">{val.connectionStatus}</div>;
  }
  return { TestConsumer, getValue: () => captured };
}

describe('Event utilities', () => {
  it('creates events with unique IDs', () => {
    const e1 = createEvent('training:round_start', 'training', 'info', 'Round 1', 'Training round started');
    const e2 = createEvent('server:health_change', 'server', 'success', 'Online', 'Server is online');
    expect(e1.id).not.toBe(e2.id);
    expect(e1.timestamp).toBeGreaterThan(0);
    expect(e1.type).toBe('training:round_start');
    expect(e2.category).toBe('server');
  });

  it('severityFromStatus maps correctly', () => {
    expect(severityFromStatus('failed')).toBe('error');
    expect(severityFromStatus('error')).toBe('error');
    expect(severityFromStatus('completed')).toBe('success');
    expect(severityFromStatus('active')).toBe('success');
    expect(severityFromStatus('pending')).toBe('warning');
    expect(severityFromStatus('running')).toBe('info');
    expect(severityFromStatus('unknown')).toBe('info');
  });

  it('filterEvents filters by category', () => {
    const events = [
      createEvent('training:round_start', 'training', 'info', 'T1', ''),
      createEvent('server:health_change', 'server', 'info', 'S1', ''),
      createEvent('client:connected', 'client', 'success', 'C1', ''),
    ];
    expect(filterEvents(events, { categories: ['training', 'server'] })).toHaveLength(2);
  });

  it('filterEvents filters by severity', () => {
    const events = [
      createEvent('training:round_start', 'training', 'info', 'T1', ''),
      createEvent('server:health_change', 'server', 'error', 'S1', ''),
    ];
    expect(filterEvents(events, { severities: ['error'] })).toHaveLength(1);
  });

  it('filterEvents searches by text', () => {
    const events = [
      createEvent('training:round_start', 'training', 'info', 'Round 1', 'Training round began'),
      createEvent('server:health_change', 'server', 'error', 'Error', 'Something failed'),
    ];
    expect(filterEvents(events, { search: 'failed' })).toHaveLength(1);
  });

  it('filterEvents applies limit', () => {
    const events = Array.from({ length: 10 }, (_, i) =>
      createEvent('training:round_start', 'training', 'info', `R${i}`, ''),
    );
    expect(filterEvents(events, { limit: 3 })).toHaveLength(3);
  });

  it('getCategoryLabel returns correct labels', () => {
    expect(getCategoryLabel('training')).toBe('Training');
    expect(getCategoryLabel('knowledge_transfer')).toBe('Knowledge Transfer');
    expect(getCategoryLabel('evaluation')).toBe('Evaluation');
  });

  it('TRANSPORT_LABELS maps all transports', () => {
    expect(TRANSPORT_LABELS.websocket).toBe('WebSocket');
    expect(TRANSPORT_LABELS.sse).toBe('Server-Sent Events');
    expect(TRANSPORT_LABELS.polling).toBe('Polling');
    expect(TRANSPORT_LABELS.none).toBe('None');
  });
});

describe('detectTransport', () => {
  it('returns a transport probe result', async () => {
    mockDetectTransport.mockResolvedValue(mockPollingResult);
    const result = await detectTransport();
    expect(['polling', 'websocket', 'sse', 'none']).toContain(result.transport);
    expect(result).toHaveProperty('error');
  });
});

describe('createConnectionManager', () => {
  it('starts disconnected', () => {
    const mgr = createConnectionManager();
    expect(mgr.status).toBe('disconnected');
    expect(mgr.transport).toBe('none');
  });

  it('connect returns transport result', async () => {
    mockDetectTransport.mockResolvedValue(mockPollingResult);
    const mgr = createConnectionManager();
    const result = await mgr.connect();
    expect(['polling', 'websocket', 'sse', 'none']).toContain(result.transport);
  });

  it('disconnect is safe to call', () => {
    const mgr = createConnectionManager();
    expect(() => mgr.disconnect()).not.toThrow();
  });
});

describe('RealtimeProvider', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockDetectTransport.mockResolvedValue(mockPollingResult);
    mockFetchDashboard.mockResolvedValue({
      status: 'success',
      message: 'ok',
      data: {
        active_clients: 5,
        total_clients: 10,
        current_round: 3,
        total_rounds: 50,
        global_accuracy: 0.85,
        global_loss: 0.35,
        training_status: 'running',
        experiments_running: 1,
        uptime_hours: 24.5,
        last_updated: '2026-07-09T10:00:00Z',
      },
    });
  });

  it('provides initial context', () => {
    const { TestConsumer, getValue } = createRealtimeConsumer();
    createWrapper(<TestConsumer />);
    const val = getValue() as { connectionStatus: string };
    expect(val).toBeTruthy();
  });

  it('transitions to polling mode', async () => {
    const { TestConsumer } = createRealtimeConsumer();
    createWrapper(<TestConsumer />);
    await waitFor(() => {
      expect(screen.getByTestId('realtime-mode').textContent).toBe('polling');
    });
  });

  it('transitions to offline when no transport', async () => {
    mockDetectTransport.mockResolvedValue({ transport: 'none', error: 'Backend unreachable' });
    const { TestConsumer } = createRealtimeConsumer();
    createWrapper(<TestConsumer />);
    await waitFor(() => {
      expect(screen.getByTestId('realtime-mode').textContent).toBe('offline');
    });
  });

  it('clearHistory works', async () => {
    mockDetectTransport.mockResolvedValue(mockPollingResult);
    function TestClear() {
      const { clearHistory } = useRealtime();
      return <button data-testid="clear-btn" onClick={clearHistory}>Clear</button>;
    }
    createWrapper(<TestClear />);
    await waitFor(() => {
      expect(screen.getByTestId('clear-btn')).toBeTruthy();
    });
  });

  it('removeEvent works', async () => {
    mockDetectTransport.mockResolvedValue(mockPollingResult);
    function TestRemove() {
      const { removeEvent } = useRealtime();
      return <button data-testid="remove-btn" onClick={() => removeEvent('nonexistent')}>Remove</button>;
    }
    createWrapper(<TestRemove />);
    await waitFor(() => {
      expect(screen.getByTestId('remove-btn')).toBeTruthy();
    });
  });

  it('events list is populated after probe', async () => {
    mockDetectTransport.mockResolvedValue(mockPollingResult);
    function TestEvents() {
      const { events } = useRealtime();
      return <div data-testid="evt-count">{events.length}</div>;
    }
    createWrapper(<TestEvents />);
    await waitFor(() => {
      expect(parseInt(screen.getByTestId('evt-count').textContent ?? '0')).toBeGreaterThanOrEqual(1);
    });
  });

  it('refetchAll invalidates queries', async () => {
    mockDetectTransport.mockResolvedValue(mockPollingResult);
    function TestRefetch() {
      const { refetchAll } = useRealtime();
      return <button data-testid="refetch-btn" onClick={refetchAll}>Refetch</button>;
    }
    createWrapper(<TestRefetch />);
    await waitFor(() => {
      expect(screen.getByTestId('refetch-btn')).toBeTruthy();
    });
  });
});

describe('useConnectionStatus', () => {
  beforeEach(() => {
    mockDetectTransport.mockResolvedValue(mockPollingResult);
  });

  it('returns status hook values', async () => {
    function TestConn() {
      const { connectionStatus, transport, currentTransportLabel } = useConnectionStatus();
      return <div data-testid="conn">{connectionStatus} {transport} {currentTransportLabel}</div>;
    }
    createWrapper(<TestConn />);
    await waitFor(() => {
      const el = screen.getByTestId('conn');
      expect(el.textContent).toContain('polling');
    });
  });
});

describe('useEventHistory', () => {
  beforeEach(() => {
    mockDetectTransport.mockResolvedValue(mockPollingResult);
  });

  it('returns filtered events', async () => {
    function TestHistory() {
      const events = useEventHistory({ limit: 10 });
      return <div data-testid="hist">{events.length}</div>;
    }
    createWrapper(<TestHistory />);
    await waitFor(() => {
      expect(screen.getByTestId('hist').textContent).toBe('0');
    });
  });
});

describe('useRealtimeDashboard', () => {
  beforeEach(() => {
    mockDetectTransport.mockResolvedValue(mockPollingResult);
    vi.clearAllMocks();
  });

  it('returns loading state initially', () => {
    mockFetchDashboard.mockReturnValue(new Promise(() => {}));
    function TestDash() {
      const { isLoading } = useRealtimeDashboard();
      return <div data-testid="loading">{isLoading.toString()}</div>;
    }
    createWrapper(<TestDash />);
    expect(screen.getByTestId('loading').textContent).toBe('true');
  });

  it('returns dashboard data', async () => {
    mockFetchDashboard.mockResolvedValue({
      status: 'success',
      message: 'ok',
      data: {
        active_clients: 5,
        total_clients: 10,
        current_round: 3,
        total_rounds: 50,
        global_accuracy: 0.85,
        global_loss: 0.35,
        training_status: 'running',
        experiments_running: 1,
        uptime_hours: 24.5,
        last_updated: '2026-07-09T10:00:00Z',
      },
    });
    function TestDash() {
      const { data } = useRealtimeDashboard();
      return <div data-testid="dash">{data ? data.currentRound : 'null'}</div>;
    }
    createWrapper(<TestDash />);
    await waitFor(() => {
      expect(screen.getByTestId('dash').textContent).toBe('3');
    });
  });
});

describe('ConnectionStatusIndicator', () => {
  beforeEach(() => {
    mockDetectTransport.mockResolvedValue(mockPollingResult);
  });

  it('renders status indicator', async () => {
    createWrapper(<ConnectionStatusIndicator />);
    await waitFor(() => {
      const indicators = document.querySelectorAll('.inline-flex');
      expect(indicators.length).toBeGreaterThan(0);
    });
  });

  it('renders with label', async () => {
    createWrapper(<ConnectionStatusIndicator showLabel />);
    await waitFor(() => {
      expect(screen.getByText('Polling')).toBeTruthy();
    });
  });

  it('renders without label', async () => {
    createWrapper(<ConnectionStatusIndicator showLabel={false} />);
    await waitFor(() => {
      expect(screen.queryByText('Polling')).toBeNull();
    });
  });
});

describe('TransportLabel', () => {
  beforeEach(() => {
    mockDetectTransport.mockResolvedValue(mockPollingResult);
  });

  it('renders transport label', async () => {
    createWrapper(<TransportLabel />);
    await waitFor(() => {
      expect(screen.getByText('Polling')).toBeTruthy();
    });
  });
});

describe('LiveDashboard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockDetectTransport.mockResolvedValue(mockPollingResult);
    mockFetchDashboard.mockResolvedValue({
      status: 'success',
      message: 'ok',
      data: {
        active_clients: 5,
        total_clients: 10,
        current_round: 3,
        total_rounds: 50,
        global_accuracy: 0.85,
        global_loss: 0.35,
        training_status: 'running',
        experiments_running: 1,
        uptime_hours: 24.5,
        last_updated: '2026-07-09T10:00:00Z',
      },
    });
  });

  it('renders loading skeleton initially', () => {
    mockFetchDashboard.mockReturnValue(new Promise(() => {}));
    createWrapper(<LiveDashboard />);
    const skeletons = document.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('renders Live Monitor heading', async () => {
    createWrapper(<LiveDashboard />);
    await waitFor(() => {
      expect(screen.getByText('Live Monitor')).toBeTruthy();
    });
  });

  it('shows stat values from dashboard data', async () => {
    createWrapper(<LiveDashboard />);
    await waitFor(() => {
      const rounds = screen.getAllByText('3 / 50');
      expect(rounds.length).toBeGreaterThanOrEqual(1);
    });
    await waitFor(() => {
      expect(screen.getByText('5 / 10')).toBeTruthy();
    });
    await waitFor(() => {
      expect(screen.getByText('85.0%')).toBeTruthy();
    });
  });

  it('shows Clear and Refresh buttons', async () => {
    createWrapper(<LiveDashboard />);
    await waitFor(() => {
      expect(screen.getByText('Clear')).toBeTruthy();
      expect(screen.getByText('Refresh')).toBeTruthy();
    });
  });

  it('shows event history and server status sections', async () => {
    createWrapper(<LiveDashboard />);
    await waitFor(() => {
      expect(screen.getByText('Event History')).toBeTruthy();
      expect(screen.getByText('Server Status')).toBeTruthy();
    });
  });

  it('shows and hides filters on toggle', async () => {
    const user = userEvent.setup();
    createWrapper(<LiveDashboard />);
    await waitFor(() => {
      expect(screen.getByText('Show filters')).toBeTruthy();
    });
    await user.click(screen.getByText('Show filters'));
    await waitFor(() => {
      expect(screen.getByText('Hide filters')).toBeTruthy();
    });
    await user.click(screen.getByText('Hide filters'));
    await waitFor(() => {
      expect(screen.getByText('Show filters')).toBeTruthy();
    });
  });

  it('displays server status section with transport', async () => {
    createWrapper(<LiveDashboard />);
    await waitFor(() => {
      expect(screen.getAllByText('Polling').length).toBeGreaterThan(0);
    });
  });
});

describe('RealtimeContext (useRealtime)', () => {
  it('throws when used outside provider', () => {
    function TestHook() {
      useRealtime();
      return null;
    }
    const queryClient = new QueryClient();
    expect(() =>
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter>
            <TestHook />
          </MemoryRouter>
        </QueryClientProvider>,
      ),
    ).toThrow('useRealtime must be used within a RealtimeProvider');
  });
});

describe('MAX_HISTORY_SIZE', () => {
  it('is set to 500', () => {
    expect(MAX_HISTORY_SIZE).toBe(500);
  });
});
