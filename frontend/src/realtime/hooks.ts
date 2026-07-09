import { useContext, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';

import { RealtimeContext } from './provider';
import type { RealtimeContextValue, RealtimeEventCategory, RealtimeEventSeverity, LiveDashboardData } from './types';
import { filterEvents } from './events';
import { fetchDashboard } from '../api/dashboard';
import type { DashboardResponse } from '../types';

export function useRealtime(): RealtimeContextValue {
  const ctx = useContext(RealtimeContext);
  if (!ctx) {
    throw new Error('useRealtime must be used within a RealtimeProvider');
  }
  return ctx;
}

export function useConnectionStatus() {
  const { connectionStatus, transport, currentTransportLabel, isLive } = useRealtime();
  return { connectionStatus, transport, currentTransportLabel, isLive };
}

export function useEventHistory(opts?: {
  categories?: RealtimeEventCategory[];
  severities?: RealtimeEventSeverity[];
  search?: string;
  limit?: number;
}) {
  const { events } = useRealtime();
  return useMemo(() => filterEvents(events, {
    categories: opts?.categories,
    severities: opts?.severities,
    search: opts?.search,
    limit: opts?.limit,
  }), [events, opts?.categories, opts?.severities, opts?.search, opts?.limit]);
}

export function useRealtimeDashboard(): {
  data: LiveDashboardData | null;
  isLoading: boolean;
  isError: boolean;
} {
  const { data: summary, isLoading, isError } = useQuery({
    queryKey: ['dashboard'],
    queryFn: fetchDashboard,
    refetchInterval: 5000,
    staleTime: 4000,
    retry: 2,
  });

  const d: DashboardResponse | null = summary?.data ?? null;

  const data: LiveDashboardData | null = d
    ? {
        currentRound: d.current_round,
        totalRounds: d.total_rounds,
        trainingStatus: d.training_status,
        activeClients: d.active_clients,
        totalClients: d.total_clients,
        globalAccuracy: d.global_accuracy,
        globalLoss: d.global_loss,
        experimentsRunning: d.experiments_running,
        uptimeHours: d.uptime_hours,
        prototypeCount: 0,
        knowledgeTransferCount: 0,
        evaluationRoundsCompleted: d.current_round,
        lastUpdated: d.last_updated,
        communicationRate: d.total_rounds > 0 ? parseFloat((d.current_round / Math.max(d.uptime_hours, 1)).toFixed(2)) : 0,
      }
    : null;

  return { data, isLoading, isError };
}
