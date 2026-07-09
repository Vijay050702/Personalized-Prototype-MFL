import { createContext, useState, useEffect, useCallback, useRef, type ReactNode } from 'react';
import { useQueryClient } from '@tanstack/react-query';

import type { ConnectionStatus, RealtimeContextValue, RealtimeEvent, TransportType } from './types';
import { MAX_HISTORY_SIZE } from './types';
import { createEvent, severityFromStatus } from './events';
import { createConnectionManager } from './connection';

export const RealtimeContext = createContext<RealtimeContextValue | null>(null);

const INVALIDATION_MAP: Record<string, string[]> = {
  training: ['training', 'dashboard'],
  client: ['clients', 'dashboard'],
  prototype: ['prototypes'],
  knowledge_transfer: ['knowledgeTransfer', 'knowledge-transfer'],
  evaluation: ['evaluation'],
  experiment: ['experiments'],
  server: ['dashboard'],
};

interface RealtimeProviderProps {
  children: ReactNode;
  pollingInterval?: number;
}

export const RealtimeProvider = ({ children, pollingInterval = 5000 }: RealtimeProviderProps) => {
  const [transport, setTransport] = useState<TransportType>('none');
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const [events, setEvents] = useState<RealtimeEvent[]>([]);
  const [transportError, setTransportError] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const pollingRef = useRef<ReturnType<typeof setInterval>>();
  const probeDone = useRef(false);

  const lastEventTimestamp = events.length > 0 ? events[0].timestamp : null;

  const addEvent = useCallback((event: RealtimeEvent) => {
    setEvents((prev) => {
      const updated = [event, ...prev];
      if (updated.length > MAX_HISTORY_SIZE) {
        return updated.slice(0, MAX_HISTORY_SIZE);
      }
      return updated;
    });
  }, []);

  const clearHistory = useCallback(() => {
    setEvents([]);
  }, []);

  const removeEvent = useCallback((id: string) => {
    setEvents((prev) => prev.filter((e) => e.id !== id));
  }, []);

  const refetchAll = useCallback(() => {
    const keys = new Set(Object.values(INVALIDATION_MAP).flat());
    keys.forEach((key) => {
      queryClient.invalidateQueries({ queryKey: [key] });
    });
  }, [queryClient]);

  const emit = useCallback(
    (type: RealtimeEvent['type'], category: RealtimeEvent['category'], severity: RealtimeEvent['severity'], title: string, description: string) => {
      const event = createEvent(type, category, severity, title, description);
      addEvent(event);
      const keys = INVALIDATION_MAP[category];
      if (keys) {
        keys.forEach((key) => {
          queryClient.invalidateQueries({ queryKey: [key] });
        });
      }
    },
    [addEvent, queryClient],
  );

  useEffect(() => {
    if (probeDone.current) return;
    probeDone.current = true;

    const manager = createConnectionManager();

    manager.connect().then((result) => {
      setTransport(result.transport);
      setTransportError(result.error);

      if (result.transport === 'polling') {
        setConnectionStatus('polling');
        addEvent(
          createEvent('server:health_change', 'server', 'info', 'Transport: Polling', 'Backend does not support WebSocket or SSE. Using React Query polling.', {
            server: { last_updated: new Date().toISOString() },
          }),
        );
      } else if (result.transport === 'none') {
        setConnectionStatus('offline');
        setTransportError(result.error);
        addEvent(
          createEvent('server:health_change', 'server', 'error', 'Backend Offline', result.error ?? 'Backend is unreachable.', {
            server: { error_message: result.error ?? 'Backend is unreachable.' },
          }),
        );
      } else {
        setConnectionStatus('connected');
        addEvent(
          createEvent('server:health_change', 'server', 'success', `Transport: ${result.transport}`, `Connected via ${result.transport}.`, {
            server: { last_updated: new Date().toISOString() },
          }),
        );
      }
    });

    return () => {
      manager.disconnect();
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [addEvent]);

  const currentTransportLabel =
    transport === 'websocket' ? 'WebSocket' :
    transport === 'sse' ? 'Server-Sent Events' :
    transport === 'polling' ? 'Polling' :
    transport === 'none' ? 'None' : 'Unknown';

  const value: RealtimeContextValue = {
    transport,
    connectionStatus,
    events,
    isLive: connectionStatus === 'connected' || connectionStatus === 'polling',
    lastEventTimestamp,
    transportError,
    clearHistory,
    removeEvent,
    refetchAll,
    currentTransportLabel,
  };

  return <RealtimeContext.Provider value={value}>{children}</RealtimeContext.Provider>;
};
