import type { ConnectionStatus, TransportType } from './types';
import client from '../api/axios';

export interface TransportProbeResult {
  transport: TransportType;
  error: string | null;
}

function wsUrl(): string {
  const base = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
  const wsBase = base.replace(/^http/, 'ws');
  return `${wsBase}/ws/health`;
}

async function probeWebSocket(timeoutMs = 3000): Promise<boolean> {
  return new Promise((resolve) => {
    let resolved = false;
    try {
      const url = wsUrl();
      const ws = new WebSocket(url);
      const timer = setTimeout(() => {
        if (!resolved) {
          resolved = true;
          ws.close();
          resolve(false);
        }
      }, timeoutMs);
      ws.onopen = () => {
        if (!resolved) {
          resolved = true;
          clearTimeout(timer);
          ws.close();
          resolve(true);
        }
      };
      ws.onerror = () => {
        if (!resolved) {
          resolved = true;
          clearTimeout(timer);
          resolve(false);
        }
      };
    } catch {
      if (!resolved) {
        resolved = true;
        resolve(false);
      }
    }
  });
}

async function probeSSE(timeoutMs = 3000): Promise<boolean> {
  return new Promise((resolve) => {
    let resolved = false;
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => {
        if (!resolved) {
          resolved = true;
          controller.abort();
          resolve(false);
        }
      }, timeoutMs);
      fetch('/api/v1/events/stream', {
        signal: controller.signal,
        headers: { Accept: 'text/event-stream' },
      })
        .then((res) => {
          if (!resolved) {
            resolved = true;
            clearTimeout(timer);
            controller.abort();
            resolve(res.ok && res.headers.get('content-type')?.includes('text/event-stream') === true);
          }
        })
        .catch(() => {
          if (!resolved) {
            resolved = true;
            clearTimeout(timer);
            resolve(false);
          }
        });
    } catch {
      if (!resolved) {
        resolved = true;
        resolve(false);
      }
    }
  });
}

async function probePolling(): Promise<boolean> {
  try {
    const res = await client.get('/api/v1/dashboard', { timeout: 5000 });
    return res.status === 200;
  } catch {
    return false;
  }
}

export async function detectTransport(): Promise<TransportProbeResult> {
  const wsOk = await probeWebSocket();
  if (wsOk) return { transport: 'websocket', error: null };

  const sseOk = await probeSSE();
  if (sseOk) return { transport: 'sse', error: null };

  const pollingOk = await probePolling();
  if (pollingOk) return { transport: 'polling', error: null };

  return { transport: 'none', error: 'No transport available. Backend is unreachable.' };
}

export interface ConnectionManager {
  status: ConnectionStatus;
  transport: TransportType;
  error: string | null;
  connect: () => Promise<TransportProbeResult>;
  disconnect: () => void;
}

export function createConnectionManager(): ConnectionManager {
  return {
    status: 'disconnected',
    transport: 'none',
    error: null,
    async connect() {
      const result = await detectTransport();
      return result;
    },
    disconnect() {
      // No-op for polling mode
    },
  };
}
