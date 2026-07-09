import type { RealtimeEvent, RealtimeEventCategory, RealtimeEventSeverity, RealtimeEventType, RealtimeEventPayload } from './types';

export type { RealtimeEvent, RealtimeEventCategory, RealtimeEventSeverity, RealtimeEventType, RealtimeEventPayload };

export const EVENT_CATEGORIES: RealtimeEventCategory[] = [
  'training', 'client', 'prototype', 'knowledge_transfer', 'evaluation', 'experiment', 'server',
];

export const EVENT_SEVERITIES: RealtimeEventSeverity[] = ['error', 'warning', 'info', 'success'];

let _counter = 0;

export function createEvent(
  type: RealtimeEventType,
  category: RealtimeEventCategory,
  severity: RealtimeEventSeverity,
  title: string,
  description: string,
  payload?: RealtimeEventPayload,
): RealtimeEvent {
  _counter += 1;
  return {
    id: `evt_${Date.now()}_${_counter}`,
    timestamp: Date.now(),
    type,
    category,
    severity,
    title,
    description,
    payload,
  };
}

export const EVENT_SEVERITY_ORDER: Record<RealtimeEventSeverity, number> = {
  error: 0,
  warning: 1,
  info: 2,
  success: 3,
};

export function severityFromStatus(status: string): RealtimeEventSeverity {
  const s = status.toLowerCase();
  if (s === 'error' || s === 'failed') return 'error';
  if (s === 'warning' || s === 'pending') return 'warning';
  if (s === 'completed' || s === 'success' || s === 'active') return 'success';
  return 'info';
}

export function filterEvents(
  events: RealtimeEvent[],
  opts: {
    categories?: RealtimeEventCategory[];
    severities?: RealtimeEventSeverity[];
    search?: string;
    limit?: number;
  },
): RealtimeEvent[] {
  let result = events;
  if (opts.categories && opts.categories.length > 0) {
    result = result.filter((e) => opts.categories!.includes(e.category));
  }
  if (opts.severities && opts.severities.length > 0) {
    result = result.filter((e) => opts.severities!.includes(e.severity));
  }
  if (opts.search) {
    const q = opts.search.toLowerCase();
    result = result.filter((e) => e.title.toLowerCase().includes(q) || e.description.toLowerCase().includes(q));
  }
  if (opts.limit && opts.limit > 0) {
    result = result.slice(0, opts.limit);
  }
  return result;
}

export function getCategoryLabel(category: RealtimeEventCategory): string {
  const labels: Record<RealtimeEventCategory, string> = {
    training: 'Training',
    client: 'Clients',
    prototype: 'Prototypes',
    knowledge_transfer: 'Knowledge Transfer',
    evaluation: 'Evaluation',
    experiment: 'Experiments',
    server: 'Server',
  };
  return labels[category];
}
