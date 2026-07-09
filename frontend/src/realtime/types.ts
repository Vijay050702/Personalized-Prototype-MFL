export type TransportType = 'websocket' | 'sse' | 'polling' | 'none';

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'reconnecting' | 'polling' | 'offline';

export type RealtimeEventCategory =
  | 'training'
  | 'client'
  | 'prototype'
  | 'knowledge_transfer'
  | 'evaluation'
  | 'experiment'
  | 'server';

export type RealtimeEventType =
  | 'training:round_start'
  | 'training:round_end'
  | 'training:status_change'
  | 'training:metric_update'
  | 'client:connected'
  | 'client:disconnected'
  | 'client:metric_update'
  | 'prototype:created'
  | 'prototype:updated'
  | 'prototype:drift_detected'
  | 'knowledge_transfer:started'
  | 'knowledge_transfer:completed'
  | 'knowledge_transfer:failed'
  | 'evaluation:started'
  | 'evaluation:completed'
  | 'evaluation:metric_update'
  | 'experiment:started'
  | 'experiment:completed'
  | 'experiment:failed'
  | 'experiment:progress'
  | 'server:health_change'
  | 'server:error';

export type RealtimeEventSeverity = 'info' | 'success' | 'warning' | 'error';

export interface RealtimeEventPayload {
  training?: {
    current_round?: number;
    total_rounds?: number;
    status?: string;
    accuracy?: number;
    loss?: number;
    round_duration_ms?: number;
    clients_participating?: number;
  };
  client?: {
    client_id?: string;
    name?: string;
    status?: string;
    accuracy?: number;
    loss?: number;
    total_clients?: number;
    active_clients?: number;
  };
  prototype?: {
    prototype_id?: string;
    modality?: string;
    client_id?: string;
    dimension?: number;
    quality_score?: number;
  };
  knowledge_transfer?: {
    transfer_id?: string;
    source_client?: string;
    target_client?: string;
    source_modality?: string;
    target_modality?: string;
    similarity_score?: number;
    transfer_loss?: number;
    status?: string;
  };
  evaluation?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    auc_roc?: number;
    client_id?: string;
    round?: number;
    samples_evaluated?: number;
  };
  experiment?: {
    experiment_id?: string;
    name?: string;
    status?: string;
    algorithm?: string;
    current_round?: number;
    total_rounds?: number;
    best_accuracy?: number;
  };
  server?: {
    uptime_hours?: number;
    experiments_running?: number;
    training_status?: string;
    last_updated?: string;
    error_message?: string;
  };
}

export interface RealtimeEvent {
  id: string;
  timestamp: number;
  type: RealtimeEventType;
  category: RealtimeEventCategory;
  severity: RealtimeEventSeverity;
  title: string;
  description: string;
  payload?: RealtimeEventPayload;
}

export interface RealtimeState {
  transport: TransportType;
  connectionStatus: ConnectionStatus;
  events: RealtimeEvent[];
  isLive: boolean;
  lastEventTimestamp: number | null;
  transportError: string | null;
}

export interface RealtimeContextValue extends RealtimeState {
  clearHistory: () => void;
  removeEvent: (id: string) => void;
  refetchAll: () => void;
  currentTransportLabel: string;
}

export interface LiveDashboardData {
  currentRound: number;
  totalRounds: number;
  trainingStatus: string;
  activeClients: number;
  totalClients: number;
  globalAccuracy: number;
  globalLoss: number;
  experimentsRunning: number;
  uptimeHours: number;
  prototypeCount: number;
  knowledgeTransferCount: number;
  evaluationRoundsCompleted: number;
  lastUpdated: string | null;
  communicationRate: number;
}

export const TRANSPORT_LABELS: Record<TransportType, string> = {
  websocket: 'WebSocket',
  sse: 'Server-Sent Events',
  polling: 'Polling',
  none: 'None',
};

export const MAX_HISTORY_SIZE = 500;
