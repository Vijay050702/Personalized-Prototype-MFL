export type Status = 'active' | 'inactive' | 'error' | 'pending';

export interface Client {
  id: string;
  name: string;
  status: Status;
  lastSeen: string;
  deviceInfo: string;
  progress: number;
  location: string;
  accuracy: number;
}

export interface Dataset {
  id: string;
  name: string;
  source: string;
  samples: number;
  features: number;
  privacyLevel: 'low' | 'medium' | 'high';
  lastUpdated: string;
}

export interface TrainingRound {
  id: string;
  round: number;
  startTime: string;
  duration: string;
  accuracy: number;
  loss: number;
  status: 'completed' | 'running' | 'failed';
  participants: number;
}

export interface SystemMetric {
  label: string;
  value: string | number;
  change: number;
  trend: 'up' | 'down' | 'neutral';
}

export interface ActivityLog {
  id: string;
  timestamp: string;
  type: 'info' | 'warning' | 'error' | 'success';
  message: string;
  user?: string;
}

export interface DashboardResponse {
  active_clients: number;
  total_clients: number;
  current_round: number;
  total_rounds: number;
  global_accuracy: number;
  global_loss: number;
  training_status: string;
  experiments_running: number;
  uptime_hours: number;
  last_updated: string;
}

export interface DashboardSummary {
  status: string;
  message: string;
  data: DashboardResponse;
}
