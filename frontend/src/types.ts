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
