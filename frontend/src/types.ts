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

export interface ClientResponse {
  id: string;
  name: string;
  status: string;
  accuracy: number;
  loss: number;
  data_size: number;
  last_round: number;
  device: string;
  region: string;
  joined_at: string;
  last_communication: string;
}

export interface ClientListResponse {
  status: string;
  message: string;
  data: ClientResponse[];
  total: number;
}

export interface DatasetResponse {
  id: string;
  name: string;
  type: string;
  modality: string;
  size_mb: number;
  samples: number;
  classes: number;
  client_id: string;
  distribution: string;
}

export interface DatasetListResponse {
  status: string;
  message: string;
  data: DatasetResponse[];
  total: number;
}

export interface DatasetMetadataResponse {
  dataset_name: string;
  modalities: string[];
  classes: string[];
  num_classes: number;
  input_shapes: Record<string, number[]>;
  num_samples: number;
  client_count: number;
  missing_modality_ratio: number;
  download_status: string;
  preprocessing_status: string;
  partition_status: string;
}

export interface DatasetDetailResponse {
  status: string;
  message: string;
  data: DatasetMetadataResponse;
}

export interface DatasetRegistrationRequest {
  name: string;
  modality?: string;
  modalities?: string[];
  path?: string;
}

export interface DownloadRequest {
  dataset_name: string;
  force?: boolean;
}

export interface PreprocessRequest {
  dataset_name: string;
  force?: boolean;
}

export interface PartitionRequest {
  dataset_name: string;
  strategy?: string;
  num_clients?: number;
  alpha?: number;
  min_samples?: number;
  seed?: number;
  balanced?: boolean;
  shards_per_client?: number;
}

export interface PartitionResponse {
  status: string;
  dataset_name: string;
  strategy: string;
  num_clients: number;
  client_distributions: Record<string, unknown>[];
  seed: number;
}

export interface MissingModalityRequest {
  dataset_name: string;
  strategy?: string;
  missing_ratio?: number;
  modalities?: string[];
  seed?: number;
}

export interface OperationResponse {
  status: string;
  message: string;
  dataset_name: string;
  operation: string;
}

export interface ValidationResponse {
  status: string;
  dataset_name: string;
  is_valid: boolean;
  errors: string[];
  warnings: string[];
  class_distribution: Record<string, unknown> | null;
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
