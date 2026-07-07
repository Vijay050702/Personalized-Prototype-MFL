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

export interface TrainingStatusData {
  status: string;
  current_round: number;
  total_rounds: number;
  epochs_completed: number;
  total_epochs: number;
  current_loss: number;
  current_accuracy: number;
  learning_rate: number;
  clients_participating: number;
  aggregation_algorithm: string;
  time_elapsed_seconds: number;
  estimated_time_remaining: number;
}

export interface TrainingStatusSummary {
  status: string;
  message: string;
  data: TrainingStatusData;
}

export interface TrainingConfigData {
  dataset: string;
  client_count: number;
  communication_rounds: number;
  local_epochs: number;
  batch_size: number;
  learning_rate: number;
  optimizer: string;
  scheduler: string;
  aggregation_strategy: string;
  knowledge_transfer_enabled: boolean;
  personalization_enabled: boolean;
}

export interface TrainingConfigSummary {
  status: string;
  message: string;
  data: TrainingConfigData;
}

export interface TrainingControlResponse {
  status: string;
  message: string;
}

export interface PrototypeResponse {
  id: string;
  modality: string;
  dimension: number;
  cluster_id: number;
  quality_score: number;
  client_id: string;
  created_round: number;
}

export interface PrototypeListResponse {
  status: string;
  message: string;
  data: PrototypeResponse[];
  total: number;
}

export interface KnowledgeTransferResponse {
  transfer_id: string;
  source_client: string;
  target_client: string;
  source_prototype: string;
  target_prototype: string;
  source_modality: string;
  target_modality: string;
  transfer_strategy: string;
  cross_modal_mapping: string;
  alignment_method: string;
  transfer_loss: number;
  similarity_score: number;
  confidence: number;
  communication_round: number;
  transfer_status: string;
  execution_time: number;
  created_at: string;
}

export interface KnowledgeTransferListResponse {
  status: string;
  message: string;
  data: KnowledgeTransferResponse[];
  total: number;
}

export interface KnowledgeTransferStatistics {
  total_transfers: number;
  successful_transfers: number;
  failed_transfers: number;
  average_similarity: number;
  average_confidence: number;
  average_transfer_loss: number;
  average_execution_time: number;
  communication_efficiency: number;
}

export interface KnowledgeTransferStatisticsResponse {
  status: string;
  message: string;
  data: KnowledgeTransferStatistics;
}

export interface KnowledgeTransferStartRequest {
  source_client: string;
  target_client: string;
  source_modality: string;
  target_modality: string;
  transfer_strategy: string;
}

export interface KnowledgeTransferHistoryResponse {
  status: string;
  message: string;
  data: KnowledgeTransferResponse[];
  total: number;
}
