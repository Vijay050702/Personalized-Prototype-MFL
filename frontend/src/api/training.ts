import type {
  TrainingStatusSummary,
  TrainingConfigSummary,
  TrainingControlResponse,
  TrainingConfigData,
} from '../types';
import client from './axios';

export async function fetchTrainingStatus(): Promise<TrainingStatusSummary> {
  const response = await client.get<TrainingStatusSummary>('/api/v1/training/status');
  return response.data;
}

export async function fetchTrainingConfig(): Promise<TrainingConfigSummary> {
  const response = await client.get<TrainingConfigSummary>('/api/v1/training/config');
  return response.data;
}

export async function startTraining(): Promise<TrainingControlResponse> {
  const response = await client.post<TrainingControlResponse>('/api/v1/training/start');
  return response.data;
}

export async function pauseTraining(): Promise<TrainingControlResponse> {
  const response = await client.post<TrainingControlResponse>('/api/v1/training/pause');
  return response.data;
}

export async function resumeTraining(): Promise<TrainingControlResponse> {
  const response = await client.post<TrainingControlResponse>('/api/v1/training/resume');
  return response.data;
}

export async function stopTraining(): Promise<TrainingControlResponse> {
  const response = await client.post<TrainingControlResponse>('/api/v1/training/stop');
  return response.data;
}

export async function saveCheckpoint(): Promise<TrainingControlResponse> {
  const response = await client.post<TrainingControlResponse>('/api/v1/training/checkpoint');
  return response.data;
}

export async function updateTrainingConfig(data: TrainingConfigData): Promise<TrainingConfigSummary> {
  const response = await client.put<TrainingConfigSummary>('/api/v1/training/config', data);
  return response.data;
}
