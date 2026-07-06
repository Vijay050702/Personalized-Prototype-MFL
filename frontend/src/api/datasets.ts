import type {
  DatasetListResponse,
  DatasetDetailResponse,
  DatasetRegistrationRequest,
  DownloadRequest,
  PreprocessRequest,
  PartitionRequest,
  PartitionResponse,
  MissingModalityRequest,
  OperationResponse,
  ValidationResponse,
} from '../types';
import client from './axios';

export async function fetchDatasets(): Promise<DatasetListResponse> {
  const response = await client.get<DatasetListResponse>('/api/v1/datasets');
  return response.data;
}

export async function fetchDatasetDetail(name: string): Promise<DatasetDetailResponse> {
  const response = await client.get<DatasetDetailResponse>(`/api/v1/datasets/${encodeURIComponent(name)}`);
  return response.data;
}

export async function registerDataset(data: DatasetRegistrationRequest): Promise<OperationResponse> {
  const response = await client.post<OperationResponse>('/api/v1/datasets/register', data);
  return response.data;
}

export async function downloadDataset(data: DownloadRequest): Promise<OperationResponse> {
  const response = await client.post<OperationResponse>('/api/v1/datasets/download', data);
  return response.data;
}

export async function preprocessDataset(data: PreprocessRequest): Promise<OperationResponse> {
  const response = await client.post<OperationResponse>('/api/v1/datasets/preprocess', data);
  return response.data;
}

export async function partitionDataset(data: PartitionRequest): Promise<PartitionResponse> {
  const response = await client.post<PartitionResponse>('/api/v1/datasets/partition', data);
  return response.data;
}

export async function simulateMissingModality(data: MissingModalityRequest): Promise<OperationResponse> {
  const response = await client.post<OperationResponse>('/api/v1/datasets/missing-modality', data);
  return response.data;
}

export async function deleteDataset(name: string): Promise<OperationResponse> {
  const response = await client.delete<OperationResponse>(`/api/v1/datasets/${encodeURIComponent(name)}`);
  return response.data;
}

export async function validateDataset(name: string): Promise<ValidationResponse> {
  const response = await client.get<ValidationResponse>(`/api/v1/datasets/${encodeURIComponent(name)}/validate`);
  return response.data;
}
