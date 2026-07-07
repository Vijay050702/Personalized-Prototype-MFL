import type {
  KnowledgeTransferListResponse,
  KnowledgeTransferResponse,
  KnowledgeTransferStatisticsResponse,
  KnowledgeTransferStartRequest,
  KnowledgeTransferHistoryResponse,
} from '../types';
import client from './axios';

export async function fetchKnowledgeTransfers(): Promise<KnowledgeTransferListResponse> {
  const response = await client.get<KnowledgeTransferListResponse>('/api/v1/knowledge-transfer');
  return response.data;
}

export async function fetchKnowledgeTransferDetail(transferId: string): Promise<KnowledgeTransferResponse> {
  const response = await client.get<KnowledgeTransferResponse>(
    `/api/v1/knowledge-transfer/${encodeURIComponent(transferId)}`,
  );
  return response.data;
}

export async function fetchKnowledgeTransferStatistics(): Promise<KnowledgeTransferStatisticsResponse> {
  const response = await client.get<KnowledgeTransferStatisticsResponse>('/api/v1/knowledge-transfer/statistics');
  return response.data;
}

export async function startKnowledgeTransfer(data: KnowledgeTransferStartRequest): Promise<{ status: string; message: string }> {
  const response = await client.post<{ status: string; message: string }>('/api/v1/knowledge-transfer/start', data);
  return response.data;
}

export async function stopKnowledgeTransfer(): Promise<{ status: string; message: string }> {
  const response = await client.post<{ status: string; message: string }>('/api/v1/knowledge-transfer/stop');
  return response.data;
}

export async function fetchKnowledgeTransferHistory(): Promise<KnowledgeTransferHistoryResponse> {
  const response = await client.get<KnowledgeTransferHistoryResponse>('/api/v1/knowledge-transfer/history');
  return response.data;
}
