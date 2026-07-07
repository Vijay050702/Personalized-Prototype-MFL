import type { SimilarityAnalysis, SimilarityListResponse, SimilarityMatrixResponse, SimilarityStatisticsResponse, SimilarityHistoryResponse } from '../types';
import client from './axios';

export async function fetchSimilarityAnalyses(): Promise<SimilarityListResponse> {
  const response = await client.get<SimilarityListResponse>('/api/v1/similarity');
  return response.data;
}

export async function fetchSimilarityDetail(analysisId: string): Promise<SimilarityAnalysis> {
  const response = await client.get<SimilarityAnalysis>(`/api/v1/similarity/${encodeURIComponent(analysisId)}`);
  return response.data;
}

export async function fetchSimilarityMatrix(): Promise<SimilarityMatrixResponse> {
  const response = await client.get<SimilarityMatrixResponse>('/api/v1/similarity/matrix');
  return response.data;
}

export async function fetchSimilarityStatistics(): Promise<SimilarityStatisticsResponse> {
  const response = await client.get<SimilarityStatisticsResponse>('/api/v1/similarity/statistics');
  return response.data;
}

export async function fetchSimilarityHistory(): Promise<SimilarityHistoryResponse> {
  const response = await client.get<SimilarityHistoryResponse>('/api/v1/similarity/history');
  return response.data;
}
