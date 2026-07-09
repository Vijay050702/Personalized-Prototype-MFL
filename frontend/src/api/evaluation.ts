import type {
  EvaluationSummary,
  ExperimentListResponse,
} from '../types';
import client from './axios';

export async function fetchEvaluation(): Promise<EvaluationSummary> {
  const response = await client.get<EvaluationSummary>('/api/v1/evaluation');
  return response.data;
}

export async function fetchExperiments(): Promise<ExperimentListResponse> {
  const response = await client.get<ExperimentListResponse>('/api/v1/experiments');
  return response.data;
}
