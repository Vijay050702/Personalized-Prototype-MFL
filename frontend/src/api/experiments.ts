import type { ExperimentListResponse } from '../types';
import client from './axios';

export async function fetchExperiments(): Promise<ExperimentListResponse> {
  const response = await client.get<ExperimentListResponse>('/api/v1/experiments');
  return response.data;
}
