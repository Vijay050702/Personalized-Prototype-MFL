import type { DashboardSummary } from '../types';
import client from './axios';

export async function fetchDashboard(): Promise<DashboardSummary> {
  const response = await client.get<DashboardSummary>('/api/v1/dashboard');
  return response.data;
}
