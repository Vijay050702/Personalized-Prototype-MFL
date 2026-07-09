import type { SettingsSummary, SettingsUpdateRequest } from '../types';
import client from './axios';

export async function fetchSettings(): Promise<SettingsSummary> {
  const response = await client.get<SettingsSummary>('/api/v1/settings');
  return response.data;
}

export async function updateSettings(data: SettingsUpdateRequest): Promise<SettingsSummary> {
  const response = await client.put<SettingsSummary>('/api/v1/settings', data);
  return response.data;
}
