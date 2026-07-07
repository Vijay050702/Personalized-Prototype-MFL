import type { PrototypeListResponse, PrototypeResponse } from '../types';
import client from './axios';

export async function fetchPrototypes(): Promise<PrototypeListResponse> {
  const response = await client.get<PrototypeListResponse>('/api/v1/prototypes');
  return response.data;
}

export async function fetchPrototypeDetail(id: string): Promise<PrototypeResponse> {
  const response = await client.get<PrototypeResponse>(`/api/v1/prototypes/${encodeURIComponent(id)}`);
  return response.data;
}

export async function createPrototype(data: Partial<PrototypeResponse>): Promise<PrototypeResponse> {
  const response = await client.post<PrototypeResponse>('/api/v1/prototypes', data);
  return response.data;
}

export async function updatePrototype(id: string, data: Partial<PrototypeResponse>): Promise<PrototypeResponse> {
  const response = await client.put<PrototypeResponse>(`/api/v1/prototypes/${encodeURIComponent(id)}`, data);
  return response.data;
}

export async function deletePrototype(id: string): Promise<void> {
  await client.delete(`/api/v1/prototypes/${encodeURIComponent(id)}`);
}
