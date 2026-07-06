import type { ClientListResponse, ClientResponse } from '../types';
import client from './axios';

export async function fetchClients(): Promise<ClientListResponse> {
  const response = await client.get<ClientListResponse>('/api/v1/clients');
  return response.data;
}

export async function fetchClientDetail(id: string): Promise<ClientResponse> {
  const response = await client.get<ClientResponse>(`/api/v1/clients/${encodeURIComponent(id)}`);
  return response.data;
}

export async function createClient(data: Partial<ClientResponse>): Promise<ClientResponse> {
  const response = await client.post<ClientResponse>('/api/v1/clients', data);
  return response.data;
}

export async function updateClient(id: string, data: Partial<ClientResponse>): Promise<ClientResponse> {
  const response = await client.put<ClientResponse>(`/api/v1/clients/${encodeURIComponent(id)}`, data);
  return response.data;
}

export async function deleteClient(id: string): Promise<void> {
  await client.delete(`/api/v1/clients/${encodeURIComponent(id)}`);
}
