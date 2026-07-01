import { Client } from '../types';

export const getClients = async (): Promise<Client[]> => {
  return [
    { id: 'c1', name: 'Alpha-01', status: 'active', lastSeen: 'Just now', deviceInfo: 'NVIDIA Jetson Nano', progress: 85, location: 'San Francisco, CA', accuracy: 0.92 },
    { id: 'c2', name: 'Beta-04', status: 'active', lastSeen: '2m ago', deviceInfo: 'Raspberry Pi 4', progress: 100, location: 'London, UK', accuracy: 0.88 },
    { id: 'c3', name: 'Gamma-09', status: 'inactive', lastSeen: '4h ago', deviceInfo: 'AWS EC2 t3.medium', progress: 0, location: 'Tokyo, JP', accuracy: 0.95 },
    { id: 'c4', name: 'Delta-12', status: 'error', lastSeen: '15m ago', deviceInfo: 'Edge TPu', progress: 45, location: 'Berlin, DE', accuracy: 0.74 },
    { id: 'c5', name: 'Epsilon-02', status: 'active', lastSeen: 'Just now', deviceInfo: 'iPhone 13 Pro', progress: 92, location: 'New York, NY', accuracy: 0.91 },
  ];
};
