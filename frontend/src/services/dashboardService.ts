import { SystemMetric, ActivityLog } from '../types';

export const getDashboardStats = async (): Promise<SystemMetric[]> => {
  return [
    { label: 'Active Clients', value: '1,284', change: 12, trend: 'up' },
    { label: 'Avg. Accuracy', value: '94.2%', change: 2.1, trend: 'up' },
    { label: 'Data Samples', value: '2.4M', change: 5.4, trend: 'up' },
    { label: 'Training Rounds', value: '48', change: 0, trend: 'neutral' },
  ];
};

export const getRecentActivity = async (): Promise<ActivityLog[]> => {
  return [
    { id: '1', timestamp: '2 mins ago', type: 'success', message: 'Round 48 completed successfully' },
    { id: '2', timestamp: '15 mins ago', type: 'info', message: 'New dataset "Edge-Logs-V4" registered' },
    { id: '3', timestamp: '1 hour ago', type: 'warning', message: 'Node "Alpha-7" reported high latency' },
    { id: '4', timestamp: '3 hours ago', type: 'success', message: 'Global model updated to v2.1.0' },
  ];
};
