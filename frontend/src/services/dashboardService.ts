/**
 * @deprecated This mock service is replaced by src/api/dashboard.ts
 * Use fetchDashboard() from '../api/dashboard' instead.
 */

import type { ActivityLog, SystemMetric } from '../types';

/** @deprecated Use fetchDashboard from src/api/dashboard instead */
export const getDashboardStats = async (): Promise<SystemMetric[]> => {
  throw new Error(
    'Mock service is deprecated. Use fetchDashboard() from src/api/dashboard instead.',
  );
};

/** @deprecated Activity log endpoint is not yet available from the backend */
export const getRecentActivity = async (): Promise<ActivityLog[]> => {
  throw new Error(
    'Mock service is deprecated. Activity log is not yet available from the backend.',
  );
};
