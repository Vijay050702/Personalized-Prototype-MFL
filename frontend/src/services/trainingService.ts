/**
 * @deprecated Migrated to api/training.ts. This file will be removed.
 * Use fetchTrainingStatus() from '../api/training' instead.
 */
import { TrainingRound } from '../types';

export const getTrainingHistory = async (): Promise<TrainingRound[]> => {
  throw new Error('Mock service deprecated. Use api/training instead.');
};

export const getConvergenceData = async () => {
  throw new Error('Mock service deprecated. Use api/training instead.');
};
