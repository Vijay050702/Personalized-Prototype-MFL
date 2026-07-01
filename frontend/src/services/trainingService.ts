import { TrainingRound } from '../types';

export const getTrainingHistory = async (): Promise<TrainingRound[]> => {
  return [
    { id: 'r48', round: 48, startTime: '2024-03-21 14:00', duration: '12m 4s', accuracy: 0.942, loss: 0.045, status: 'completed', participants: 1150 },
    { id: 'r47', round: 47, startTime: '2024-03-21 12:00', duration: '11m 58s', accuracy: 0.938, loss: 0.048, status: 'completed', participants: 1142 },
    { id: 'r46', round: 46, startTime: '2024-03-21 10:00', duration: '14m 20s', accuracy: 0.931, loss: 0.052, status: 'completed', participants: 1080 },
    { id: 'r45', round: 45, startTime: '2024-03-21 08:00', duration: '12m 15s', accuracy: 0.925, loss: 0.058, status: 'failed', participants: 420 },
    { id: 'r44', round: 44, startTime: '2024-03-21 06:00', duration: '10m 50s', accuracy: 0.924, loss: 0.059, status: 'completed', participants: 1205 },
  ];
};

export const getConvergenceData = async () => {
  return [
    { round: 40, accuracy: 0.88, loss: 0.12 },
    { round: 41, accuracy: 0.89, loss: 0.11 },
    { round: 42, accuracy: 0.90, loss: 0.09 },
    { round: 43, accuracy: 0.91, loss: 0.08 },
    { round: 44, accuracy: 0.92, loss: 0.07 },
    { round: 45, accuracy: 0.92, loss: 0.075 },
    { round: 46, accuracy: 0.93, loss: 0.06 },
    { round: 47, accuracy: 0.94, loss: 0.05 },
    { round: 48, accuracy: 0.942, loss: 0.045 },
  ];
};
