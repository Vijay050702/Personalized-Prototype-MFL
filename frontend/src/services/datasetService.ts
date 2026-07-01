import { Dataset } from '../types';

export const getDatasets = async (): Promise<Dataset[]> => {
  return [
    { id: 'd1', name: 'User-Behavior-Logs', source: 'Edge Nodes', samples: 1250000, features: 42, privacyLevel: 'high', lastUpdated: '2024-03-20' },
    { id: 'd2', name: 'Sensor-Telemetry', source: 'IoT Fleet', samples: 840000, features: 12, privacyLevel: 'medium', lastUpdated: '2024-03-19' },
    { id: 'd3', name: 'Network-Traffic', source: 'Core Routers', samples: 2100000, features: 128, privacyLevel: 'high', lastUpdated: '2024-03-15' },
    { id: 'd4', name: 'Image-Validation', source: 'Mobile App', samples: 120000, features: 2048, privacyLevel: 'low', lastUpdated: '2024-03-10' },
  ];
};
