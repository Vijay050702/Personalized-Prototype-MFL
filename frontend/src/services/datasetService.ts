/**
 * @deprecated Use src/api/datasets.ts instead.
 *   - fetchDatasets()     -> GET  /api/v1/datasets
 *   - fetchDatasetDetail() -> GET  /api/v1/datasets/{name}
 *   - registerDataset()   -> POST /api/v1/datasets/register
 *   - downloadDataset()   -> POST /api/v1/datasets/download
 *   - preprocessDataset() -> POST /api/v1/datasets/preprocess
 *   - partitionDataset()  -> POST /api/v1/datasets/partition
 *   - simulateMissingModality() -> POST /api/v1/datasets/missing-modality
 *   - deleteDataset()     -> DELETE /api/v1/datasets/{name}
 */
export const getDatasets = async (): Promise<never> => {
  throw new Error(
    'datasetService is deprecated. Import from src/api/datasets.ts instead.',
  );
};
