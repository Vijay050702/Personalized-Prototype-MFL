# PP-MFL Backend

## Project Structure

```
backend/
├── app/
│   ├── core/           # Config, logging (Phase 1)
│   ├── datasets/       # Dataset adapters, registry, cache (Phase 2A)
│   ├── data/           # Multimodal data interface layer (Phase 2B)
│   │   ├── modality.py              # Modality enum & mask utilities
│   │   ├── multimodal_sample.py     # Single multimodal sample
│   │   ├── multimodal_batch.py      # Batched multimodal sample
│   │   ├── multimodal_dataset.py    # PyTorch Dataset
│   │   ├── datamodule.py            # Lightning-style DataModule
│   │   ├── dataloader.py            # Deterministic DataLoader wrapper
│   │   ├── collate.py               # Collation & padding utilities
│   │   ├── transforms.py            # Modality-specific transforms
│   │   ├── validation.py            # Input validation & exceptions
│   │   ├── statistics.py            # Dataset statistics computation
│   │   ├── factory.py               # DataFactory with caching
│   │   └── dataloaders/             # Modality-specific loaders
│   ├── routers/        # FastAPI routers
│   ├── schemas/        # Pydantic schemas
│   └── services/       # Business logic
├── tests/              # Test suite (249 tests, 1 skipped)
└── sample_data/        # UCI-HAR sample dataset
```

## Phase 2B: Multimodal Data Interface Layer

The `app/data/` package provides the standardized PyTorch-ready interface for all deep learning modules. It converts Phase 2A registered datasets into multimodal samples and batches.

### Key Components

- **`Modality` enum**: `IMAGE`, `TEXT`, `AUDIO`, `SENSOR` with ordering constants and mask utilities
- **`MultimodalSample`**: Single data point with labeled modality tensors + dict conversion
- **`MultimodalBatch`**: Batched tensors with `pin_memory()`, `to(device)`, modality masks
- **`MultimodalDataset`**: PyTorch `Dataset` consuming `DatasetLoadResult` with lazy loading and caching
- **`MultimodalDataLoader`**: Deterministic `DataLoader` supporting `drop_last` and seed control
- **`DataModule`**: Lightning-style prepare → setup → dataloaders pipeline
- **`collate_multimodal_samples`**: Padded collation of mixed-modality samples
- **Transforms**: `ImageTransform`, `TextTransform`, `AudioTransform`, `SensorTransform` with training/eval modes
- **`DatasetStatistics`**: Computes modality availability, missing ratios, class distributions, sequence lengths
- **`DataFactory`**: Singleton factory caching datasets by name

### Exception Hierarchy

```
DataValidationError
├── EmptyTensorError
├── InvalidLabelError
├── InvalidModalityMaskError
├── CorruptedSampleError
├── ShapeMismatchError
└── MissingMetadataError
```

## Tests

```bash
cd backend
python -m pytest                    # Run all tests (249 pass, 1 skip)
python -m pytest --cov=app.data     # With coverage (92% overall)
```
