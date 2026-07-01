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
│   ├── models/         # Multimodal model layer (Phase 3)
│   │   ├── base/                    # BaseModel, BaseEncoder
│   │   ├── encoders/                # Image, Text, Audio, Sensor encoders
│   │   ├── fusion/                  # Concat, Attention, Weighted fusion
│   │   ├── projection/              # ProjectionHead (shared embedding space)
│   │   ├── classifier/              # ClassifierHead
│   │   ├── losses/                  # Classification, Contrastive losses
│   │   ├── factory.py               # Registration-based ModelFactory
│   │   ├── initialization.py        # Weight init strategies
│   │   └── utils.py                 # Parameter counting, timing, memory
│   ├── routers/        # FastAPI routers
│   ├── schemas/        # Pydantic schemas
│   └── services/       # Business logic
├── tests/              # Test suite (374 tests, 1 skipped)
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

## Phase 3: Multimodal Model Layer

The `app/models/` package provides reusable deep learning components — encoders, fusion, projection, classifier, and losses — that consume `MultimodalBatch` from Phase 2B and produce shared embeddings.

### Architecture & Data Flow

```
MultimodalBatch ──► ImageEncoder ──┐
                   ├──► TextEncoder ──┤
                   ├──► AudioEncoder ─┤──► MultimodalFusion ──► ProjectionHead ──► ClassifierHead
                   └──► SensorEncoder ┘         │                              │
                                              Shared Embedding             Class Logits
                                              (for Prototype Learning,       (for training)
                                               Knowledge Transfer, etc.)
```

### Base Classes

- **`BaseModel`** (`base/base_model.py`): Abstract `nn.Module` with `freeze()`, `unfreeze()`, `freeze_module(name)`, `save(path)`, `load(path)`, `get_grad_norm()`, `num_parameters`, `num_trainable_parameters`, device tracking
- **`BaseEncoder`** (`base/base_encoder.py`): Abstract encoder with `embedding_dim`, `output_dim`, `encode()` method

### Encoders

| Encoder | File | Architecture | Configurable |
|---|---|---|---|
| **ImageEncoder** | `encoders/image_encoder.py` | ResNet18 backbone → AdaptiveAvgPool → Linear proj | `pretrained`, `freeze_backbone`, `dropout`, `normalize`, `embedding_dim`, `output_dim` |
| **TextEncoder** | `encoders/text_encoder.py` | Token embedding → PositionalEncoding → TransformerEncoder → Mean pooling | `vocab_size`, `hidden_dim`, `num_heads`, `num_layers`, `max_seq_length`, `normalize` |
| **AudioEncoder** | `encoders/audio_encoder.py` | 1D Conv + BatchNorm + ReLU + MaxPool blocks → Global pooling → Linear proj | `in_channels`, `base_filters`, `num_layers`, `pooling` (mean/max), `normalize` |
| **SensorEncoder** | `encoders/sensor_encoder.py` | BiLSTM or TemporalCNN | `input_channels`, `hidden_dim`, `bidirectional`, `encoder_type`, `normalize` |

### Fusion Modules

- **`ConcatFusion`**: Concatenates modality embeddings → optional dropout
- **`AttentionFusion`**: Cross-modal attention with modality projections and residual connection
- **`WeightedFusion`**: Learnable weighted sum with softmax-normalized weights
- **`MultimodalFusion`**: Wrapper that selects strategy (`concat`/`attention`/`weighted`), handles missing modalities via dynamic modality masks, supports per-modality projectors

### Projection & Classifier

- **`ProjectionHead`**: MLP (Linear → LayerNorm → ReLU → Dropout) × N layers → normalized output embedding. This shared embedding space is used by Prototype Learning and Knowledge Transfer in later phases.
- **`ClassifierHead`**: Linear layer(s) → ReLU → Dropout → Logits. Supports `predict()` (softmax) and `predict_classes()` (argmax).

### Losses

- **`ClassificationLoss`**: Wraps `CrossEntropyLoss` with label smoothing and class weights
- **`ContrastiveLoss`**: Supervised NT-Xent (normalized temperature-scaled cross entropy) with optional margin
- **`EmbeddingSimilarityLoss`**: Cosine/MSE/L1 embedding comparison

### Model Factory

Registration-based factory (no if-else chains):
```python
ModelFactory.create_encoder("image", embedding_dim=512, pretrained=False)
ModelFactory.create_encoder("text", embedding_dim=256, vocab_size=30000)
ModelFactory.create_fusion(strategy="attention", embed_dim=256, num_heads=4)
ModelFactory.create_projection_head(input_dim=512, output_dim=128)
ModelFactory.create_classifier(input_dim=128, num_classes=10)
```

Custom encoders can be registered via `@register_encoder("name")`.

### Example Pipeline

```python
img_enc = ModelFactory.create_encoder("image", embedding_dim=256, pretrained=False)
txt_enc = ModelFactory.create_encoder("text", embedding_dim=128, vocab_size=1000)
fusion = MultimodalFusion(embed_dim=256, strategy="concat")
proj = ProjectionHead(input_dim=384, output_dim=64)
clf = ClassifierHead(input_dim=64, num_classes=5)

images, texts = torch.randn(4, 3, 224, 224), torch.randint(0, 100, (4, 20))
embeddings = {"image": img_enc(images), "text": txt_enc(texts)}
fused = fusion(embeddings)            # (4, 384)
shared = proj(fused)                  # (4, 64)  ← used by prototypes
logits = clf(shared)                  # (4, 5)
```

## Tests

```bash
cd backend
python -m pytest                       # Run all 374 tests (374 pass, 1 skip)
python -m pytest --cov=app.data --cov=app.models  # Coverage (94% overall)
```
