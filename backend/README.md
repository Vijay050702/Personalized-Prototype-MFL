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
│   ├── prototypes/     # Prototype Learning Engine (Phase 4)
│   │   ├── prototype.py              # Prototype data class
│   │   ├── repository.py             # In-memory store/retrieve/CRUD
│   │   ├── generator.py              # Centroid/weighted/median generation
│   │   ├── memory.py                 # Global/local memory, snapshots, aging
│   │   ├── updater.py                # EMA/moving-average/adaptive update
│   │   ├── matcher.py                # Nearest/top-k/rank/batch matching
│   │   ├── similarity.py             # Cosine/euclidean/manhattan/dot
│   │   ├── confidence.py             # Confidence estimation & stability
│   │   ├── clustering.py             # KMeans/Hierarchical/DBSCAN
│   │   ├── visualization.py          # Embedding viz, heatmaps, trajectories
│   │   ├── losses.py                 # Compactness, Separation, Center, etc.
│   │   ├── metrics.py                # Intra/inter-class, purity, drift
│   │   ├── factory.py                # Static factory with default_system
│   │   └── utils.py                  # Validation, matrix utils, Timer
│   ├── evaluation/     # Research Evaluation & Paper Reproduction (Phase 9)
│   │   ├── registry.py               # MetricRegistry, BaselineRegistry, AblationRegistry, ExperimentRegistry
│   │   ├── factory.py                # EvaluationFactory, MetricFactory
│   │   ├── metrics.py                # Classification, Communication, Training, Prototype, KnowledgeTransfer, Personalization metrics
│   │   ├── baselines.py              # FedAvg, FedProx, SCAFFOLD, stub baselines
│   │   ├── ablation.py               # Ablation study configurations
│   │   ├── statistical_analysis.py   # T-test, Wilcoxon, Cohen's d, Hedges' g
│   │   ├── evaluator.py              # EvaluationEngine (training/validation/testing)
│   │   ├── experiment_runner.py      # Single/batch/ablation/baseline/multi-seed runs
│   │   ├── benchmark.py              # Model/baseline benchmarking, inference/training speed
│   │   ├── visualization_data.py     # Structured JSON data for plotting
│   │   ├── exporter.py               # JSON, CSV, Excel, Markdown, LaTeX export
│   │   ├── report_generator.py       # Auto-generated experiment reports
│   │   └── leaderboard.py            # Ranking by accuracy, F1, cost, time, quality
│   ├── routers/        # FastAPI routers
│   ├── schemas/        # Pydantic schemas
│   └── services/       # Business logic
├── tests/              # Test suite (1750 tests, 1 skipped)
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

## Phase 4: Prototype Learning Engine

The `app/prototypes/` package provides a complete local prototype learning system that consumes `torch.Tensor` embeddings from Phase 3 encoders/projections and produces prototypes for matching, similarity, confidence, clustering, and metric computation. Designed for future Personalization, Knowledge Transfer, and Federated Learning engines.

### Architecture

```
ProjectionHead Embeddings ──► PrototypeGenerator ──► PrototypeRepository
                                      │                      │
                                      │               ┌──────┴──────┐
                                      │          PrototypeMemory  PrototypeMatcher
                                      │           (global/local,   (nearest, top-k,
                                      │            snapshots,       rank, class match)
                                      │             aging)
                                      │
                              PrototypeUpdater
                              (EMA, moving_avg,
                               replacement,
                               weighted, adaptive)
```

### Core Components

- **`Prototype`** (`prototype.py`): Data class with `embedding`, `class_id`, `modality`, `sample_count`, `confidence`, `timestamp`, `metadata`. Supports `distance()`, `similarity()`, `normalize()`, `clone()`, `to_dict()`, input validation on creation and setters.
- **`PrototypeRepository`** (`repository.py`): In-memory CRUD with `store()`, `retrieve()`, `replace()`, `update()`, `remove()`, `filter()`, `by_class()`, `by_modality()`, `get_embeddings_matrix()`, `export_state()` / `import_state()`, `statistics()`.
- **`PrototypeGenerator`** (`generator.py`): Generates prototype embeddings from grouped class embeddings using `centroid` (mean), `weighted_centroid`, or `median` strategies. Supports `generate_from_embeddings()`, `generate_all()`, `generate_from_repository()`, `incremental_update()`, `batch_update()`.
- **`PrototypeMemory`** (`memory.py`): Dual-repository memory (global + local) with store, promote local→global, `snapshot()` / `restore_snapshot()`, `age_prototypes()` (time-based eviction), automatic eviction when capacity exceeded, per-prototype `get_history()`, `clear()`.
- **`PrototypeUpdater`** (`updater.py`): Update strategies — `ema` (exponential moving avg), `moving_average` (cumulative), `replacement`, `weighted`, `adaptive` (distance-scaled alpha). Supports `batch_update()`.
- **`PrototypeMatcher`** (`matcher.py`): Query engine with `match()` (top-k), `batch_match()`, `nearest_prototype()`, `rank()` (full ordering), `match_to_class()` (class-level aggregation). Supports class and modality filters.
- **`SimilarityEngine`** (`similarity.py`): Metric computation — `similarity()`, `distance()`, `batch_similarity()`, `pairwise_similarity_matrix()`, `prototype_similarity_matrix()`, `prototype_distance_matrix()`. Metrics: `cosine`, `euclidean`, `manhattan`, `dot`.
- **`ConfidenceEstimator`** (`confidence.py`): Multi-factor confidence — sample count factor, base confidence, distance factor (sigmoid-scaled similarity). Supports `batch_estimate()`, `stability_score()`, `normalized_confidence()`.
- **`PrototypeClustering`** (`clustering.py`): Pure-PyTorch clustering — `kmeans` (iterative with convergence), `hierarchical` (agglomerative), `dbscan` (density-based). Supports `cluster_prototypes()` with prototype ID tracking.
- **`VisualizationSupport`** (`visualization.py`): Data extraction for plotting — `embedding_data()`, `prototype_trajectories()`, `similarity_heatmap()`, `cluster_plot_data()`, `pairwise_distances()`, `prototype_summary()`.
- **Losses** (`losses.py`): 5 `nn.Module` losses:
  - `PrototypeCompactnessLoss` — distances between embeddings and same-class prototypes
  - `PrototypeSeparationLoss` — margin-based separation from different-class prototypes
  - `CenterLoss` — learned class centers as `nn.Parameter`
  - `PrototypeConsistencyLoss` — cosine consistency between student/teacher embeddings
  - `PrototypeDiversityLoss` — pairwise similarity penalty between prototypes
- **`PrototypeMetrics`** (`metrics.py`): Evaluation — `intra_class_distance()`, `inter_class_distance()`, `prototype_purity()`, `prototype_coverage()`, `prototype_variance()`, `prototype_drift()`, `average_confidence()`, `to_dict()`.
- **`PrototypeFactory`** (`factory.py`): Static factory — `create_*()` methods for each component, `default_system()` returning wired dictionary, `register()` / `get()` for extensibility.
- **Utils** (`utils.py`): `validate_embedding()`, `check_nan()`, `validate_class_id()`, `validate_similarity_metric()`, `cosine_similarity_matrix()`, `euclidean_distance_matrix()`, `Timer` context manager.

### Quick Start

```python
from app.prototypes import PrototypeFactory
import torch

# Build a wired default system
system = PrototypeFactory.default_system(metric="cosine")

# Generate prototypes from random embeddings
generator = system["generator"]
embeddings = torch.randn(100, 64)
labels = torch.randint(0, 5, (100,))
protos = generator.generate_all(embeddings, labels)

# Store and query
repo = system["repository"]
for p in protos:
    repo.store(p)

matcher = system["matcher"]
result = matcher.match(torch.randn(64), top_k=3)     # [(Prototype, score), ...]
best = matcher.nearest_prototype(torch.randn(64))     # (Prototype, score) | None

# Prototype updates
updater = system["updater"]
updater.update(protos[0], torch.randn(64), alpha=0.9)

# Losses
from app.prototypes.losses import CenterLoss
center_loss = CenterLoss(num_classes=5, embedding_dim=64)
loss = center_loss(embeddings, labels)

# Metrics
from app.prototypes.metrics import PrototypeMetrics
metrics = PrototypeMetrics(protos)
stats = metrics.to_dict()             # intra/inter distance, purity, drift, etc.
```

## Phase 9: Research Evaluation & Paper Reproduction

The `app/evaluation/` package provides a comprehensive evaluation framework for validating the PP-MFL system against research paper methodology. It integrates with the Phase 8 training engine to run baselines, ablations, statistical analysis, and generate paper-ready reports and exports.

### Architecture

```
EvaluationFactory ──► EvaluationEngine ──► ExperimentRunner ──► Benchmark ──► Leaderboard
       │                      │                    │                │              │
       │               ┌──────┴──────┐            │                │              │
       │          Metrics Registry   │      AblationStudy         │              │
       │               │             │            │                │              │
       │        ClassificationMetrics│     BaselineRegistry       │              │
       │        CommunicationMetrics │            │                │              │
       │        TrainingMetrics      │     FedAvgBaseline         │              │
       │        PrototypeMetrics     │     FedProxBaseline        │              │
       │    KnowledgeTransferMetrics │     SCAFFOLDBaseline       │              │
       │    PersonalizationMetrics   │     (8 baselines total)    │              │
       │                            │                            │              │
       └──────────► StatisticalAnalysis ◄─────────────────────────┘              │
       ┌──────────► Exporter ◄──────────────────────────────────────────────────┘
       │            (JSON/CSV/Excel/MD/LaTeX)
       └──────────► ReportGenerator
                    (auto-generated reports)
```

### Key Components

- **`MetricRegistry` / `BaselineRegistry` / `AblationRegistry` / `ExperimentRegistry`** (`registry.py`): Module-level registries with `register()`, `get()`, `list()`, `unregister()`, `contains()`, `clear()`.
- **`EvaluationFactory` / `MetricFactory`** (`factory.py`): Factory pattern for creating engines, metrics, baselines, ablations, and experiment runners from config dicts.
- **Metrics** (`metrics.py`): 7 metric families with static methods:
  - `ClassificationMetrics` — accuracy, precision, recall, F1 (macro/micro/weighted), balanced accuracy, confusion matrix, ROC-AUC
  - `CommunicationMetrics` — communication cost, bandwidth, latency, bytes transferred
  - `TrainingMetrics` — training time, inference time
  - `PrototypeMetrics` — drift, diversity, stability, similarity, compactness
  - `KnowledgeTransferMetrics` — alignment score, transfer accuracy, success rate, cross-modal similarity
  - `PersonalizationMetrics` — personalization gain, client adaptation score, fusion quality, confidence calibration (ECE)
  - Each family has `compute_all()` returning a dict of all metrics
- **Baselines** (`baselines.py`): 8 baseline implementations registered in `BaselineRegistry`:
  - `FedAvgBaseline`, `FedProxBaseline`, `SCAFFOLDBaseline` (full training+aggregation)
  - `PrototypeOnlyBaseline`, `WithoutPersonalizationBaseline`, `WithoutKnowledgeTransferBaseline`, `WithoutAdaptiveAggregationBaseline`, `FullPPMFLLBaseline` (stub baselines)
  - `BaselineFactory` for creating baselines by name
- **`AblationStudy`** (`ablation.py`): 6 ablation functions registered in `AblationRegistry`: `without_prototypes`, `without_aggregation`, `without_knowledge_transfer`, `without_personalization`, `without_adaptive_weighting`, `without_prototype_memory`.
- **`StatisticalAnalysis`** (`statistical_analysis.py`): Statistical tools — mean, median, variance, std, confidence interval, paired t-test, Wilcoxon signed-rank test, Cohen's d, Hedges' g, group comparison with relative improvement.
- **`EvaluationEngine`** (`evaluator.py`): Core evaluation with training/validation/testing, single/multi-client, single/multi-modality, missing modality tests, personalized evaluation, prototype evaluation, knowledge transfer evaluation, communication evaluation, resource tracking.
- **`ExperimentRunner`** (`experiment_runner.py`): Run single/batch/ablation/baseline/multi-seed/multi-dataset experiments. Save/load results as JSON. Select best experiment by metric.
- **`Benchmark`** (`benchmark.py`): Model and baseline benchmarking — inference speed, training speed, throughput, memory usage, parameter count.
- **`VisualizationDataGenerator`** (`visualization_data.py`): Generates structured JSON data for: training loss curves, accuracy curves, communication rounds, prototype drift, prototype evolution, client participation, knowledge transfer quality, personalization gain.
- **`Exporter`** (`exporter.py`): Multi-format export — JSON, CSV, Excel (via openpyxl), Markdown tables, LaTeX tables.
- **`ReportGenerator`** (`report_generator.py`): Auto-generated experiment reports with sections: summary, dataset, configuration, metrics, communication, prototype, knowledge transfer, personalization, best model, conclusions.
- **`Leaderboard`** (`leaderboard.py`): Ranking of experiment results by accuracy, F1, communication cost, training time, prototype quality, personalization gain. Supports `get_best()` for top-ranked result.

### Quick Start

```python
from app.evaluation import EvaluationFactory, MetricFactory
from app.evaluation.metrics import ClassificationMetrics
import torch

# Compute metrics
outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
targets = torch.tensor([1, 0])
acc = ClassificationMetrics.accuracy(outputs, targets)   # 1.0
f1 = ClassificationMetrics.f1_score(outputs, targets)    # 1.0
auc = ClassificationMetrics.roc_auc(outputs, targets)    # 1.0

# Use MetricFactory
fn = MetricFactory.create("accuracy")                    # returns callable
acc2 = fn(outputs, targets)                              # 1.0
acc3 = MetricFactory.compute("accuracy", outputs, targets)  # 1.0

# Create evaluation engine
engine = EvaluationFactory.create_engine({"rounds": 10})

# Run baselines
baseline = EvaluationFactory.create_baseline("fedavg", {"lr": 0.01})

# From config
setup = EvaluationFactory.from_config({
    "metrics": ["accuracy", "f1_score", "roc_auc"],
    "rounds": 20,
})

# List available
EvaluationFactory.list_available_metrics()    # ["accuracy", "balanced_accuracy", ...]
EvaluationFactory.list_available_baselines()  # ["fedavg", "fedprox", "scaffold", ...]
```

### Running Experiments

```python
from app.evaluation import ExperimentRunner, EvaluationFactory, AblationStudy

runner = ExperimentRunner({"rounds": 5, "num_clients": 3})

# Single experiment
result = runner.run_single(experiment_id="exp_001")

# Batch experiments
results = runner.run_batch([
    {"experiment_id": "exp_001", "rounds": 5},
    {"experiment_id": "exp_002", "rounds": 10},
])

# Ablation study
ablated = runner.run_ablation("without_prototypes")

# Baseline comparison
baseline_results = runner.run_baseline("fedavg")

# Full ablation suite
all_ablations = AblationStudy.run_all()
```

### Statistical Analysis

```python
from app.evaluation import StatisticalAnalysis
import numpy as np

baseline_scores = np.random.randn(10) * 0.1 + 0.8
proposed_scores = np.random.randn(10) * 0.1 + 0.85

t_stat, p_value = StatisticalAnalysis.paired_ttest(baseline_scores, proposed_scores)
w_stat, p_wilcoxon = StatisticalAnalysis.wilcoxon_test(baseline_scores, proposed_scores)
effect_size = StatisticalAnalysis.cohens_d(baseline_scores, proposed_scores)
ci = StatisticalAnalysis.confidence_interval(proposed_scores)  # 95% CI
```

### Exporting Results

```python
from app.evaluation import Exporter

exporter = Exporter(output_dir="./results")
exporter.export_json(results, "experiment_results.json")
exporter.export_csv(results, "experiment_results.csv")
exporter.export_excel(results, "experiment_results.xlsx")
exporter.export_markdown(results, "experiment_results.md")
exporter.export_latex(results, "experiment_results.tex")
```

### Generating Reports

```python
from app.evaluation import ReportGenerator

report_gen = ReportGenerator(output_dir="./reports")
report = report_gen.generate_report(
    train_results=train_metrics,
    test_results=test_metrics,
    config={"rounds": 10},
    baseline_results=baseline_metrics,
    ablation_results=ablated_metrics,
)
# report is a dict with all sections; can be exported or printed
```

## Tests

```bash
cd backend
python -m pytest                       # Run all 1750 tests (1750 pass, 1 skip)
python -m pytest --cov=app.data --cov=app.models --cov=app.prototypes --cov=app.evaluation  # Coverage (96%+)
```
