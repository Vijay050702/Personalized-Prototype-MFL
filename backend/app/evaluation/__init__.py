from app.evaluation.ablation import (
    AblationStudy,
    without_prototypes,
    without_aggregation,
)
from app.evaluation.ablation import without_knowledge_transfer, without_personalization
from app.evaluation.ablation import without_adaptive_weighting, without_prototype_memory
from app.evaluation.baselines import (
    Baseline,
    BaselineFactory,
    FedAvgBaseline,
    FedProxBaseline,
    FullPPMFLLBaseline,
    PrototypeOnlyBaseline,
    SCAFFOLDBaseline,
    WithoutAdaptiveAggregationBaseline,
    WithoutKnowledgeTransferBaseline,
    WithoutPersonalizationBaseline,
)
from app.evaluation.benchmark import Benchmark
from app.evaluation.evaluator import EvaluationEngine
from app.evaluation.experiment_runner import ExperimentRunner
from app.evaluation.exporter import Exporter
from app.evaluation.factory import EvaluationFactory
from app.evaluation.leaderboard import Leaderboard
from app.evaluation.metrics import (
    ClassificationMetrics,
    CommunicationMetrics,
    KnowledgeTransferMetrics,
    MetricFactory,
    PersonalizationMetrics,
    PrototypeMetrics,
    TrainingMetrics,
)
from app.evaluation.registry import (
    AblationRegistry,
    BaselineRegistry,
    ExperimentRegistry,
    MetricRegistry,
)
from app.evaluation.report_generator import ReportGenerator
from app.evaluation.statistical_analysis import StatisticalAnalysis
from app.evaluation.visualization_data import VisualizationDataGenerator

__all__ = [
    # Registry
    "MetricRegistry",
    "BaselineRegistry",
    "AblationRegistry",
    "ExperimentRegistry",
    # Factory
    "EvaluationFactory",
    "MetricFactory",
    "BaselineFactory",
    # Engine
    "EvaluationEngine",
    # Metrics
    "ClassificationMetrics",
    "CommunicationMetrics",
    "TrainingMetrics",
    "PrototypeMetrics",
    "KnowledgeTransferMetrics",
    "PersonalizationMetrics",
    # Baselines
    "Baseline",
    "FedAvgBaseline",
    "FedProxBaseline",
    "SCAFFOLDBaseline",
    "PrototypeOnlyBaseline",
    "WithoutPersonalizationBaseline",
    "WithoutKnowledgeTransferBaseline",
    "WithoutAdaptiveAggregationBaseline",
    "FullPPMFLLBaseline",
    "BaselineFactory",
    # Ablation
    "AblationStudy",
    "without_prototypes",
    "without_aggregation",
    "without_knowledge_transfer",
    "without_personalization",
    "without_adaptive_weighting",
    "without_prototype_memory",
    # Statistical Analysis
    "StatisticalAnalysis",
    # Experiment Runner
    "ExperimentRunner",
    # Benchmark
    "Benchmark",
    # Visualization Data
    "VisualizationDataGenerator",
    # Exporter
    "Exporter",
    # Report Generator
    "ReportGenerator",
    # Leaderboard
    "Leaderboard",
]
