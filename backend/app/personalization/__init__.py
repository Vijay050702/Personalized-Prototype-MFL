from app.personalization.adaptive_gate import AdaptiveGate
from app.personalization.adaptation import AdaptationEngine
from app.personalization.confidence import PersonalizedConfidence
from app.personalization.factory import PersonalizationFactory
from app.personalization.fusion_engine import FusionEngine
from app.personalization.gating_network import GatingNetwork
from app.personalization.losses import (
    AdaptiveWeightingLoss,
    CombinedPersonalizationLoss,
    ConsistencyLoss,
    FusionLoss,
    PersonalizationLoss,
    PrototypeRegularizationLoss,
)
from app.personalization.metrics import PersonalizationMetrics
from app.personalization.personalized_memory import PersonalizedMemory
from app.personalization.personalized_prototype import PersonalizedPrototype
from app.personalization.profile import ClientProfile
from app.personalization.prototype_selector import PrototypeSelector
from app.personalization.regularization import (
    FusionSmoothnessRegularization,
    PrototypeConsistencyRegularization,
    PrototypeStabilityRegularization,
    TemporalConsistencyRegularization,
)
from app.personalization.registry import PersonalizationRegistry
from app.personalization.utils import PersonalizationLogger
from app.personalization.validation import (
    validate_confidence_range,
    validate_dimensions,
    validate_duplicate_prototypes,
    validate_fusion_sources,
    validate_missing_modalities,
    validate_shape_match,
    validate_weights_sum_to_one,
)
from app.personalization.weighting import WeightCalculator

__all__ = [
    "AdaptiveGate",
    "AdaptationEngine",
    "AdaptiveWeightingLoss",
    "ClientProfile",
    "CombinedPersonalizationLoss",
    "ConsistencyLoss",
    "FusionEngine",
    "FusionLoss",
    "FusionSmoothnessRegularization",
    "GatingNetwork",
    "PersonalizationFactory",
    "PersonalizationLogger",
    "PersonalizationMetrics",
    "PersonalizationLoss",
    "PersonalizationRegistry",
    "PersonalizedConfidence",
    "PersonalizedMemory",
    "PersonalizedPrototype",
    "PrototypeConsistencyRegularization",
    "PrototypeRegularizationLoss",
    "PrototypeSelector",
    "PrototypeStabilityRegularization",
    "TemporalConsistencyRegularization",
    "WeightCalculator",
    "validate_confidence_range",
    "validate_dimensions",
    "validate_duplicate_prototypes",
    "validate_fusion_sources",
    "validate_missing_modalities",
    "validate_shape_match",
    "validate_weights_sum_to_one",
]
