from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from app.evaluation.registry import MetricRegistry


class MetricFactory:
    @staticmethod
    def create(name: str, **kwargs: Any) -> Any:
        cls_or_fn = MetricRegistry.get(name)
        if isinstance(cls_or_fn, type):
            return cls_or_fn(**kwargs) if kwargs else cls_or_fn()
        return cls_or_fn

    @staticmethod
    def compute(name: str, *args: Any, **kwargs: Any) -> float:
        cls_or_fn = MetricRegistry.get(name)
        if isinstance(cls_or_fn, type):
            return cls_or_fn().compute(*args, **kwargs)
        return cls_or_fn(*args, **kwargs)


class ClassificationMetrics:
    @staticmethod
    def accuracy(
        outputs: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
    ) -> float:
        if isinstance(outputs, np.ndarray):
            outputs = torch.from_numpy(outputs)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        if outputs.size(0) == 0:
            return 0.0
        if outputs.ndim > 1:
            preds = outputs.argmax(dim=1)
        else:
            preds = outputs
        return float((preds == targets).float().mean().item())

    @staticmethod
    def precision(
        outputs: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        average: str = "macro",
    ) -> float:
        if isinstance(outputs, np.ndarray):
            outputs = torch.from_numpy(outputs)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        preds = outputs.argmax(dim=1) if outputs.ndim > 1 else outputs
        num_classes = int(
            outputs.size(1) if outputs.ndim > 1 else int(targets.max().item() + 1)
        )
        precisions: list[float] = []
        for c in range(num_classes):
            tp = float(((preds == c) & (targets == c)).sum().item())
            fp = float(((preds == c) & (targets != c)).sum().item())
            denom = tp + fp
            precisions.append(tp / denom if denom > 0 else 0.0)
        if average == "macro":
            return float(np.mean(precisions)) if precisions else 0.0
        if average == "micro":
            tp_total = sum(
                float(((preds == c) & (targets == c)).sum().item())
                for c in range(num_classes)
            )
            fp_total = sum(
                float(((preds == c) & (targets != c)).sum().item())
                for c in range(num_classes)
            )
            return (
                tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
            )
        if average == "weighted":
            weights = [float((targets == c).sum().item()) for c in range(num_classes)]
            total = sum(weights)
            if total == 0:
                return 0.0
            return float(np.average(precisions, weights=weights))
        return float(np.mean(precisions))

    @staticmethod
    def recall(
        outputs: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        average: str = "macro",
    ) -> float:
        if isinstance(outputs, np.ndarray):
            outputs = torch.from_numpy(outputs)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        preds = outputs.argmax(dim=1) if outputs.ndim > 1 else outputs
        num_classes = int(
            outputs.size(1) if outputs.ndim > 1 else int(targets.max().item() + 1)
        )
        recalls: list[float] = []
        for c in range(num_classes):
            tp = float(((preds == c) & (targets == c)).sum().item())
            fn = float(((preds != c) & (targets == c)).sum().item())
            denom = tp + fn
            recalls.append(tp / denom if denom > 0 else 0.0)
        if average == "macro":
            return float(np.mean(recalls)) if recalls else 0.0
        if average == "micro":
            tp_total = sum(
                float(((preds == c) & (targets == c)).sum().item())
                for c in range(num_classes)
            )
            fn_total = sum(
                float(((preds != c) & (targets == c)).sum().item())
                for c in range(num_classes)
            )
            return (
                tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            )
        if average == "weighted":
            weights = [float((targets == c).sum().item()) for c in range(num_classes)]
            total = sum(weights)
            if total == 0:
                return 0.0
            return float(np.average(recalls, weights=weights))
        return float(np.mean(recalls))

    @staticmethod
    def f1_score(
        outputs: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        average: str = "macro",
    ) -> float:
        if isinstance(outputs, np.ndarray):
            outputs = torch.from_numpy(outputs)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        if outputs.size(0) == 0:
            return 0.0
        preds = outputs.argmax(dim=1) if outputs.ndim > 1 else outputs
        num_classes = int(
            outputs.size(1) if outputs.ndim > 1 else int(targets.max().item() + 1)
        )
        f1_scores: list[float] = []
        for c in range(num_classes):
            tp = float(((preds == c) & (targets == c)).sum().item())
            fp = float(((preds == c) & (targets != c)).sum().item())
            fn = float(((preds != c) & (targets == c)).sum().item())
            denom = 2.0 * tp + fp + fn
            f1_scores.append(2.0 * tp / denom if denom > 0 else 0.0)
        if average == "macro":
            return float(np.mean(f1_scores)) if f1_scores else 0.0
        if average == "micro":
            tp_total = sum(
                float(((preds == c) & (targets == c)).sum().item())
                for c in range(num_classes)
            )
            fp_total = sum(
                float(((preds == c) & (targets != c)).sum().item())
                for c in range(num_classes)
            )
            fn_total = sum(
                float(((preds != c) & (targets == c)).sum().item())
                for c in range(num_classes)
            )
            denom = 2.0 * tp_total + fp_total + fn_total
            return 2.0 * tp_total / denom if denom > 0 else 0.0
        if average == "weighted":
            weights = [float((targets == c).sum().item()) for c in range(num_classes)]
            total = sum(weights)
            if total == 0:
                return 0.0
            return float(np.average(f1_scores, weights=weights))
        return float(np.mean(f1_scores))

    @staticmethod
    def balanced_accuracy(
        outputs: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
    ) -> float:
        if isinstance(outputs, np.ndarray):
            outputs = torch.from_numpy(outputs)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        preds = outputs.argmax(dim=1) if outputs.ndim > 1 else outputs
        num_classes = int(
            outputs.size(1) if outputs.ndim > 1 else int(targets.max().item() + 1)
        )
        recalls: list[float] = []
        for c in range(num_classes):
            tp = float(((preds == c) & (targets == c)).sum().item())
            fn = float(((preds != c) & (targets == c)).sum().item())
            denom = tp + fn
            recalls.append(tp / denom if denom > 0 else 0.0)
        return float(np.mean(recalls)) if recalls else 0.0

    @staticmethod
    def confusion_matrix(
        outputs: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        num_classes: int | None = None,
    ) -> np.ndarray:
        if isinstance(outputs, np.ndarray):
            outputs = torch.from_numpy(outputs)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        preds = outputs.argmax(dim=1) if outputs.ndim > 1 else outputs
        if num_classes is None:
            num_classes = max(int(preds.max().item()), int(targets.max().item())) + 1
        cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for t, p in zip(targets, preds):
            cm[t.long(), p.long()] += 1
        return cm.numpy()

    @staticmethod
    def roc_auc(
        outputs: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
    ) -> float:
        if isinstance(outputs, np.ndarray):
            outputs = torch.from_numpy(outputs)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        if outputs.size(0) == 0 or outputs.size(1) < 2:
            return 0.5
        num_classes = outputs.size(1)
        probs = torch.softmax(outputs, dim=1)
        if num_classes == 2:
            return float(ClassificationMetrics._auc(probs[:, 1], targets.float()))
        aucs: list[float] = []
        for c in range(num_classes):
            binary_targets = (targets == c).float()
            aucs.append(ClassificationMetrics._auc(probs[:, c], binary_targets))
        return float(np.mean(aucs)) if aucs else 0.5

    @staticmethod
    def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
        sorted_idx = torch.argsort(scores)
        labels_sorted = labels[sorted_idx]
        pos_count = int(labels_sorted.sum().item())
        neg_count = int((1 - labels_sorted).sum().item())
        if pos_count == 0 or neg_count == 0:
            return 0.5
        pos_indices = torch.where(labels_sorted == 1)[0].float()
        pos_rank_sum = float((pos_indices + 1).sum().item())
        return (pos_rank_sum - pos_count * (pos_count + 1) / 2) / (
            pos_count * neg_count
        )

    @staticmethod
    def compute_all(
        outputs: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
    ) -> dict[str, float]:
        return {
            "accuracy": ClassificationMetrics.accuracy(outputs, targets),
            "precision": ClassificationMetrics.precision(outputs, targets),
            "recall": ClassificationMetrics.recall(outputs, targets),
            "f1_score": ClassificationMetrics.f1_score(outputs, targets),
            "macro_f1": ClassificationMetrics.f1_score(
                outputs, targets, average="macro"
            ),
            "micro_f1": ClassificationMetrics.f1_score(
                outputs, targets, average="micro"
            ),
            "balanced_accuracy": ClassificationMetrics.balanced_accuracy(
                outputs, targets
            ),
        }


class CommunicationMetrics:
    @staticmethod
    def communication_cost(
        bytes_sent: int = 0,
        bytes_received: int = 0,
    ) -> float:
        return float(bytes_sent + bytes_received)

    @staticmethod
    def bandwidth(bytes_sent: int = 0, duration_seconds: float = 1.0) -> float:
        if duration_seconds <= 0:
            return 0.0
        return float(bytes_sent) / duration_seconds

    @staticmethod
    def latency(
        send_time: float,
        receive_time: float,
    ) -> float:
        return float(max(0.0, receive_time - send_time))

    @staticmethod
    def bytes_transferred(
        messages: list[dict[str, Any]] | None = None,
    ) -> int:
        if not messages:
            return 0
        return sum(len(str(m.get("payload", "")).encode("utf-8")) for m in messages)

    @staticmethod
    def compute_all(
        bytes_sent: int = 0,
        bytes_received: int = 0,
        duration_seconds: float = 1.0,
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        return {
            "communication_cost": CommunicationMetrics.communication_cost(
                bytes_sent, bytes_received
            ),
            "bandwidth": CommunicationMetrics.bandwidth(bytes_sent, duration_seconds),
            "bytes_transferred": float(
                CommunicationMetrics.bytes_transferred(messages)
            ),
        }


class TrainingMetrics:
    @staticmethod
    def training_time(start_time: float, end_time: float) -> float:
        return float(max(0.0, end_time - start_time))

    @staticmethod
    def inference_time(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        repetitions: int = 10,
    ) -> float:
        model.eval()
        with torch.no_grad():
            for _ in range(3):
                model(inputs)
        starter = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )
        ender = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )
        if starter is not None and ender is not None:
            starter.record()
            for _ in range(repetitions):
                model(inputs)
            ender.record()
            torch.cuda.synchronize()
            return float(starter.elapsed_time(ender)) / repetitions
        total = 0.0
        with torch.no_grad():
            for _ in range(repetitions):
                t0 = __import__("time").time()
                model(inputs)
                total += __import__("time").time() - t0
        return (total / repetitions) * 1000.0

    @staticmethod
    def compute_all(
        start_time: float = 0.0,
        end_time: float = 0.0,
    ) -> dict[str, float]:
        return {
            "training_time": TrainingMetrics.training_time(start_time, end_time),
        }


class PrototypeMetrics:
    @staticmethod
    def prototype_drift(
        old_embeddings: list[torch.Tensor] | torch.Tensor,
        new_embeddings: list[torch.Tensor] | torch.Tensor,
    ) -> float:
        if isinstance(old_embeddings, list):
            if not old_embeddings:
                return 0.0
            old_t = (
                torch.stack(old_embeddings)
                if isinstance(old_embeddings[0], torch.Tensor)
                else torch.tensor(old_embeddings)
            )
        else:
            old_t = old_embeddings
        if isinstance(new_embeddings, list):
            if not new_embeddings:
                return 0.0
            new_t = (
                torch.stack(new_embeddings)
                if isinstance(new_embeddings[0], torch.Tensor)
                else torch.tensor(new_embeddings)
            )
        else:
            new_t = new_embeddings
        return float(torch.nn.functional.pairwise_distance(old_t, new_t).mean().item())

    @staticmethod
    def prototype_diversity(
        embeddings: list[torch.Tensor] | torch.Tensor,
    ) -> float:
        if isinstance(embeddings, list):
            if not embeddings:
                return 0.0
            emb_t = (
                torch.stack(embeddings)
                if isinstance(embeddings[0], torch.Tensor)
                else torch.tensor(embeddings)
            )
        else:
            emb_t = embeddings
        if emb_t.size(0) < 2:
            return 0.0
        emb_norm = emb_t / (emb_t.norm(dim=1, keepdim=True) + 1e-8)
        sim_matrix = torch.mm(emb_norm, emb_norm.T)
        mask = 1.0 - torch.eye(emb_t.size(0), device=emb_t.device)
        mean_sim = (sim_matrix * mask).sum() / mask.sum()
        return float(max(0.0, min(1.0, 1.0 - mean_sim)))

    @staticmethod
    def prototype_stability(
        history: list[float] | list[torch.Tensor],
    ) -> float:
        if len(history) < 2:
            return 1.0
        vals = torch.tensor([float(v) for v in history], dtype=torch.float32)
        variance = vals.var().item()
        return float(1.0 / (1.0 + variance))

    @staticmethod
    def prototype_similarity(
        emb1: torch.Tensor,
        emb2: torch.Tensor,
    ) -> float:
        return float(
            torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0), emb2.unsqueeze(0)
            ).item()
        )

    @staticmethod
    def prototype_compactness(
        embeddings: list[torch.Tensor] | torch.Tensor,
        centers: list[torch.Tensor] | torch.Tensor | None = None,
    ) -> float:
        if isinstance(embeddings, list):
            if not embeddings:
                return 0.0
            emb_t = (
                torch.stack(embeddings)
                if isinstance(embeddings[0], torch.Tensor)
                else torch.tensor(embeddings)
            )
        else:
            emb_t = embeddings
        if centers is None:
            center = emb_t.mean(dim=0, keepdim=True)
        else:
            if isinstance(centers, list):
                centers_t = (
                    torch.stack(centers)
                    if isinstance(centers[0], torch.Tensor)
                    else torch.tensor(centers)
                )
            else:
                centers_t = centers
            center = centers_t.mean(dim=0, keepdim=True)
        dists = torch.norm(emb_t - center, p=2, dim=1)
        return float(dists.mean().item())

    @staticmethod
    def compute_all(
        old_embeddings: list[torch.Tensor] | None = None,
        new_embeddings: list[torch.Tensor] | None = None,
        all_embeddings: list[torch.Tensor] | None = None,
        history: list[float] | None = None,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if old_embeddings is not None and new_embeddings is not None:
            metrics["prototype_drift"] = PrototypeMetrics.prototype_drift(
                old_embeddings, new_embeddings
            )
        if all_embeddings is not None:
            metrics["prototype_diversity"] = PrototypeMetrics.prototype_diversity(
                all_embeddings
            )
            metrics["prototype_compactness"] = PrototypeMetrics.prototype_compactness(
                all_embeddings
            )
        if history is not None:
            metrics["prototype_stability"] = PrototypeMetrics.prototype_stability(
                history
            )
        return metrics


class KnowledgeTransferMetrics:
    @staticmethod
    def alignment_score(
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> float:
        sims = torch.nn.functional.cosine_similarity(
            source_embeddings, target_embeddings
        )
        return float(sims.mean().item())

    @staticmethod
    def transfer_accuracy(
        predicted: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
    ) -> float:
        return ClassificationMetrics.accuracy(predicted, targets)

    @staticmethod
    def transfer_success_rate(
        valid_transfers: int,
        total_attempts: int,
    ) -> float:
        if total_attempts == 0:
            return 0.0
        return float(valid_transfers) / float(total_attempts)

    @staticmethod
    def cross_modal_similarity(
        modality_a_embeddings: torch.Tensor,
        modality_b_embeddings: torch.Tensor,
    ) -> float:
        if modality_a_embeddings.size(0) != modality_b_embeddings.size(0):
            n = min(modality_a_embeddings.size(0), modality_b_embeddings.size(0))
            modality_a_embeddings = modality_a_embeddings[:n]
            modality_b_embeddings = modality_b_embeddings[:n]
        sims = torch.nn.functional.cosine_similarity(
            modality_a_embeddings, modality_b_embeddings
        )
        return float(sims.mean().item())

    @staticmethod
    def compute_all(
        source_embeddings: torch.Tensor | None = None,
        target_embeddings: torch.Tensor | None = None,
        predicted: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        valid_transfers: int = 0,
        total_attempts: int = 0,
        mod_a_embeddings: torch.Tensor | None = None,
        mod_b_embeddings: torch.Tensor | None = None,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if source_embeddings is not None and target_embeddings is not None:
            metrics["alignment_score"] = KnowledgeTransferMetrics.alignment_score(
                source_embeddings, target_embeddings
            )
        if predicted is not None and targets is not None:
            metrics["transfer_accuracy"] = KnowledgeTransferMetrics.transfer_accuracy(
                predicted, targets
            )
        if total_attempts > 0:
            metrics["transfer_success_rate"] = (
                KnowledgeTransferMetrics.transfer_success_rate(
                    valid_transfers, total_attempts
                )
            )
        if mod_a_embeddings is not None and mod_b_embeddings is not None:
            metrics["cross_modal_similarity"] = (
                KnowledgeTransferMetrics.cross_modal_similarity(
                    mod_a_embeddings, mod_b_embeddings
                )
            )
        return metrics


class PersonalizationMetrics:
    @staticmethod
    def personalization_gain(
        personalized: torch.Tensor,
        global_prototype: torch.Tensor,
    ) -> float:
        sim = torch.nn.functional.cosine_similarity(
            personalized.unsqueeze(0), global_prototype.unsqueeze(0)
        )
        return float((1.0 - sim).item())

    @staticmethod
    def client_adaptation_score(
        pre_adaptation: torch.Tensor,
        post_adaptation: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        pre_acc = ClassificationMetrics.accuracy(pre_adaptation, targets)
        post_acc = ClassificationMetrics.accuracy(post_adaptation, targets)
        return float(max(0.0, post_acc - pre_acc))

    @staticmethod
    def prototype_fusion_quality(
        fusion_weights: dict[str, float],
    ) -> float:
        if not fusion_weights:
            return 0.0
        values = list(fusion_weights.values())
        if len(values) <= 1:
            return 1.0
        vals_t = torch.tensor(values, dtype=torch.float32)
        entropy = float(-(vals_t * torch.log(vals_t + 1e-8)).sum().item())
        max_entropy = float(math.log(len(values)))
        if max_entropy == 0:
            return 1.0
        return float(1.0 - (entropy / max_entropy))

    @staticmethod
    def confidence_calibration(
        confidences: list[float],
        accuracies: list[float],
        num_bins: int = 10,
    ) -> float:
        if not confidences or not accuracies:
            return 0.0
        if len(confidences) != len(accuracies):
            n = min(len(confidences), len(accuracies))
            confidences = confidences[:n]
            accuracies = accuracies[:n]
        conf_t = torch.tensor(confidences, dtype=torch.float32)
        acc_t = torch.tensor(accuracies, dtype=torch.float32)
        bin_edges = torch.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        for i in range(num_bins):
            mask = (conf_t >= bin_edges[i]) & (conf_t < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_conf = conf_t[mask].mean().item()
            bin_acc = acc_t[mask].mean().item()
            ece += (mask.sum().item() / len(conf_t)) * abs(bin_acc - bin_conf)
        return float(ece)

    @staticmethod
    def compute_all(
        personalized: torch.Tensor | None = None,
        global_prototype: torch.Tensor | None = None,
        pre_adaptation: torch.Tensor | None = None,
        post_adaptation: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        fusion_weights: dict[str, float] | None = None,
        confidences: list[float] | None = None,
        accuracies: list[float] | None = None,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if personalized is not None and global_prototype is not None:
            metrics["personalization_gain"] = (
                PersonalizationMetrics.personalization_gain(
                    personalized, global_prototype
                )
            )
        if (
            pre_adaptation is not None
            and post_adaptation is not None
            and targets is not None
        ):
            metrics["client_adaptation_score"] = (
                PersonalizationMetrics.client_adaptation_score(
                    pre_adaptation, post_adaptation, targets
                )
            )
        if fusion_weights is not None:
            metrics["prototype_fusion_quality"] = (
                PersonalizationMetrics.prototype_fusion_quality(fusion_weights)
            )
        if confidences is not None and accuracies is not None:
            metrics["confidence_calibration"] = (
                PersonalizationMetrics.confidence_calibration(confidences, accuracies)
            )
        return metrics


MetricRegistry.register("accuracy", ClassificationMetrics.accuracy)
MetricRegistry.register("precision", ClassificationMetrics.precision)
MetricRegistry.register("recall", ClassificationMetrics.recall)
MetricRegistry.register("f1_score", ClassificationMetrics.f1_score)
MetricRegistry.register(
    "macro_f1", lambda o, t: ClassificationMetrics.f1_score(o, t, average="macro")
)
MetricRegistry.register(
    "micro_f1", lambda o, t: ClassificationMetrics.f1_score(o, t, average="micro")
)
MetricRegistry.register("balanced_accuracy", ClassificationMetrics.balanced_accuracy)
MetricRegistry.register("confusion_matrix", ClassificationMetrics.confusion_matrix)
MetricRegistry.register("roc_auc", ClassificationMetrics.roc_auc)
MetricRegistry.register("communication_cost", CommunicationMetrics.communication_cost)
MetricRegistry.register("bandwidth", CommunicationMetrics.bandwidth)
MetricRegistry.register("latency", CommunicationMetrics.latency)
MetricRegistry.register("bytes_transferred", CommunicationMetrics.bytes_transferred)
MetricRegistry.register("training_time", TrainingMetrics.training_time)
MetricRegistry.register("inference_time", TrainingMetrics.inference_time)
MetricRegistry.register("prototype_drift", PrototypeMetrics.prototype_drift)
MetricRegistry.register("prototype_diversity", PrototypeMetrics.prototype_diversity)
MetricRegistry.register("prototype_stability", PrototypeMetrics.prototype_stability)
MetricRegistry.register("prototype_similarity", PrototypeMetrics.prototype_similarity)
MetricRegistry.register("prototype_compactness", PrototypeMetrics.prototype_compactness)
MetricRegistry.register("alignment_score", KnowledgeTransferMetrics.alignment_score)
MetricRegistry.register("transfer_accuracy", KnowledgeTransferMetrics.transfer_accuracy)
MetricRegistry.register(
    "transfer_success_rate", KnowledgeTransferMetrics.transfer_success_rate
)
MetricRegistry.register(
    "cross_modal_similarity", KnowledgeTransferMetrics.cross_modal_similarity
)
MetricRegistry.register(
    "personalization_gain", PersonalizationMetrics.personalization_gain
)
MetricRegistry.register(
    "client_adaptation_score", PersonalizationMetrics.client_adaptation_score
)
MetricRegistry.register(
    "prototype_fusion_quality", PersonalizationMetrics.prototype_fusion_quality
)
MetricRegistry.register(
    "confidence_calibration", PersonalizationMetrics.confidence_calibration
)
