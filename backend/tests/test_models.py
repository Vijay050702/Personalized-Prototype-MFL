from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from app.models.base.base_encoder import BaseEncoder
from app.models.base.base_model import BaseModel
from app.models.classifier.classifier_head import ClassifierHead
from app.models.encoders.audio_encoder import AudioEncoder
from app.models.encoders.image_encoder import ImageEncoder
from app.models.encoders.sensor_encoder import SensorEncoder
from app.models.encoders.text_encoder import TextEncoder
from app.models.factory import (
    ModelFactory,
    _ENCODER_REGISTRY,
    get_encoder_class,
    register_encoder,
)
from app.models.fusion.attention_fusion import AttentionFusion
from app.models.fusion.concat_fusion import ConcatFusion
from app.models.fusion.fusion_strategy import FusionStrategy
from app.models.fusion.multimodal_fusion import MultimodalFusion, WeightedFusion
from app.models.initialization import (
    _INITIALIZERS,
    get_initializer,
    initialize_weights,
    kaiming_uniform,
    normal_init,
    orthogonal_init,
    register_initializer,
    xavier_normal,
    xavier_uniform,
)
from app.models.losses.classification_loss import ClassificationLoss
from app.models.losses.contrastive_loss import ContrastiveLoss, EmbeddingSimilarityLoss
from app.models.projection.projection_head import ProjectionHead
from app.models.utils import (
    Timer,
    count_parameters,
    estimate_memory_usage,
    get_device,
    log_model_summary,
)


class TestBaseModel:
    def test_freeze_unfreeze(self):
        model = _ConcreteModel()
        assert model.num_trainable_parameters > 0
        model.freeze()
        assert model.num_trainable_parameters == 0
        model.unfreeze()
        assert model.num_trainable_parameters > 0

    def test_freeze_module(self):
        model = _ConcreteModel()
        model.freeze_module("fc")
        assert all(not p.requires_grad for p in model.fc.parameters())

    def test_freeze_module_invalid(self):
        model = _ConcreteModel()
        with pytest.raises(AttributeError):
            model.freeze_module("nonexistent")

    def test_unfreeze_module(self):
        model = _ConcreteModel()
        model.freeze()
        model.unfreeze_module("fc")
        assert any(p.requires_grad for p in model.fc.parameters())

    def test_num_parameters(self):
        model = _ConcreteModel()
        assert model.num_parameters > 0
        assert isinstance(model.num_parameters, int)

    def test_num_trainable_parameters(self):
        model = _ConcreteModel()
        model.freeze()
        assert model.num_trainable_parameters == 0

    def test_save_load(self):
        model = _ConcreteModel()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = _ConcreteModel()
            loaded.load(path)
            assert loaded.num_parameters == model.num_parameters
        finally:
            Path(path).unlink(missing_ok=True)

    def test_to_device(self):
        model = _ConcreteModel()
        device = torch.device("cpu")
        moved = model.to(device)
        assert moved.device == device

    def test_device_property(self):
        model = _ConcreteModel()
        model.to("cpu")
        assert model.device.type == "cpu"

    def test_get_grad_norm(self):
        model = _ConcreteModel()
        x = torch.randn(2, 10)
        out = model(x).sum()
        out.backward()
        grad_norm = model.get_grad_norm()
        assert grad_norm > 0

    def test_repr(self):
        model = _ConcreteModel()
        r = repr(model)
        assert "ConcreteModel" in r
        assert "params" in r


class _ConcreteModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestBaseEncoder:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseEncoder(embedding_dim=128)

    def test_concrete_encoder(self):
        enc = _ConcreteEncoder(embedding_dim=64, output_dim=128)
        assert enc.embedding_dim == 64
        assert enc.output_dim == 128
        x = torch.randn(2, 10)
        out = enc(x)
        assert out.shape == (2, 128)

    def test_default_output_dim(self):
        enc = _ConcreteEncoder(embedding_dim=64)
        assert enc.output_dim == 64


class _ConcreteEncoder(BaseEncoder):
    def __init__(self, embedding_dim: int = 64, output_dim: int | None = None):
        super().__init__(embedding_dim=embedding_dim, output_dim=output_dim)
        self.net = torch.nn.Linear(10, self._output_dim)

    def encode(self, x: torch.Tensor, **kwargs):
        return self.net(x)


class TestImageEncoder:
    def test_forward_shape(self):
        encoder = ImageEncoder(embedding_dim=512, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 512)

    def test_forward_normalize(self):
        encoder = ImageEncoder(embedding_dim=512, pretrained=False, normalize=True)
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert torch.allclose(out.norm(p=2, dim=1), torch.ones(2), atol=1e-5)

    def test_forward_dropout(self):
        encoder = ImageEncoder(embedding_dim=256, pretrained=False, dropout=0.5)
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 256)

    def test_freeze_backbone(self):
        encoder = ImageEncoder(
            embedding_dim=256, pretrained=False, freeze_backbone=True
        )
        for name, param in encoder.backbone.named_parameters():
            assert not param.requires_grad, f"backbone param {name} is not frozen"

    def test_custom_output_dim(self):
        encoder = ImageEncoder(embedding_dim=512, output_dim=256, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 256)

    def test_encoder_properties(self):
        encoder = ImageEncoder(embedding_dim=512, pretrained=False)
        assert encoder.embedding_dim == 512
        assert encoder.num_parameters > 0


class TestTextEncoder:
    def test_forward_shape(self):
        encoder = TextEncoder(
            vocab_size=1000, embedding_dim=64, num_heads=4, num_layers=2, hidden_dim=128
        )
        x = torch.randint(0, 100, (2, 20))
        out = encoder(x)
        assert out.shape == (2, 64)

    def test_with_attention_mask(self):
        encoder = TextEncoder(
            vocab_size=1000, embedding_dim=64, num_heads=4, num_layers=2, hidden_dim=128
        )
        x = torch.randint(0, 100, (2, 20))
        mask = torch.ones(2, 20, dtype=torch.bool)
        mask[1, 15:] = False
        out = encoder(x, mask=mask)
        assert out.shape == (2, 64)

    def test_custom_output_dim(self):
        encoder = TextEncoder(
            vocab_size=1000,
            embedding_dim=64,
            output_dim=128,
            num_heads=4,
            num_layers=2,
            hidden_dim=128,
        )
        x = torch.randint(0, 100, (2, 20))
        out = encoder(x)
        assert out.shape == (2, 128)

    def test_normalize(self):
        encoder = TextEncoder(
            vocab_size=1000,
            embedding_dim=64,
            num_heads=4,
            num_layers=2,
            hidden_dim=128,
            normalize=True,
        )
        x = torch.randint(0, 100, (2, 20))
        out = encoder(x)
        assert torch.allclose(out.norm(p=2, dim=1), torch.ones(2), atol=1e-5)

    def test_variable_length(self):
        encoder = TextEncoder(
            vocab_size=1000, embedding_dim=64, num_heads=4, num_layers=2, hidden_dim=128
        )
        x1 = torch.randint(0, 100, (2, 10))
        x2 = torch.randint(0, 100, (2, 30))
        out1 = encoder(x1)
        out2 = encoder(x2)
        assert out1.shape == out2.shape

    def test_encoder_properties(self):
        encoder = TextEncoder(
            vocab_size=1000, embedding_dim=64, num_heads=4, num_layers=2, hidden_dim=128
        )
        assert encoder.embedding_dim == 64
        assert encoder.num_parameters > 0


class TestAudioEncoder:
    def test_forward_shape(self):
        encoder = AudioEncoder(embedding_dim=128, in_channels=1, num_layers=2)
        x = torch.randn(2, 1, 16000)
        out = encoder(x)
        assert out.shape == (2, 128)

    def test_max_pooling(self):
        encoder = AudioEncoder(
            embedding_dim=128, in_channels=1, num_layers=2, pooling="max"
        )
        x = torch.randn(2, 1, 16000)
        out = encoder(x)
        assert out.shape == (2, 128)

    def test_no_pooling(self):
        encoder = AudioEncoder(
            embedding_dim=128, in_channels=1, num_layers=2, pooling="none"
        )
        x = torch.randn(2, 1, 16000)
        out = encoder(x)
        assert out.shape == (2, 128)

    def test_variable_length(self):
        encoder = AudioEncoder(embedding_dim=128, in_channels=1, num_layers=2)
        x_short = torch.randn(1, 1, 8000)
        x_long = torch.randn(1, 1, 32000)
        out_short = encoder(x_short)
        out_long = encoder(x_long)
        assert out_short.shape == out_long.shape

    def test_normalize(self):
        encoder = AudioEncoder(
            embedding_dim=128, in_channels=1, num_layers=2, normalize=True
        )
        x = torch.randn(2, 1, 16000)
        out = encoder(x)
        assert torch.allclose(out.norm(p=2, dim=1), torch.ones(2), atol=1e-5)

    def test_multi_channel(self):
        encoder = AudioEncoder(embedding_dim=128, in_channels=2, num_layers=2)
        x = torch.randn(2, 2, 16000)
        out = encoder(x)
        assert out.shape == (2, 128)


class TestSensorEncoder:
    def test_bilstm_forward(self):
        encoder = SensorEncoder(
            embedding_dim=128, input_channels=9, encoder_type="bilstm"
        )
        x = torch.randn(2, 128, 9)
        out = encoder(x)
        assert out.shape == (2, 128)

    def test_tcnn_forward(self):
        encoder = SensorEncoder(
            embedding_dim=128, input_channels=9, encoder_type="tcnn"
        )
        x = torch.randn(2, 9, 500)
        out = encoder(x)
        assert out.shape == (2, 128)

    def test_invalid_encoder_type(self):
        with pytest.raises(ValueError):
            SensorEncoder(embedding_dim=128, input_channels=9, encoder_type="invalid")

    def test_bidirectional(self):
        encoder = SensorEncoder(
            embedding_dim=128,
            input_channels=9,
            encoder_type="bilstm",
            bidirectional=False,
        )
        x = torch.randn(2, 128, 9)
        out = encoder(x)
        assert out.shape == (2, 128)

    def test_normalize(self):
        encoder = SensorEncoder(
            embedding_dim=128, input_channels=9, encoder_type="bilstm", normalize=True
        )
        x = torch.randn(2, 128, 9)
        out = encoder(x)
        assert torch.allclose(out.norm(p=2, dim=1), torch.ones(2), atol=1e-5)

    def test_variable_length_lstm(self):
        encoder = SensorEncoder(
            embedding_dim=128, input_channels=9, encoder_type="bilstm"
        )
        x_short = torch.randn(1, 64, 9)
        x_long = torch.randn(1, 256, 9)
        assert encoder(x_short).shape == encoder(x_long).shape


class TestConcatFusion:
    def test_fuse_two_modalities(self):
        fusion = ConcatFusion()
        embeddings = {
            "image": torch.randn(4, 256),
            "text": torch.randn(4, 128),
        }
        out = fusion.forward(embeddings)
        assert out.shape == (4, 384)

    def test_fuse_three_modalities(self):
        fusion = ConcatFusion()
        embeddings = {
            "image": torch.randn(4, 256),
            "text": torch.randn(4, 128),
            "audio": torch.randn(4, 64),
        }
        out = fusion.forward(embeddings)
        assert out.shape == (4, 448)

    def test_fuse_single_modality(self):
        fusion = ConcatFusion()
        embeddings = {"image": torch.randn(4, 256)}
        out = fusion.forward(embeddings)
        assert out.shape == (4, 256)

    def test_fuse_no_embeddings(self):
        fusion = ConcatFusion()
        with pytest.raises(ValueError):
            fusion.forward({})

    def test_with_dropout(self):
        fusion = ConcatFusion(dropout=0.5)
        fusion.train()
        embeddings = {
            "image": torch.randn(4, 256),
            "text": torch.randn(4, 128),
        }
        out = fusion.forward(embeddings)
        assert out.shape == (4, 384)

    def test_is_fusion_strategy(self):
        fusion = ConcatFusion()
        assert isinstance(fusion, FusionStrategy)


class TestAttentionFusion:
    def test_fuse_two_modalities(self):
        fusion = AttentionFusion(embed_dim=256, num_heads=4)
        embeddings = {
            "image": torch.randn(4, 256),
            "text": torch.randn(4, 256),
        }
        out = fusion.forward(embeddings)
        assert out.shape == (4, 256)

    def test_fuse_single_modality(self):
        fusion = AttentionFusion(embed_dim=256, num_heads=4)
        embeddings = {"image": torch.randn(4, 256)}
        out = fusion.forward(embeddings)
        assert out.shape == (4, 256)

    def test_no_embeddings(self):
        fusion = AttentionFusion(embed_dim=256, num_heads=4)
        with pytest.raises(ValueError):
            fusion.forward({})

    def test_with_modality_mask(self):
        fusion = AttentionFusion(embed_dim=256, num_heads=4)
        embeddings = {
            "image": torch.randn(4, 256),
            "text": torch.randn(4, 256),
        }
        mask = torch.ones(4, 4, dtype=torch.bool)
        out = fusion.forward(embeddings, modality_mask=mask)
        assert out.shape == (4, 256)

    def test_output_dim(self):
        fusion = AttentionFusion(embed_dim=512, num_heads=4)
        assert fusion.output_dim == 512

    def test_is_fusion_strategy(self):
        fusion = AttentionFusion(embed_dim=256, num_heads=4)
        assert isinstance(fusion, FusionStrategy)


class TestWeightedFusion:
    def test_fuse_two_modalities(self):
        fusion = WeightedFusion(embed_dim=128)
        embeddings = {
            "image": torch.randn(4, 128),
            "text": torch.randn(4, 128),
        }
        out = fusion.forward(embeddings)
        assert out.shape == (4, 128)

    def test_fuse_single_modality(self):
        fusion = WeightedFusion(embed_dim=128)
        embeddings = {"image": torch.randn(4, 128)}
        out = fusion.forward(embeddings)
        assert out.shape == (4, 128)

    def test_no_embeddings(self):
        fusion = WeightedFusion(embed_dim=128)
        with pytest.raises(ValueError):
            fusion.forward({})

    def test_output_dim(self):
        fusion = WeightedFusion(embed_dim=256)
        assert fusion.output_dim == 256


class TestMultimodalFusion:
    def test_concat_strategy(self):
        fusion = MultimodalFusion(embed_dim=256, strategy="concat")
        embeddings = {
            "image": torch.randn(4, 256),
            "text": torch.randn(4, 128),
        }
        out = fusion.forward(embeddings)
        assert out.shape == (4, 384)

    def test_attention_strategy(self):
        fusion = MultimodalFusion(embed_dim=256, strategy="attention", num_heads=4)
        embeddings = {
            "image": torch.randn(4, 256),
            "text": torch.randn(4, 256),
        }
        out = fusion.forward(embeddings)
        assert out.shape == (4, 256)

    def test_weighted_strategy(self):
        fusion = MultimodalFusion(embed_dim=256, strategy="weighted")
        embeddings = {
            "image": torch.randn(4, 256),
            "text": torch.randn(4, 256),
        }
        out = fusion.forward(embeddings)
        assert out.shape == (4, 256)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            MultimodalFusion(embed_dim=256, strategy="invalid")

    def test_missing_modalities(self):
        fusion = MultimodalFusion(embed_dim=256, strategy="concat")
        embeddings = {"image": torch.randn(4, 256)}
        out = fusion.forward(embeddings)
        assert out.shape == (4, 256)

    def test_set_modality_projector(self):
        fusion = MultimodalFusion(embed_dim=256, strategy="concat")
        fusion.set_modality_projector("image", torch.nn.Linear(256, 256))
        embeddings = {"image": torch.randn(4, 256)}
        out = fusion.forward(embeddings)
        assert out.shape == (4, 256)


class TestProjectionHead:
    def test_forward_default(self):
        proj = ProjectionHead(input_dim=512, output_dim=128)
        x = torch.randn(4, 512)
        out = proj(x)
        assert out.shape == (4, 128)

    def test_normalized_output(self):
        proj = ProjectionHead(input_dim=512, output_dim=128, normalize_output=True)
        x = torch.randn(4, 512)
        out = proj(x)
        assert torch.allclose(out.norm(p=2, dim=1), torch.ones(4), atol=1e-5)

    def test_no_normalization(self):
        proj = ProjectionHead(input_dim=512, output_dim=128, normalize_output=False)
        x = torch.randn(4, 512)
        out = proj(x)
        assert not torch.allclose(out.norm(p=2, dim=1), torch.ones(4), atol=1e-5)

    def test_single_layer(self):
        proj = ProjectionHead(input_dim=512, output_dim=128, num_layers=1)
        x = torch.randn(4, 512)
        out = proj(x)
        assert out.shape == (4, 128)

    def test_three_layers(self):
        proj = ProjectionHead(
            input_dim=512, output_dim=128, num_layers=3, hidden_dim=256
        )
        x = torch.randn(4, 512)
        out = proj(x)
        assert out.shape == (4, 128)

    def test_no_layer_norm(self):
        proj = ProjectionHead(
            input_dim=512, output_dim=128, num_layers=2, use_layer_norm=False
        )
        x = torch.randn(4, 512)
        out = proj(x)
        assert out.shape == (4, 128)

    def test_properties(self):
        proj = ProjectionHead(input_dim=512, output_dim=128)
        assert proj.input_dim == 512
        assert proj.output_dim == 128
        assert proj.num_parameters > 0


class TestClassifierHead:
    def test_forward_shape(self):
        clf = ClassifierHead(input_dim=512, num_classes=10)
        x = torch.randn(4, 512)
        out = clf(x)
        assert out.shape == (4, 10)

    def test_predict_softmax(self):
        clf = ClassifierHead(input_dim=512, num_classes=10)
        x = torch.randn(4, 512)
        probs = clf.predict(x)
        assert probs.shape == (4, 10)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_predict_classes(self):
        clf = ClassifierHead(input_dim=512, num_classes=10)
        x = torch.randn(4, 512)
        classes = clf.predict_classes(x)
        assert classes.shape == (4,)

    def test_with_hidden_layers(self):
        clf = ClassifierHead(
            input_dim=512, num_classes=10, hidden_dims=[256, 128], dropout=0.3
        )
        x = torch.randn(4, 512)
        out = clf(x)
        assert out.shape == (4, 10)

    def test_with_layer_norm(self):
        clf = ClassifierHead(
            input_dim=512, num_classes=10, hidden_dims=[256], use_layer_norm=True
        )
        x = torch.randn(4, 512)
        out = clf(x)
        assert out.shape == (4, 10)

    def test_no_hidden_layers(self):
        clf = ClassifierHead(input_dim=512, num_classes=10)
        x = torch.randn(4, 512)
        out = clf(x)
        assert out.shape == (4, 10)

    def test_properties(self):
        clf = ClassifierHead(input_dim=512, num_classes=10)
        assert clf.input_dim == 512
        assert clf.num_classes == 10

    def test_forward_pass_gradient_flow(self):
        clf = ClassifierHead(input_dim=512, num_classes=10)
        x = torch.randn(4, 512, requires_grad=True)
        out = clf(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestClassificationLoss:
    def test_forward_shape(self):
        loss_fn = ClassificationLoss()
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_label_smoothing(self):
        loss_fn = ClassificationLoss(label_smoothing=0.1)
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_class_weights(self):
        weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        loss_fn = ClassificationLoss(class_weights=weights)
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_reduction_none(self):
        loss_fn = ClassificationLoss(reduction="none")
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, targets)
        assert loss.shape == (4,)


class TestContrastiveLoss:
    def test_with_labels(self):
        loss_fn = ContrastiveLoss(temperature=0.07)
        embeddings = torch.randn(8, 128)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(embeddings, labels)
        assert loss.ndim == 0
        assert loss > 0

    def test_without_labels(self):
        loss_fn = ContrastiveLoss(temperature=0.07)
        embeddings = torch.randn(8, 128)
        loss = loss_fn(embeddings)
        assert loss.ndim == 0

    def test_with_margin(self):
        loss_fn = ContrastiveLoss(temperature=0.07, margin=0.1)
        embeddings = torch.randn(8, 128)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(embeddings, labels)
        assert loss.ndim == 0

    def test_temperature_effect(self):
        loss_high = ContrastiveLoss(temperature=0.5)
        loss_low = ContrastiveLoss(temperature=0.01)
        embeddings = torch.randn(8, 128)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        lh = loss_high(embeddings, labels)
        ll = loss_low(embeddings, labels)
        assert lh != ll


class TestEmbeddingSimilarityLoss:
    def test_cosine_similarity(self):
        loss_fn = EmbeddingSimilarityLoss(similarity_type="cosine")
        emb_a = torch.randn(4, 128)
        emb_b = torch.randn(4, 128)
        loss = loss_fn(emb_a, emb_b)
        assert loss.ndim == 0

    def test_mse_similarity(self):
        loss_fn = EmbeddingSimilarityLoss(similarity_type="mse")
        emb_a = torch.randn(4, 128)
        emb_b = torch.randn(4, 128)
        loss = loss_fn(emb_a, emb_b)
        assert loss.ndim == 0

    def test_l1_similarity(self):
        loss_fn = EmbeddingSimilarityLoss(similarity_type="l1")
        emb_a = torch.randn(4, 128)
        emb_b = torch.randn(4, 128)
        loss = loss_fn(emb_a, emb_b)
        assert loss.ndim == 0

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            EmbeddingSimilarityLoss(similarity_type="invalid")

    def test_identical_embeddings_cosine(self):
        loss_fn = EmbeddingSimilarityLoss(similarity_type="cosine")
        emb = torch.randn(4, 128)
        loss = loss_fn(emb, emb)
        assert loss.item() <= 0.0

    def test_with_targets(self):
        loss_fn = EmbeddingSimilarityLoss(similarity_type="cosine")
        emb_a = torch.randn(4, 128)
        emb_b = torch.randn(4, 128)
        targets = torch.ones(4)
        loss = loss_fn(emb_a, emb_b, targets)
        assert loss.ndim == 0


class TestModelFactory:
    def test_create_image_encoder(self):
        encoder = ModelFactory.create_encoder(
            "image", embedding_dim=256, pretrained=False
        )
        assert isinstance(encoder, ImageEncoder)
        assert encoder.embedding_dim == 256

    def test_create_text_encoder(self):
        encoder = ModelFactory.create_encoder(
            "text", embedding_dim=128, vocab_size=1000
        )
        assert isinstance(encoder, TextEncoder)
        assert encoder.embedding_dim == 128

    def test_create_audio_encoder(self):
        encoder = ModelFactory.create_encoder("audio", embedding_dim=64)
        assert isinstance(encoder, AudioEncoder)
        assert encoder.embedding_dim == 64

    def test_create_sensor_encoder(self):
        encoder = ModelFactory.create_encoder("sensor", embedding_dim=64)
        assert isinstance(encoder, SensorEncoder)
        assert encoder.embedding_dim == 64

    def test_create_invalid_encoder(self):
        with pytest.raises(ValueError):
            ModelFactory.create_encoder("invalid_modality")

    def test_create_concat_fusion(self):
        fusion = ModelFactory.create_fusion(strategy="concat")
        assert isinstance(fusion, MultimodalFusion)

    def test_create_attention_fusion(self):
        fusion = ModelFactory.create_fusion(
            strategy="attention", embed_dim=256, num_heads=4
        )
        assert isinstance(fusion, MultimodalFusion)

    def test_create_weighted_fusion(self):
        fusion = ModelFactory.create_fusion(strategy="weighted", embed_dim=256)
        assert isinstance(fusion, MultimodalFusion)

    def test_create_projection_head(self):
        proj = ModelFactory.create_projection_head(input_dim=512, output_dim=128)
        assert isinstance(proj, ProjectionHead)
        assert proj.output_dim == 128

    def test_create_classifier(self):
        clf = ModelFactory.create_classifier(input_dim=128, num_classes=10)
        assert isinstance(clf, ClassifierHead)
        assert clf.num_classes == 10

    def test_registry_and_get_encoder_class(self):
        assert "image" in _ENCODER_REGISTRY
        assert get_encoder_class("image") == ImageEncoder

    def test_get_encoder_class_invalid(self):
        with pytest.raises(ValueError):
            get_encoder_class("nonexistent")

    def test_register_encoder_decorator(self):
        @register_encoder("test_modality")
        class TestEncoder(BaseEncoder):
            def __init__(self, embedding_dim=64, output_dim=None):
                super().__init__(embedding_dim=embedding_dim, output_dim=output_dim)
                self.net = torch.nn.Linear(10, self._output_dim)

            def encode(self, x, **kwargs):
                return self.net(x)

        assert "test_modality" in _ENCODER_REGISTRY
        assert get_encoder_class("test_modality") == TestEncoder


class TestInitialization:
    def test_xavier_uniform(self):
        layer = torch.nn.Linear(10, 20)
        xavier_uniform(layer)
        assert layer.weight is not None

    def test_xavier_normal(self):
        layer = torch.nn.Linear(10, 20)
        xavier_normal(layer)
        assert layer.weight is not None

    def test_kaiming_uniform(self):
        layer = torch.nn.Linear(10, 20)
        kaiming_uniform(layer)
        assert layer.weight is not None

    def test_normal_init(self):
        layer = torch.nn.Linear(10, 20)
        normal_init(layer, mean=0.0, std=0.02)
        assert layer.weight is not None

    def test_orthogonal_init(self):
        layer = torch.nn.Linear(10, 20)
        orthogonal_init(layer)
        assert layer.weight is not None

    def test_initialize_weights_model(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5),
        )
        initialize_weights(model, init_type="xavier_uniform")
        assert model[0].weight is not None

    def test_initialize_weights_skip_modules(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.Linear(20, 5),
        )
        initialize_weights(model, init_type="xavier_uniform", skip_modules=["0"])
        assert model[0].weight is not None

    def test_get_initializer(self):
        fn = get_initializer("kaiming_uniform")
        assert callable(fn)

    def test_get_initializer_invalid(self):
        with pytest.raises(ValueError):
            get_initializer("nonexistent_initializer")

    def test_register_initializer(self):
        def my_init(m):
            pass

        register_initializer("my_init", my_init)
        assert "my_init" in _INITIALIZERS


class TestUtils:
    def test_count_parameters(self):
        model = torch.nn.Linear(10, 5)
        total = count_parameters(model, trainable_only=False)
        assert total == 10 * 5 + 5

    def test_count_parameters_trainable(self):
        model = torch.nn.Linear(10, 5)
        model.weight.requires_grad = False
        trainable = count_parameters(model, trainable_only=True)
        assert trainable == 5

    def test_get_device_default(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_get_device_string(self):
        device = get_device("cpu")
        assert device == torch.device("cpu")

    def test_get_device_torch_device(self):
        device = get_device(torch.device("cpu"))
        assert device == torch.device("cpu")

    def test_estimate_memory_usage(self):
        model = torch.nn.Linear(100, 50)
        mem = estimate_memory_usage(model)
        assert "parameters_mb" in mem
        assert "gradients_mb" in mem
        assert mem["parameters_mb"] > 0

    def test_estimate_memory_with_input(self):
        model = torch.nn.Linear(100, 50)
        mem = estimate_memory_usage(model, input_shape=(1, 100))
        assert mem["estimate_forward_mb"] > 0

    def test_timer(self):
        timer = Timer("test", log_on_exit=False)
        with timer:
            pass
        assert timer.elapsed >= 0

    def test_log_model_summary(self, caplog):
        import logging

        caplog.set_level(logging.INFO)
        model = torch.nn.Linear(10, 5)
        log_model_summary(model)
        assert any("Model:" in record.message for record in caplog.records)


class TestEndToEnd:
    def test_encoder_fusion_classifier_pipeline(self):
        batch_size = 4
        img_enc = ImageEncoder(embedding_dim=256, pretrained=False)
        txt_enc = TextEncoder(
            vocab_size=1000,
            embedding_dim=128,
            num_heads=4,
            num_layers=2,
            hidden_dim=256,
        )
        fusion = ConcatFusion()
        proj = ProjectionHead(input_dim=384, output_dim=64)
        clf = ClassifierHead(input_dim=64, num_classes=5)

        images = torch.randn(batch_size, 3, 224, 224)
        texts = torch.randint(0, 100, (batch_size, 20))

        img_emb = img_enc(images)
        txt_emb = txt_enc(texts)
        embeddings = {"image": img_emb, "text": txt_emb}
        fused = fusion.forward(embeddings)
        projected = proj(fused)
        logits = clf(projected)

        assert img_emb.shape == (batch_size, 256)
        assert txt_emb.shape == (batch_size, 128)
        assert fused.shape == (batch_size, 384)
        assert projected.shape == (batch_size, 64)
        assert logits.shape == (batch_size, 5)

    def test_factory_pipeline(self):
        img_enc = ModelFactory.create_encoder(
            "image", embedding_dim=256, pretrained=False
        )
        txt_enc = ModelFactory.create_encoder(
            "text", embedding_dim=128, vocab_size=1000
        )
        fusion = ModelFactory.create_fusion(strategy="concat")
        proj = ModelFactory.create_projection_head(input_dim=384, output_dim=64)
        clf = ModelFactory.create_classifier(input_dim=64, num_classes=5)

        assert isinstance(img_enc, ImageEncoder)
        assert isinstance(txt_enc, TextEncoder)
        assert isinstance(proj, ProjectionHead)
        assert isinstance(clf, ClassifierHead)

    def test_freeze_part_of_pipeline(self):
        encoder = ImageEncoder(embedding_dim=256, pretrained=False)
        encoder.freeze()
        for param in encoder.parameters():
            assert not param.requires_grad
        encoder.unfreeze()
        assert all(p.requires_grad for p in encoder.parameters())

    def test_save_load_classifier(self):
        clf = ClassifierHead(input_dim=64, num_classes=5)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            clf.save(path)
            loaded = ClassifierHead(input_dim=64, num_classes=5)
            loaded.load(path)
            x = torch.randn(2, 64)
            torch.allclose(clf(x), loaded(x))
        finally:
            Path(path).unlink(missing_ok=True)
