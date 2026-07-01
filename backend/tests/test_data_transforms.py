from __future__ import annotations

import pytest
import torch

from app.data.transforms import (
    AudioTransform,
    ComposeTransform,
    IdentityTransform,
    ImageTransform,
    SensorTransform,
    TextTransform,
    TypeCastTransform,
)


class TestIdentityTransform:
    def test_identity(self):
        t = IdentityTransform()
        x = torch.randn(3, 32)
        assert torch.equal(t(x), x)


class TestTypeCastTransform:
    def test_cast_to_float(self):
        t = TypeCastTransform(dtype=torch.float32)
        x = torch.randint(0, 255, (3, 32), dtype=torch.uint8)
        result = t(x)
        assert result.dtype == torch.float32

    def test_cast_to_long(self):
        t = TypeCastTransform(dtype=torch.long)
        x = torch.randn(3, 32)
        result = t(x)
        assert result.dtype == torch.long


class TestImageTransform:
    def test_default_normalize(self):
        t = ImageTransform()
        x = torch.randn(3, 32, 32)
        result = t(x)
        assert result.dtype == torch.float32
        assert result.shape == x.shape

    def test_no_normalize(self):
        t = ImageTransform(normalize=False)
        x = torch.randn(3, 32, 32)
        result = t(x)
        assert result.dtype == torch.float32
        assert torch.allclose(result, x.float())

    def test_2d_input(self):
        t = ImageTransform()
        x = torch.randn(32, 32)
        result = t(x)
        assert result.shape == (1, 32, 32)


class TestTextTransform:
    def test_cast_to_long(self):
        t = TextTransform()
        x = torch.randn(50)
        result = t(x)
        assert result.dtype == torch.long


class TestAudioTransform:
    def test_default_normalize(self):
        t = AudioTransform()
        x = torch.randn(80, 100)
        result = t(x)
        assert result.dtype == torch.float32
        assert result.shape == x.shape

    def test_no_normalize(self):
        t = AudioTransform(normalize=False)
        x = torch.randn(80, 100)
        result = t(x)
        assert result.dtype == torch.float32
        assert torch.allclose(result, x.float())

    def test_normalize_constant(self):
        t = AudioTransform()
        x = torch.ones(80, 100) * 5.0
        result = t(x)
        assert result.dtype == torch.float32


class TestSensorTransform:
    def test_default_normalize(self):
        t = SensorTransform()
        x = torch.randn(128, 6)
        result = t(x)
        assert result.dtype == torch.float32
        assert result.shape == x.shape

    def test_1d_input(self):
        t = SensorTransform()
        x = torch.randn(561)
        result = t(x)
        assert result.dtype == torch.float32
        assert result.shape == (561,)

    def test_no_normalize(self):
        t = SensorTransform(normalize=False)
        x = torch.randn(128, 6)
        result = t(x)
        assert torch.allclose(result, x.float())


class TestComposeTransform:
    def test_compose(self):
        t = ComposeTransform(
            [
                TypeCastTransform(torch.float32),
                IdentityTransform(),
            ]
        )
        x = torch.randint(0, 255, (3, 32), dtype=torch.uint8)
        result = t(x)
        assert result.dtype == torch.float32

    def test_empty_compose(self):
        t = ComposeTransform([])
        x = torch.randn(3, 32)
        result = t(x)
        assert torch.equal(result, x)
