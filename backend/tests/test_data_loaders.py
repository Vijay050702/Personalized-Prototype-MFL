from __future__ import annotations

import numpy as np
import pytest
import torch

from app.data.dataloaders import AudioLoader, ImageLoader, SensorLoader, TextLoader


class TestImageLoader:
    def test_load_from_numpy(self):
        loader = ImageLoader()
        arr = np.random.randn(3, 64, 64).astype(np.float32)
        tensor = loader.load(arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)

    def test_load_from_tensor(self):
        loader = ImageLoader()
        t = torch.randn(3, 32, 32)
        result = loader.load(t)
        assert torch.equal(result, t.float())

    def test_invalid_type(self):
        loader = ImageLoader()
        with pytest.raises(TypeError):
            loader.load([1, 2, 3])

    def test_ensure_channels_2d_to_3ch(self):
        loader = ImageLoader()
        t = torch.randn(32, 32)
        result = loader.ensure_channels(t, channels=3)
        assert result.shape == (3, 32, 32)

    def test_ensure_channels_1ch_to_3ch(self):
        loader = ImageLoader()
        t = torch.randn(1, 32, 32)
        result = loader.ensure_channels(t, channels=3)
        assert result.shape == (3, 32, 32)

    def test_ensure_shape(self):
        loader = ImageLoader()
        t = torch.randn(3, 64, 64)
        result = loader.ensure_shape(t, target_shape=(3, 32, 32))
        assert result.shape == (3, 32, 32)


class TestTextLoader:
    def test_load_from_numpy(self):
        loader = TextLoader()
        arr = np.array([1, 2, 3, 4, 5])
        tensor = loader.load(arr)
        assert tensor.dtype == torch.long
        assert tensor.tolist() == [1, 2, 3, 4, 5]

    def test_load_from_list(self):
        loader = TextLoader()
        tensor = loader.load([10, 20, 30])
        assert tensor.tolist() == [10, 20, 30]

    def test_truncate(self):
        loader = TextLoader()
        t = torch.tensor([1, 2, 3, 4, 5])
        result = loader.truncate(t, max_length=3)
        assert result.tolist() == [1, 2, 3]

    def test_pad_or_truncate_longer(self):
        loader = TextLoader()
        t = torch.tensor([1, 2, 3, 4, 5])
        result = loader.pad_or_truncate(t, length=3)
        assert result.tolist() == [1, 2, 3]

    def test_pad_or_truncate_shorter(self):
        loader = TextLoader()
        t = torch.tensor([1, 2])
        result = loader.pad_or_truncate(t, length=5)
        assert result.tolist() == [1, 2, 0, 0, 0]


class TestAudioLoader:
    def test_load_from_numpy(self):
        loader = AudioLoader()
        arr = np.random.randn(16000).astype(np.float32)
        tensor = loader.load(arr)
        assert tensor.ndim == 2

    def test_resample(self):
        loader = AudioLoader()
        t = torch.randn(1, 16000)
        result = loader.resample(t, orig_sr=16000, target_sr=8000)
        assert result.size(-1) == 8000

    def test_resample_same_sr(self):
        loader = AudioLoader()
        t = torch.randn(1, 16000)
        result = loader.resample(t, orig_sr=16000, target_sr=16000)
        assert result.size(-1) == 16000

    def test_ensure_length_truncate(self):
        loader = AudioLoader()
        t = torch.randn(1, 16000)
        result = loader.ensure_length(t, target_length=8000)
        assert result.size(-1) == 8000

    def test_ensure_length_pad(self):
        loader = AudioLoader()
        t = torch.randn(1, 8000)
        result = loader.ensure_length(t, target_length=16000)
        assert result.size(-1) == 16000

    def test_invalid_type(self):
        loader = AudioLoader()
        with pytest.raises(TypeError):
            loader.load("not_a_tensor")


class TestSensorLoader:
    def test_load_from_numpy(self):
        loader = SensorLoader()
        arr = np.random.randn(128, 6).astype(np.float32)
        tensor = loader.load(arr)
        assert tensor.shape == (128, 6)

    def test_1d_input_becomes_2d(self):
        loader = SensorLoader()
        arr = np.random.randn(561).astype(np.float32)
        tensor = loader.load(arr)
        assert tensor.ndim == 2

    def test_ensure_windowed(self):
        loader = SensorLoader()
        t = torch.randn(100, 6)
        windows = loader.ensure_windowed(t, window_size=20, stride=10)
        assert windows.size(0) == 9  # (100-20)//10 + 1

    def test_normalize_channels_2d(self):
        loader = SensorLoader()
        t = torch.randn(128, 6)
        result = loader.normalize_channels(t)
        assert result.shape == t.shape

    def test_normalize_channels_1d(self):
        loader = SensorLoader()
        t = torch.randn(561)
        result = loader.normalize_channels(t)
        assert result.shape == (561,)
