from __future__ import annotations

import numpy as np
import pytest
import torch

from app.data.modality import (
    MODALITY_KEYS,
    Modality,
    all_modalities_available,
    available_modalities,
    modality_mask_from_dict,
    modality_mask_to_tensor,
)


class TestModality:
    def test_enum_values(self):
        assert Modality.IMAGE.value == "image"
        assert Modality.TEXT.value == "text"
        assert Modality.AUDIO.value == "audio"
        assert Modality.SENSOR.value == "sensor"

    def test_all_returns_all(self):
        all_mods = Modality.all()
        assert len(all_mods) == 4
        assert Modality.IMAGE in all_mods

    def test_from_str(self):
        assert Modality.from_str("image") == Modality.IMAGE
        assert Modality.from_str("TEXT") == Modality.TEXT
        assert Modality.from_str("Audio") == Modality.AUDIO

    def test_from_str_invalid(self):
        with pytest.raises(ValueError):
            Modality.from_str("unknown")


class TestModalityMask:
    def test_mask_from_full_dict(self):
        sample = {"image": torch.randn(3, 32, 32), "text": torch.randint(0, 100, (50,))}
        mask = modality_mask_from_dict(sample)
        assert mask["image"] is True
        assert mask["text"] is True
        assert mask["audio"] is False
        assert mask["sensor"] is False

    def test_mask_with_missing_flag(self):
        sample = {
            "image": torch.randn(3, 32, 32),
            "text": torch.tensor([]),
            "text_missing": True,
        }
        mask = modality_mask_from_dict(sample)
        assert mask["image"] is True
        assert mask["text"] is False

    def test_mask_to_tensor(self):
        mask = {"image": True, "text": False, "audio": True, "sensor": False}
        tensor = modality_mask_to_tensor(mask)
        assert tensor.dtype == torch.bool
        assert tensor.tolist() == [True, False, True, False]

    def test_available_modalities(self):
        mask = {"image": True, "text": False, "audio": True, "sensor": False}
        avail = available_modalities(mask)
        assert avail == ["image", "audio"]

    def test_missing_modalities(self):
        mask = {"image": True, "text": False, "audio": False, "sensor": True}
        missing = [m for m in MODALITY_KEYS if not mask[m]]
        assert missing == ["text", "audio"]

    def test_all_available_true(self):
        mask = {m: True for m in MODALITY_KEYS}
        assert all_modalities_available(mask) is True

    def test_all_available_false(self):
        mask = {m: True for m in MODALITY_KEYS}
        mask["text"] = False
        assert all_modalities_available(mask) is False
