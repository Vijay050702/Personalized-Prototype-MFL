from __future__ import annotations

from typing import Any

import numpy as np

from app.datasets.base import BasePreprocessor
from app.datasets.transforms import to_spectrogram


class AudioPreprocessor(BasePreprocessor):
    def __init__(
        self,
        target_sr: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 128,
        max_duration: float = 5.0,
    ):
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_duration = max_duration

    def fit(self, samples: list[dict[str, np.ndarray]]) -> None:
        pass

    def process(self, data: np.ndarray) -> np.ndarray:
        audio = np.asarray(data, dtype=np.float64)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        max_samples = int(self.target_sr * self.max_duration)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)), mode="constant")
        return to_spectrogram(audio, self.n_fft, self.hop_length, self.n_mels)

    def __repr__(self) -> str:
        return f"AudioPreprocessor(sr={self.target_sr}, n_mels={self.n_mels})"
