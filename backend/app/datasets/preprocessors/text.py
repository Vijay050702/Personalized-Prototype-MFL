from __future__ import annotations

from typing import Any

import numpy as np

from app.datasets.base import BasePreprocessor
from app.datasets.transforms import build_vocabulary, pad_sequences, text_to_ids


class TextPreprocessor(BasePreprocessor):
    def __init__(
        self,
        max_length: int = 128,
        max_vocab_size: int = 10000,
        min_freq: int = 1,
    ):
        self.max_length = max_length
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}

    def fit(self, samples: list[dict[str, np.ndarray]]) -> None:
        texts = []
        for s in samples:
            for mod, arr in s.items():
                if arr.dtype.kind in ("U", "S", "O"):
                    texts.append(str(arr))
        if not texts:
            return
        self.vocab = build_vocabulary(texts, self.max_vocab_size, self.min_freq)

    def process(self, data: np.ndarray) -> np.ndarray:
        text = str(data) if data.dtype.kind in ("U", "S", "O") else str(data)
        unk_id = self.vocab.get("<UNK>", 1)
        pad_id = self.vocab.get("<PAD>", 0)
        tokens = text.lower().split()
        ids = [self.vocab.get(t, unk_id) for t in tokens[: self.max_length]]
        ids += [pad_id] * (self.max_length - len(ids))
        return np.array(ids, dtype=np.int64)

    def process_batch(self, texts: list[str]) -> np.ndarray:
        return text_to_ids(texts, self.vocab, self.max_length)

    def __repr__(self) -> str:
        return f"TextPreprocessor(vocab_size={len(self.vocab)}, max_length={self.max_length})"
