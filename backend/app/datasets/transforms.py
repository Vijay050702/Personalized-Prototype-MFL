from __future__ import annotations

from typing import Any

import numpy as np


def normalize_mean_std(
    data: np.ndarray,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> np.ndarray:
    if mean is None:
        mean = tuple(data.mean(axis=(0, 1, 2))) if data.ndim == 4 else (data.mean(),)
    if std is None:
        std = tuple(data.std(axis=(0, 1, 2))) if data.ndim == 4 else (data.std(),)
    mean_arr = (
        np.array(mean, dtype=data.dtype).reshape(1, 1, 1, -1)
        if data.ndim == 4
        else np.array(mean, dtype=data.dtype)
    )
    std_arr = (
        np.array(std, dtype=data.dtype).reshape(1, 1, 1, -1)
        if data.ndim == 4
        else np.array(std, dtype=data.dtype)
    )
    std_arr = np.where(std_arr == 0, 1e-8, std_arr)
    return (data - mean_arr) / std_arr


def resize(data: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    h, w = data.shape[:2]
    t_h, t_w = target_size
    if h == t_h and w == t_w:
        return data
    y_ratio = h / t_h
    x_ratio = w / t_w
    y_indices = (np.arange(t_h) * y_ratio).astype(int)
    x_indices = (np.arange(t_w) * x_ratio).astype(int)
    y_indices = np.clip(y_indices, 0, h - 1)
    x_indices = np.clip(x_indices, 0, w - 1)
    return data[y_indices[:, None], x_indices[None, :]]


def pad_sequences(
    sequences: list[np.ndarray],
    max_length: int | None = None,
    padding_value: float = 0.0,
) -> np.ndarray:
    lengths = [len(s) for s in sequences]
    if max_length is None:
        max_length = max(lengths)
    padded = []
    for s in sequences:
        if len(s) >= max_length:
            padded.append(s[:max_length])
        else:
            pad_width = [(0, max_length - len(s))] + [(0, 0)] * (s.ndim - 1)
            padded.append(np.pad(s, pad_width, constant_values=padding_value))
    return np.array(padded)


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[labels]


def normalize_min_max(data: np.ndarray) -> np.ndarray:
    d_min = data.min()
    d_max = data.max()
    if d_max - d_min == 0:
        return np.zeros_like(data)
    return (data - d_min) / (d_max - d_min)


def sliding_window(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    n = len(data)
    indices = np.arange(0, n - window_size + 1, stride)
    return np.array([data[i : i + window_size] for i in indices])


def to_spectrogram(
    audio: np.ndarray, n_fft: int = 512, hop_length: int = 256, n_mels: int = 128
) -> np.ndarray:
    import numpy as np

    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    pad = n_fft - (len(audio) % n_fft)
    if pad < n_fft:
        audio = np.pad(audio, (0, pad), mode="reflect")

    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop_length]
    window = np.hanning(n_fft)
    frames = frames * window

    spectrum = np.fft.rfft(frames, n=n_fft)
    magnitude = np.abs(spectrum)

    mel_bins = np.linspace(0, n_fft // 2, n_mels + 2).astype(int)
    mel_spectrogram = np.zeros((len(frames), n_mels))
    for i in range(n_mels):
        start = mel_bins[i]
        end = mel_bins[i + 2]
        if end > magnitude.shape[1]:
            end = magnitude.shape[1]
        if start >= end:
            continue
        mel_spectrogram[:, i] = magnitude[:, start:end].mean(axis=1)

    mel_spectrogram = np.where(mel_spectrogram == 0, 1e-10, mel_spectrogram)
    log_mel = np.log(mel_spectrogram)
    return log_mel.T


def build_vocabulary(
    texts: list[str], max_vocab_size: int = 10000, min_freq: int = 1
) -> dict[str, int]:
    from collections import Counter

    all_tokens = []
    for t in texts:
        all_tokens.extend(t.lower().split())
    counter = Counter(all_tokens)
    most_common = [
        w for w, c in counter.most_common(max_vocab_size - 2) if c >= min_freq
    ]
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w in most_common:
        vocab[w] = len(vocab)
    return vocab


def text_to_ids(
    texts: list[str], vocab: dict[str, int], max_length: int = 128
) -> np.ndarray:
    unk_id = vocab.get("<UNK>", 1)
    pad_id = vocab.get("<PAD>", 0)
    ids = []
    for t in texts:
        tokens = t.lower().split()
        token_ids = [vocab.get(t, unk_id) for t in tokens[:max_length]]
        token_ids += [pad_id] * (max_length - len(token_ids))
        ids.append(token_ids)
    return np.array(ids, dtype=np.int64)
