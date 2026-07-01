from __future__ import annotations

from typing import Any

import torch

from app.data.modality import MODALITY_KEYS, modality_mask_to_tensor
from app.data.multimodal_batch import MultimodalBatch
from app.data.multimodal_sample import MultimodalSample


def pad_sequence(
    sequences: list[torch.Tensor],
    padding_value: float = 0.0,
    dim: int = 0,
) -> torch.Tensor:
    max_len = max(s.size(dim) for s in sequences)
    result_dims = list(sequences[0].shape)
    result_dims[dim] = max_len

    batch = torch.full(
        (len(sequences), *result_dims),
        padding_value,
        dtype=sequences[0].dtype,
    )
    for i, s in enumerate(sequences):
        slices = [i] + [slice(None)] * s.ndim
        slices[dim + 1] = slice(0, s.size(dim))
        batch[tuple(slices)] = s
    return batch


def stack_tensors(
    tensors: list[torch.Tensor],
    padding_value: float = 0.0,
) -> torch.Tensor:
    shapes = [t.shape for t in tensors]
    if len(set(shapes)) == 1:
        return torch.stack(tensors, dim=0)

    max_ndim = max(t.ndim for t in tensors)
    expanded = []
    for t in tensors:
        while t.ndim < max_ndim:
            t = t.unsqueeze(-1)
        expanded.append(t)

    max_shape = [len(tensors)]
    for d in range(max_ndim):
        max_shape.append(max(t.size(d) for t in expanded))

    batch = torch.full(
        max_shape,
        padding_value,
        dtype=tensors[0].dtype,
    )
    for i, t in enumerate(expanded):
        slices = [i] + [slice(0, t.size(d)) for d in range(max_ndim)]
        batch[tuple(slices)] = t
    return batch


def collate_multimodal_samples(
    samples: list[MultimodalSample],
    padding_value: float = 0.0,
    pad_text: bool = True,
    pad_audio: bool = True,
) -> MultimodalBatch:
    if not samples:
        return MultimodalBatch()

    labels = torch.tensor([s.label for s in samples], dtype=torch.long)
    client_ids = [s.client_id for s in samples]
    sample_ids = [s.sample_id for s in samples]

    masks = torch.stack([s.modality_mask_tensor for s in samples], dim=0)

    tensors: dict[str, torch.Tensor | None] = {}
    for mod in MODALITY_KEYS:
        available = [s.get_tensor(mod) for s in samples if s.has_modality(mod)]
        if not available:
            tensors[mod] = None
            continue

        should_pad = (mod == "text" and pad_text) or (mod == "audio" and pad_audio)

        if should_pad:
            tensors[mod] = pad_sequence(available, padding_value=padding_value)
        else:
            all_same = len(set(t.shape for t in available)) == 1
            if all_same:
                tensors[mod] = torch.stack(available, dim=0)
            else:
                tensors[mod] = stack_tensors(available, padding_value=padding_value)

    batch_metadata: dict[str, Any] = {
        "batch_size": len(samples),
        "available_modalities": [
            mod for mod in MODALITY_KEYS if tensors.get(mod) is not None
        ],
        "modality_counts": {
            mod: sum(1 for s in samples if s.has_modality(mod)) for mod in MODALITY_KEYS
        },
    }

    return MultimodalBatch(
        images=tensors.get("image"),
        text=tensors.get("text"),
        audio=tensors.get("audio"),
        sensor=tensors.get("sensor"),
        labels=labels,
        client_ids=client_ids,
        modality_masks=masks,
        sample_ids=sample_ids,
        metadata=batch_metadata,
    )
