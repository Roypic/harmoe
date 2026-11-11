"""Utilities for harmonizing partially overlapping dataset label spaces."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class GlobalVocabulary:
    classes: tuple[str, ...]
    dataset_indices: dict[str, tuple[int, ...]]

    def expand_labels(self, dataset_name: str, local_labels: Tensor) -> Tensor:
        """Map local labels into the global space and fill unannotated entries with -1."""
        indices = self.dataset_indices[dataset_name]
        if local_labels.ndim != 2 or local_labels.shape[1] != len(indices):
            raise ValueError("local label width does not match the dataset vocabulary")
        global_labels = torch.full(
            (local_labels.shape[0], len(self.classes)),
            -1,
            dtype=local_labels.dtype,
            device=local_labels.device,
        )
        global_labels[:, list(indices)] = local_labels
        return global_labels


def build_global_vocabulary(dataset_classes: Mapping[str, Sequence[str]]) -> GlobalVocabulary:
    """Build a stable union vocabulary after caller-supplied synonym normalization."""
    classes: list[str] = []
    class_to_index: dict[str, int] = {}
    dataset_indices: dict[str, tuple[int, ...]] = {}
    for dataset_name, local_classes in dataset_classes.items():
        indices = []
        for raw_name in local_classes:
            name = raw_name.strip().lower()
            if not name:
                raise ValueError("class names cannot be empty")
            if name not in class_to_index:
                class_to_index[name] = len(classes)
                classes.append(name)
            indices.append(class_to_index[name])
        dataset_indices[dataset_name] = tuple(indices)
    return GlobalVocabulary(tuple(classes), dataset_indices)
