from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class SeqWindowDataset(Dataset):
    """Dataset returning fixed length windows of sequential features."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, L: int):
        assert len(features) == len(labels), "features and labels must match"
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.L = L

    def __len__(self) -> int:
        return len(self.features) - self.L + 1

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.features[idx : idx + self.L])
        y = torch.tensor(self.labels[idx + self.L // 2], dtype=torch.long)
        return x, y


def collate_windows(batch: Sequence[tuple[torch.Tensor, torch.Tensor]]):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)
