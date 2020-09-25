from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class ZScoreScaler:
    def __init__(self, mean: float, std: float):
        assert std > 0
        self.mean = mean
        self.std = std

    def transform(self, x: Tensor, nan_val: float):
        zeros = torch.eq(x, nan_val)
        x = (x - self.mean) / self.std
        x[zeros] = 0.0
        return x

    def inverse_transform(self, x: Tensor, nan_val: float):
        zeros = torch.eq(x, nan_val)
        x = x * self.std + self.mean
        x[zeros] = 0.0
        return x


def get_dataloaders(datasets: Dict[str, Dataset],
                    batch_size: int,
                    num_workers: int = 16,
                    collate_fn=None) -> Dict[str, DataLoader]:
    return {key: DataLoader(dataset=ds,
                            batch_size=batch_size,
                            shuffle=(key == 'train'),
                            num_workers=num_workers,
                            collate_fn=collate_fn) for key, ds in datasets.items()}
