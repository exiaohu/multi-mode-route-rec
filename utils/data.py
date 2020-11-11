from typing import Dict

import numpy as np
import pandas as pd
import torch
from functools import wraps
from torch.utils.data import Dataset, DataLoader


def scalar_method_wrapper(method):
    @wraps(method)
    def _impl(self, x, nan_val):
        res_type, context = 'tensor', None
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
            res_type = 'ndarray'

        if isinstance(self.mean, torch.Tensor) and isinstance(self.std, torch.Tensor):
            mean, std = self.mean[:x.shape[-1]].to(x.device), self.std[:x.shape[-1]].to(x.device)
        else:
            mean, std = self.mean, self.std

        res = method(self, x, nan_val, mean, std)

        if res_type == 'ndarray':
            return res.cpu().numpy()
        else:
            return res

    return _impl


class ZScoreScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @scalar_method_wrapper
    def transform(self, x: torch.Tensor, nan_val, mean, std):
        return torch.where(torch.eq(x, nan_val), torch.tensor(0., device=x.device), (x - mean) / std)

    @scalar_method_wrapper
    def inverse_transform(self, x: torch.Tensor, nan_val, mean, std):
        return torch.where(torch.eq(x, nan_val), torch.tensor(0., device=x.device), x * std + mean)


def get_dataloaders(datasets: Dict[str, Dataset],
                    batch_size: int,
                    num_workers: int = 16,
                    collate_fn=None) -> Dict[str, DataLoader]:
    return {key: DataLoader(dataset=ds,
                            batch_size=batch_size,
                            shuffle=(key == 'train'),
                            num_workers=num_workers,
                            collate_fn=collate_fn) for key, ds in datasets.items()}
