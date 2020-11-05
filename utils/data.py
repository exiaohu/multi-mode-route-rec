from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader


class ZScoreScaler:
    def __init__(
            self,
            mean=torch.tensor([32.61261681, 0.55228212, 1.74994048], dtype=torch.float32),
            std=torch.tensor([16.81696799, 0.7368644, 1.62378311], dtype=torch.float32)):
        self.mean = mean
        self.std = std

    def transform(self, x: torch.Tensor, nan_val):
        dv = x.device

        mean, std = self.mean[:x.shape[-1]].to(dv), self.std[:x.shape[-1]].to(dv)

        return torch.where(torch.eq(x, nan_val), torch.tensor(0., device=dv), (x - mean) / std)

    def inverse_transform(self, x: torch.Tensor, nan_val):
        dv = x.device

        mean, std = self.mean[:x.shape[-1]].to(dv), self.std[:x.shape[-1]].to(dv)

        return torch.where(torch.eq(x, nan_val), torch.tensor(0., device=dv), x * std + mean)


def get_dataloaders(datasets: Dict[str, Dataset],
                    batch_size: int,
                    num_workers: int = 16,
                    collate_fn=None) -> Dict[str, DataLoader]:
    return {key: DataLoader(dataset=ds,
                            batch_size=batch_size,
                            shuffle=(key == 'train'),
                            num_workers=num_workers,
                            collate_fn=collate_fn) for key, ds in datasets.items()}
