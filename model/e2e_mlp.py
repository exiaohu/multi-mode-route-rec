from typing import List

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

from .mlp import MLP


class E2EMLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hids: List[int],
                 output_dim: int):
        super(E2EMLP, self).__init__()
        self.projector = MLP(in_dim, hids, output_dim)

    def forward(self, inputs: PackedSequence) -> Tensor:
        unpacked, lens = pad_packed_sequence(inputs)  # [T, B, D_in] and [B]
        lengths = unpacked[..., -1:]  # [T, B, 1]
        total_length = torch.sum(lengths, dim=0).squeeze(0)  # [B, 1]
        h = self.projector(unpacked)  # [T, B, D_out]
        time = torch.sum(lengths / h, dim=0).squeeze(0)  # [B, D_out]

        return total_length / time
