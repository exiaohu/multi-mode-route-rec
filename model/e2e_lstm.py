from typing import Tuple, List

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence

from .mlp import MLP


class E2ELSTM(nn.Module):
    def __init__(self,
                 in_dim: int,
                 rnn_hid_units: int,
                 out_hids: List[int],
                 output_dim: int):
        super(E2ELSTM, self).__init__()
        self.encoder = nn.LSTM(in_dim, rnn_hid_units)
        self.projector = MLP(rnn_hid_units * 2, out_hids, output_dim)

    def forward(self, inputs: PackedSequence) -> Tensor:
        h = torch.cat(self.encoding(self.encoder, inputs), -1).squeeze(0)
        return self.projector(h)

    def encoding(self, encoder, inputs: PackedSequence) -> Tuple[Tensor, Tensor]:
        """
        encoding
        :param inputs: tensor, [n_hist, B * N, input_dim]
        :return: 2-tuple tensor, each with shape [n_rnn_layers, B * N, hidden_size]
        """
        _, (h, c) = encoder(inputs)
        return h, c
