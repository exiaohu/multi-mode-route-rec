from typing import Tuple, List

import torch
from torch import nn, Tensor

from .mlp import MLP


class LSTM(nn.Module):
    def __init__(self,
                 n_hist1: int,
                 n_hist2: int,
                 rnn_hid_units: int,
                 feat_dim: int,
                 hid_dims: List[int],
                 hid_feat_dim: int,
                 output_dim: int):
        super(LSTM, self).__init__()
        self.n_hist1 = n_hist1
        self.n_hist2 = n_hist2
        self.output_dim = output_dim
        self.encoder1 = nn.LSTM(1, rnn_hid_units)
        self.encoder2 = nn.LSTM(1, rnn_hid_units)
        self.feat_trans = MLP(feat_dim, hid_dims, hid_feat_dim)
        self.projector = nn.Linear(4 * rnn_hid_units + hid_feat_dim, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        seq1, seq2, feat = inputs.split_with_sizes([8, 8, 2], dim=-1)
        b, len1 = seq1.shape
        _, len2 = seq2.shape
        seq1 = seq1.transpose(0, 1).reshape(len1, b, -1)
        seq2 = seq2.transpose(0, 1).reshape(len2, b, -1)
        h1 = torch.cat(self.encoding(self.encoder1, seq1), -1).squeeze(0)  # [B, 16]
        h2 = torch.cat(self.encoding(self.encoder2, seq2), -1).squeeze(0)  # [B, 16]
        hid_feat = self.feat_trans(feat)  # [B, 16]
        return self.projector(torch.cat([h1, h2, hid_feat], -1))

    def encoding(self, encoder, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        encoding
        :param inputs: tensor, [n_hist, B * N, input_dim]
        :return: 2-tuple tensor, each with shape [n_rnn_layers, B * N, hidden_size]
        """
        _, (h, c) = encoder(inputs)
        return h, c
