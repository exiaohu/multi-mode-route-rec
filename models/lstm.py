from typing import Tuple

import torch
from torch import nn, Tensor


class FCLSTM(nn.Module):
    def __init__(self,
                 n_hist: int,
                 n_pred: int,
                 hidden_size: int,
                 n_rnn_layers: int,
                 input_dim: int,
                 output_dim: int):
        super(FCLSTM, self).__init__()
        self.n_hist = n_hist
        self.n_pred = n_pred
        self.output_dim = output_dim
        self.encoder = nn.LSTM(input_dim, hidden_size, n_rnn_layers)
        self.decoder = nn.LSTM(output_dim, hidden_size, n_rnn_layers)
        self.projector = nn.Linear(hidden_size, output_dim)

    def forward(self, attr, inputs: Tensor, net, targets: Tensor = None) -> Tensor:
        """
        dynamic convoluitonal recurrent neural network
        :param attr: ignore it, for compatibility with trainer.
        :param inputs: [N, n_hist, input_dim]
        :param net: ignore it, for compatibility with trainer.
        :param targets: exists for training, tensor, [N, n_pred, output_dim]
        :return: tensor, [N, n_pred, input_dim]
        """
        n, _, input_dim = inputs.shape
        inputs = inputs.transpose(0, 1)
        if targets is not None:
            targets = targets.transpose(0, 1)
        h, c = self.encoding(inputs)
        outputs = self.decoding((h, c), targets)
        return outputs.reshape(self.n_pred, n, -1).transpose(0, 1)

    def encoding(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        encoding
        :param inputs: tensor, [n_hist, B * N, input_dim]
        :return: 2-tuple tensor, each with shape [n_rnn_layers, B * N, hidden_size]
        """
        _, (h, c) = self.encoder(inputs)
        return h, c

    def decoding(self, hc: Tuple[Tensor, Tensor], targets: Tensor):
        """
        decoding
        :param hc: 2-tuple tensor, each with shape [n_rnn_layers, B * N, hidden_size]
        :param targets: optional, exists while training, tensor, [n_pred, B, N, output_dim]
        :return: tensor, shape as same as targets
        """
        h, c = hc
        decoder_input = torch.zeros(1, h.shape[1], self.output_dim, device=h.device, dtype=h.dtype)

        outputs = list()
        for t in range(self.n_pred):
            decoder_input, (h, c) = self.decoder(decoder_input, (h, c))
            decoder_input = self.projector(decoder_input)
            outputs.append(decoder_input)
            if targets is not None:
                decoder_input = targets[t].unsqueeze(0)
        return torch.cat(outputs)
