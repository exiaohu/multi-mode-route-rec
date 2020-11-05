import math
import random
from typing import Tuple

import dgl
import torch
from dgl.nn.pytorch import GraphConv
from torch import nn, Tensor


class DCGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(DCGRUCell, self).__init__()
        self.hidden_size = hidden_size

        self.ru_gate_g_conv = GraphConv(input_size + hidden_size, hidden_size * 2)
        self.candidate_g_conv = GraphConv(input_size + hidden_size, hidden_size)

    def forward(self, inputs: Tensor, nets: dgl.DGLGraph, states) -> Tuple[Tensor, Tensor]:
        r_u = torch.sigmoid(self.ru_gate_g_conv(nets, torch.cat([inputs, states], -1)))
        r, u = r_u.split(self.hidden_size, -1)
        c = torch.tanh(self.candidate_g_conv(nets, torch.cat([inputs, r * states], -1)))
        outputs = new_state = u * states + (1 - u) * c

        return outputs, new_state


class DCRNNEncoder(nn.ModuleList):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        super(DCRNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.append(DCGRUCell(input_size, hidden_size))
        for _ in range(1, n_layers):
            self.append(DCGRUCell(hidden_size, hidden_size))

    def forward(self, inputs: Tensor, nets: dgl.DGLGraph) -> Tensor:
        """
        :param inputs: tensor, [N, T, input_size]
        :param nets: batched dgl graph, with `N` nodes
        :return: tensor, [n_layers, N, hidden_size]
        """
        n, t, _ = inputs.shape
        dv, dt = inputs.device, inputs.dtype

        states = list(torch.zeros(len(self), n, self.hidden_size, device=dv, dtype=dt))
        inputs = list(inputs.transpose(0, 1))

        for i_layer, cell in enumerate(self):
            for i_t in range(t):
                inputs[i_t], states[i_layer] = cell(inputs[i_t], nets, states[i_layer])
        return torch.stack(states)


class DCRNNDecoder(nn.ModuleList):
    def __init__(self, output_size: int, hidden_size: int, n_layers: int, n_preds: int):
        super(DCRNNDecoder, self).__init__()
        self.output_size = output_size
        self.n_preds = n_preds
        self.append(DCGRUCell(output_size, hidden_size))
        for _ in range(1, n_layers):
            self.append(DCGRUCell(hidden_size, hidden_size))
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, nets: dgl.DGLGraph, states: Tensor,
                targets: Tensor = None, teacher_force: bool = 0.5) -> Tensor:
        """
        :param nets: batched dgl graph, with `N` nodes
        :param states: tensor, [n_layers, N, hidden_size]
        :param targets: None or tensor, [N, T, output_size]
        :param teacher_force: random to use targets as decoder inputs
        :return: tensor, [N, T, output_size]
        """
        n_layers, n, _ = states.shape

        inputs = torch.zeros(n, self.output_size, device=states.device, dtype=states.dtype)

        states = list(states)
        assert len(states) == n_layers

        new_outputs = list()
        for i_t in range(self.n_preds):
            for i_layer in range(n_layers):
                inputs, states[i_layer] = self[i_layer](inputs, nets, states[i_layer])
            inputs = self.out(inputs)
            new_outputs.append(inputs)
            if targets is not None and random.random() < teacher_force:
                inputs = targets[:, i_t]

        return torch.stack(new_outputs, 1)


class DCRNN(nn.Module):
    def __init__(self,
                 n_pred: int = 12,
                 hidden_size: int = 64,
                 n_rnn_layers: int = 2,
                 input_dim: int = 3,
                 output_dim: int = 2,
                 cl_decay_steps: int = 1000):
        super(DCRNN, self).__init__()
        self.cl_decay_steps = cl_decay_steps
        self.encoder = DCRNNEncoder(input_dim, hidden_size, n_rnn_layers)
        self.decoder = DCRNNDecoder(output_dim, hidden_size, n_rnn_layers, n_pred)

    def forward(self, inputs: Tensor, nets: dgl.DGLGraph, targets: Tensor = None,
                batch_seen: int = None) -> Tensor:
        """
        dynamic convolutional recurrent neural network
        :param inputs: [N, n_hist, input_dim]
        :param nets: batched dgl graph, with `N` nodes
        :param targets: exists for training, tensor, [N, n_pred, output_dim]
        :param batch_seen: int, the number of batches the model has seen
        :return: [N, n_pred, output_dim]
        """
        states = self.encoder(inputs, nets)
        outputs = self.decoder(nets, states, targets, self._compute_sampling_threshold(batch_seen))
        return outputs

    def _compute_sampling_threshold(self, batches_seen: int):
        return self.cl_decay_steps / (self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps))
