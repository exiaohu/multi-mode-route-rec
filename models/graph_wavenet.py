import dgl
import torch
from dgl.nn.pytorch import GraphConv
from torch import nn, Tensor
from torch.nn import functional as F


class STLayer(nn.Module):
    def __init__(self, n_residuals: int, n_dilations: int, kernel_size: int, dilation: int, n_skip: int,
                 dropout: float):
        super(STLayer, self).__init__()
        # dilated convolutions
        self.filter_conv = nn.Conv1d(n_residuals, n_dilations, kernel_size=kernel_size, dilation=dilation)
        self.gate_conv = nn.Conv1d(n_residuals, n_dilations, kernel_size=kernel_size, dilation=dilation)

        # 1x1 convolution for residual connection
        self.gconv = GraphConv(n_dilations, n_residuals, 4)
        self.dropout = nn.Dropout(dropout, inplace=True)

        # 1x1 convolution for skip connection
        self.skip_conv = nn.Conv1d(n_dilations, n_skip, kernel_size=1)
        self.bn = nn.BatchNorm1d(n_residuals)

    def forward(self, x: Tensor, skip: Tensor, graph: dgl.DGLGraph):
        residual = x
        # dilated convolution
        _filter = self.filter_conv(residual)
        _filter = torch.tanh(_filter)
        _gate = self.gate_conv(residual)
        _gate = torch.sigmoid(_gate)
        x = _filter * _gate

        # parametrized skip connection
        s = x
        s = self.skip_conv(s)
        skip = skip[:, :, -s.size(-1):]
        skip = s + skip

        x = self.gconv(graph, x.transpose(1, 2)).transpose(1, 2)
        self.dropout(x)

        x = x + residual[:, :, -x.size(-1):]

        x = self.bn(x)
        return x, skip


class STBlock(nn.ModuleList):
    def __init__(self, n_layers: int, kernel_size: int, n_residuals: int, n_dilations: int, n_skips: int,
                 dropout: float):
        super(STBlock, self).__init__()
        for i in range(n_layers):
            self.append(
                STLayer(n_residuals, n_dilations, kernel_size, 2 ** i, n_skips, dropout)
            )

    def forward(self, x: Tensor, skip: Tensor, graph: dgl.DGLGraph):
        for layer in self:
            x, skip = layer(x, skip, graph)

        return x, skip


class StackedSTBlocks(nn.ModuleList):
    def __init__(self, n_blocks, n_layers: int, kernel_size: int, n_residuals: int, n_dilations: int, n_skips: int,
                 dropout: float):
        self.n_skips = n_skips
        super(StackedSTBlocks, self).__init__()
        for _ in range(n_blocks):
            self.append(
                STBlock(n_layers, kernel_size, n_residuals, n_dilations, n_skips, dropout))

    def forward(self, x: Tensor, graph: dgl.DGLGraph):
        n, f, t = x.shape
        skip = torch.zeros(n, self.n_skips, t, dtype=torch.float32, device=x.device)
        for block in self:
            x, skip = block(x, skip, graph)
        return x, skip


class GraphWaveNet(nn.Module):
    def __init__(self,
                 n_in: int = 3,
                 n_out: int = 2,
                 n_pred: int = 12,
                 n_residuals: int = 32,
                 n_dilations: int = 32,
                 n_skips: int = 256,
                 n_ends: int = 512,
                 kernel_size: int = 2,
                 n_blocks: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.3):
        super(GraphWaveNet, self).__init__()
        # n_in = n_in + 2
        self.t_pred = n_pred

        self.receptive_field = n_blocks * (kernel_size - 1) * (2 ** n_layers - 1) + 1

        self.enter = nn.Conv1d(n_in, n_residuals, kernel_size=1)

        self.blocks = StackedSTBlocks(n_blocks, n_layers, kernel_size, n_residuals, n_dilations, n_skips, dropout)

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_skips, n_ends, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(n_ends, n_pred * n_out, kernel_size=1)
        )

    def forward(self, _, inputs: Tensor, graph: dgl.DGLGraph, __=None):
        """
        : params inputs: tensor, [N, T, F]
        """
        inputs = inputs.permute(0, 2, 1)  # shape to: [N, F, T]

        in_len = inputs.size(-1)
        if in_len < self.receptive_field:
            x = F.pad(inputs, [self.receptive_field - in_len, 0])
        else:
            x = inputs

        x = self.enter(x)

        n, c, t = x.shape

        x, skip = self.blocks(x, graph)

        y_ = self.out(skip)

        return y_.reshape(n, self.t_pred, -1)


if __name__ == '__main__':
    m = GraphWaveNet()
    x = torch.rand(256, 12, 3)
    g = dgl.DGLGraph()
    g.add_nodes(256)
    g = dgl.add_self_loop(g)

    y = m(None, x, g)
    print(y.shape)
