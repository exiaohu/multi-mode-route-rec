from typing import Tuple, List

import torch
import dgl
from dgl import laplacian_lambda_max, broadcast_nodes, function as fn
from scipy.sparse.linalg import ArpackNoConvergence
from torch import nn, Tensor
from torch.nn import functional as F, init


class ChebConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k,
                 bias=True):
        super(ChebConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.ModuleList([
            nn.Linear(in_feats, out_feats, bias=False) for _ in range(k)
        ])
        self._k = k
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.bias is not None:
            init.zeros_(self.bias)
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight, init.calculate_gain('relu'))
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, graph: dgl.DGLGraph, feat, lambda_max=None):
        shp = (len(graph.nodes()),) + tuple(1 for _ in range(feat.dim() - 1))
        with graph.local_scope():
            norm = torch.pow(
                graph.in_degrees().float().clamp(min=1), -0.5).reshape(shp).to(feat.device)
            if lambda_max is None:
                try:
                    lambda_max = laplacian_lambda_max(graph)
                except ArpackNoConvergence:
                    lambda_max = [2.] * graph.batch_size
            if isinstance(lambda_max, list):
                lambda_max = torch.tensor(lambda_max).to(feat.device)
            if lambda_max.dim() < 1:
                lambda_max = lambda_max.unsqueeze(-1)  # (B,) to (B, 1)
            # broadcast from (B, 1) to (N, 1)
            lambda_max = torch.reshape(broadcast_nodes(graph, lambda_max), shp).float()
            # T0(X)
            Tx_0 = feat
            rst = self.fc[0](Tx_0)
            # T1(X)
            if self._k > 1:
                graph.ndata['h'] = Tx_0 * norm
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                h = graph.ndata.pop('h') * norm
                # Λ = 2 * (I - D ^ -1/2 A D ^ -1/2) / lambda_max - I
                #   = - 2(D ^ -1/2 A D ^ -1/2) / lambda_max + (2 / lambda_max - 1) I
                Tx_1 = -2. * h / lambda_max + Tx_0 * (2. / lambda_max - 1)
                rst = rst + self.fc[1](Tx_1)
            # Ti(x), i = 2...k
            for i in range(2, self._k):
                graph.ndata['h'] = Tx_1 * norm
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                h = graph.ndata.pop('h') * norm
                # Tx_k = 2 * Λ * Tx_(k-1) - Tx_(k-2)
                #      = - 4(D ^ -1/2 A D ^ -1/2) / lambda_max Tx_(k-1) +
                #        (4 / lambda_max - 2) Tx_(k-1) -
                #        Tx_(k-2)
                Tx_2 = -4. * h / lambda_max + Tx_1 * (4. / lambda_max - 2) - Tx_0
                rst = rst + self.fc[i](Tx_2)
                Tx_1, Tx_0 = Tx_2, Tx_1
            # add bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 **kwargs):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            **kwargs)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, inputs):
        """
        :param inputs: tensor, [N, C_{in}, L_{in}]
        :return: tensor, [N, C_{out}, L_{out}]
        """
        outputs = super(CausalConv1d, self).forward(F.pad(inputs, [self.__padding, 0]))
        return outputs[:, :, :outputs.shape[-1] - self.__padding]


class ResShortcut(nn.Module):
    def __init__(self, f_in: int, f_out: int):
        super(ResShortcut, self).__init__()
        if f_in > f_out:
            self.linear = nn.Linear(f_in, f_out)
        self.f_in, self.f_out = f_in, f_out

    def forward(self, inputs):
        # residual connection, first map the input to the same shape as output
        if self.f_in > self.f_out:
            return self.linear(inputs)
        elif self.f_in < self.f_out:
            zero_shape = inputs.shape[:-1] + (self.f_out - self.f_in,)
            zeros = torch.zeros(zero_shape, dtype=inputs.dtype, device=inputs.device)
            return torch.cat([inputs, zeros], dim=-1)
        return inputs


class TemporalConvLayer(nn.Module):
    def __init__(self, f_in: int, f_out: int, kernel_size: int):
        super(TemporalConvLayer, self).__init__()
        self.causal_conv = CausalConv1d(f_in, 2 * f_out, kernel_size)
        self.shortcut = ResShortcut(f_in, f_out)

        self.f_in, self.f_out, self.kernel_size = f_in, f_out, kernel_size

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Temporal causal convolution layer.
        :param inputs: tensor, [N, T, F_in]
        :return: tensor, [N, T - kernel_size + 1, F_out]
        """
        n, t, _ = inputs.shape

        x_res = self.shortcut(inputs[:, self.kernel_size - 1:, :])

        # shape => [N, F_in, T]
        inputs = inputs.transpose(1, 2)

        # equal shape => [N, 2 * F_out, T - kernel_size + 1] => [N, T - kernel_size + 1, 2 * F_out]
        outputs = self.causal_conv(inputs).transpose(1, 2)

        return (outputs[..., :self.f_out] + x_res) * torch.sigmoid(outputs[..., self.f_out:])


class SpatialConvLayer(nn.Module):
    def __init__(self, f_in: int, f_out: int, k_hop: int):
        super(SpatialConvLayer, self).__init__()
        self.g_conv = ChebConv(f_in, f_out, k_hop)
        self.shortcut = ResShortcut(f_in, f_out)
        self.act = nn.ReLU(True)

    def forward(self, inputs: Tensor, graph: dgl.DGLGraph) -> Tensor:
        """
        Spatial graph convolution layer.
        :param inputs: tensor, [N, T, F_in]
        :param graph: DGLGraph, with `N` nodes
        :return: tensor, [N, T, F_out]
        """
        x_res = self.shortcut(inputs)
        outputs = self.g_conv(graph, inputs)
        return self.act(outputs + x_res)


class STConvBlock(nn.Module):
    def __init__(self,
                 k_hop: int,
                 t_cnv_krnl_sz: int,
                 channels: Tuple[int, int, int],
                 dropout: float):
        """
        Spatio-temporal convolutional block, which contains two temporal gated convolution layers
        and one spatial graph convolution layer in the middle.
        :param k_hop: length of Chebychev polynomial, i.e., kernel size of spatial convolution
        :param t_cnv_krnl_sz: kernel size of temporal convolution
        :param channels: three integers, define each of the sub-blocks
        :param dropout: dropout
        """
        super(STConvBlock, self).__init__()
        self.dropout = dropout

        f_in, f_m, f_out = channels
        self.t_conv1 = TemporalConvLayer(f_in, f_m, t_cnv_krnl_sz)
        self.s_conv = SpatialConvLayer(f_m, f_m, k_hop)
        self.t_conv2 = TemporalConvLayer(f_m, f_out, t_cnv_krnl_sz)
        # self.ln = nn.LayerNorm([n_node, f_out])

    def forward(self, inputs: Tensor, graph: dgl.DGLGraph) -> Tensor:
        """
        forward of spatio-temporal convolutional block
        :param inputs: tensor, [N, T, F_in]
        :param graph: DGLGraph, with `N` nodes
        :return: tensor, [N, T, F_out]
        """
        outputs = self.t_conv1(inputs)
        outputs = self.s_conv(outputs, graph)
        outputs = self.t_conv2(outputs)
        # outputs = self.ln(outputs)
        return torch.dropout(outputs, p=self.dropout, train=self.training)


class OutputLayer(nn.Module):
    def __init__(self, f_in: int, f_out, t_cnv_krnl_sz: int):
        super(OutputLayer, self).__init__()
        self.out = nn.Sequential(
            TemporalConvLayer(f_in, f_in, t_cnv_krnl_sz),
            # nn.LayerNorm([n_node, f_in]),
            TemporalConvLayer(f_in, f_in, 1),
            nn.Linear(f_in, f_out),
            nn.ReLU(True)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Output layer: temporal convolution layers attach with one fully connected layer,
        which map outputs of the last st_conv block to a single-step prediction.
        :param inputs: tensor, [N, 1, F]
        :return: tensor, [N, 1]
        """
        return self.out(inputs).squeeze(1)


class STGCN(nn.Module):
    def __init__(self,
                 n_history: int = 12,
                 k_hop: int = 3,
                 t_cnv_krnl_sz: int = 3,
                 dims: List[Tuple[int, int, int]] = ((64, 16, 64), (64, 16, 64)),
                 in_dim: int = 2,
                 out_dim: int = 2,
                 n_preds: int = 1,
                 dropout: float = 0.3):
        super(STGCN, self).__init__()
        n_history -= 2 * (t_cnv_krnl_sz - 1) * len(dims)

        assert n_history > 0

        self.out_dim = out_dim
        self.n_preds = n_preds

        self.in_ = nn.Linear(in_dim, dims[0][0])
        self.st_blocks = nn.ModuleList([STConvBlock(k_hop, t_cnv_krnl_sz, dim, dropout) for dim in dims])
        self.out = OutputLayer(dims[-1][-1], n_preds * out_dim, n_history)

    def forward(self, _, inputs: Tensor, graph: dgl.DGLGraph, __=None) -> Tensor:
        """
        STGCN product single step prediction
        :param inputs: tensor, [N, T, F]
        :param graph: DGLGraph, with `N` nodes
        :return: tensor, [N, F]
        """
        outputs = self.in_(inputs)
        for st_block in self.st_blocks:
            outputs = st_block(outputs, graph)
        return self.out(outputs)


class STGCN_multi_step(STGCN):
    def __init__(self, n_preds: int = 12, *args, **kwargs):
        super(STGCN_multi_step, self).__init__(*args, **kwargs)
        self.n_preds = n_preds

    def forward(self, _, inputs: Tensor, graph: dgl.DGLGraph, targets: Tensor = None) -> Tensor:
        outputs = []

        for t in range(self.n_preds):
            output = super(STGCN_multi_step, self).forward(_, inputs, graph)
            outputs.append(output)

            if targets is None:
                inputs = torch.cat([inputs[:, :-1, :], output.unsqueeze(1)], dim=1)
            else:
                inputs = torch.cat([inputs[:, :-1, :], targets[:, t: t + 1, :]], dim=1)

        return torch.stack(outputs, dim=1)


if __name__ == '__main__':
    m = STGCN()
    x = torch.randn(256, 12, 2, dtype=torch.float32)
    g = dgl.DGLGraph()
    g.add_nodes(256)
    g = dgl.add_self_loop(g)
    y = m(None, x, g)
    print(y.shape)
