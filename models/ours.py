import torch
import dgl
from dgl import function as fn
from dgl.nn import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair, DGLError
from torch import nn, Tensor
from torch.nn import functional as F


class GATConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(*h_src.shape[:-1], self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(*h_dst.shape[:-1], self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(*h_src.shape[:-1], self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst


class STLayer(nn.Module):
    def __init__(self, n_residuals: int, n_dilations: int, kernel_size: int, dilation: int, n_skip: int, n_heads: int,
                 dropout: float):
        super(STLayer, self).__init__()
        # dilated convolutions
        self.filter_conv = nn.Conv1d(n_residuals, n_dilations, kernel_size=kernel_size, dilation=dilation)
        self.gate_conv = nn.Conv1d(n_residuals, n_dilations, kernel_size=kernel_size, dilation=dilation)

        # 1x1 convolution for residual connection
        self.gconv = GATConv(n_dilations, n_residuals, n_heads)
        self.agg = nn.Linear(n_residuals * n_heads, n_residuals)
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

        x = x.transpose(1, 2)  # shape => [N, T, C]
        x = self.gconv(graph, x)  # shape => [N, T, n_heads, C]
        x = x.view(*x.shape[:-2], -1)  # shape => [N, T, n_heads * C]
        x = self.agg(x).transpose(1, 2)  # shape => [N, C, T]

        self.dropout(x)

        x = x + residual[:, :, -x.size(-1):]

        x = self.bn(x)
        return x, skip


class STBlock(nn.ModuleList):
    def __init__(self, n_layers: int, kernel_size: int, n_residuals: int, n_dilations: int, n_skips: int, n_heads: int,
                 dropout: float):
        super(STBlock, self).__init__()
        for i in range(n_layers):
            self.append(
                STLayer(n_residuals, n_dilations, kernel_size, 2 ** i, n_skips, n_heads, dropout)
            )

    def forward(self, x: Tensor, skip: Tensor, graph: dgl.DGLGraph):
        for layer in self:
            x, skip = layer(x, skip, graph)

        return x, skip


class StackedSTBlocks(nn.ModuleList):
    def __init__(self, n_blocks, n_layers: int, kernel_size: int, n_residuals: int, n_dilations: int, n_skips: int,
                 n_heads: int, dropout: float):
        self.n_skips = n_skips
        super(StackedSTBlocks, self).__init__()
        for _ in range(n_blocks):
            self.append(
                STBlock(n_layers, kernel_size, n_residuals, n_dilations, n_skips, n_heads, dropout))

    def forward(self, x: Tensor, graph: dgl.DGLGraph):
        n, f, t = x.shape
        skip = torch.zeros(n, self.n_skips, t, dtype=torch.float32, device=x.device)
        for block in self:
            x, skip = block(x, skip, graph)
        return x, skip


class Ours(nn.Module):
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
                 n_heads: int = 4,
                 n_attr: int = 63,
                 use_attr: bool = True,
                 dropout: float = 0.3):
        super(Ours, self).__init__()
        self.n_skips = n_skips
        self.n_ends = n_ends

        # n_in = n_in + 2
        self.t_pred = n_pred

        self.receptive_field = n_blocks * (kernel_size - 1) * (2 ** n_layers - 1) + 1

        self.enter = nn.Conv1d(n_in, n_residuals, kernel_size=1)

        self.blocks = StackedSTBlocks(n_blocks, n_layers, kernel_size, n_residuals,
                                      n_dilations, n_skips, n_heads, dropout)
        if use_attr:
            self.attr = nn.ModuleDict({
                'w': nn.Linear(n_attr, n_skips * n_skips),
                'b': nn.Linear(n_attr, n_skips)
            })

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_skips, n_ends, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(n_ends, n_pred * n_out, kernel_size=1)
        )

    def forward(self, attr: Tensor, inputs: Tensor, graph: dgl.DGLGraph, __=None):
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

        if hasattr(self, 'attr'):
            skip = torch.relu(skip).transpose(1, 2)
            w = self.attr['w'](attr).view(len(graph.nodes()), self.n_skips, self.n_skips)
            b = self.attr['b'](attr).view(len(graph.nodes()), 1, self.n_skips)
            skip = (torch.bmm(skip, w) + b).transpose(1, 2)

        y_ = self.out(skip)

        return y_.reshape(n, self.t_pred, -1)


if __name__ == '__main__':
    m = Ours()
    x = torch.rand(256, 12, 3)
    g = dgl.DGLGraph()
    g.add_nodes(256)
    g = dgl.add_self_loop(g)
    attr = torch.rand(256, 63)
    y = m(attr, x, g)
    print(y.shape)
