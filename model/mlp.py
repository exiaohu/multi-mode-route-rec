from typing import List

from torch import nn


class MLP(nn.Module):
    def __init__(self, in_size: int, hid_sizes: List[int], out_size: int):
        super(MLP, self).__init__()
        lins = [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip([in_size] + hid_sizes, hid_sizes + [out_size])]
        acts = [nn.ReLU() for _ in range(len(lins) - 1)]

        modules = list()
        for lin, act in zip(lins, acts):
            modules.append(lin)
            modules.append(act)

        modules.append(lins[-1])

        self.linear = nn.Sequential(*modules)

    def forward(self, x):
        return self.linear(x)
