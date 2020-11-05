import collections
import itertools
from typing import Tuple, List

import dgl
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .graph import graph_partition
from .helper import timing, Number


def get_datasets(*args, **kwargs):
    return {phase: BJSpeedDataset(phase, *args, **kwargs) for phase in ['train', 'val', 'test']}


class BJSpeedDataset(Dataset):
    class_inited = False

    def __init__(self, phase: str, *args, **kwargs):
        assert phase in ['train', 'val', 'test']
        self.phase = phase

        if not self.class_inited:
            self.init_class(*args, **kwargs)

    @classmethod
    def init_class(
            cls,
            road_states_path: str = '/home/huxiao/data/bj_data/roads-20180801-20180831.parquet',
            road_net_path: str = '/home/huxiao/data/bj_data/bj_roads/bj_roads.shp',
            ratio: Tuple[Number, Number, Number] = (6, 2, 2),
            n_hist: int = 12,
            n_pred: int = 12,
            in_dim: int = 3,
            out_dim: int = 2,
            n_partitions: int = None
    ):
        if not cls.class_inited:
            cls.in_dim = in_dim
            cls.out_dim = out_dim

            cls.n_hist = n_hist
            cls.n_pred = n_pred
            cls.ratio = ratio

            road_states = cls.init_road_states(road_states_path)
            road_attributes = cls.init_road_attributes(road_net_path)
            road_net = cls.init_road_net(road_net_path)

            valid_segments = set(road_states['speed'].index)
            road_attributes = road_attributes.loc[valid_segments]
            road_net = road_net.subgraph(valid_segments)

            partitions = graph_partition(road_net, n_partitions or int(len(road_net.nodes) ** 0.5))

            time_series = road_states['speed'].columns.to_series()
            time_interval = min(j - i for i, j in zip(time_series.index[:-1], time_series.index[1:]))
            sequence = list()
            for ti in time_series.index:
                seq = list(time_series[time_series.between(ti, ti + time_interval * (n_hist + n_pred - 1))])
                if len(seq) == (n_hist + n_pred):
                    sequence.append(seq)

            cls._states = road_states
            cls._attributes = road_attributes
            cls._net: nx.DiGraph = road_net
            cls._partitions = partitions
            cls._sequence = sequence
            cls.class_inited = True

    def __getitem__(self, idx):
        """
        :param idx: index of the sample.
        :return: tuple of (attributes, states_x, states_y, net)
            attributes: Tensor of shape (N, P)
            states_x: Tensor of shape (N, T, D_in)
            states_y: Tensor of shape (N, T, D_out)
            net: dgl graph with `N` nodes
        """
        part_id, seq_id = idx % len(self._partitions), idx // len(self._partitions)

        nodes, seq = sorted(self._partitions[part_id]), self.sequence[seq_id]

        attr = torch.tensor(np.nan_to_num(self._attributes.loc[nodes].to_numpy()), dtype=torch.float32)

        speed = self._states['speed'].loc[nodes, seq].to_numpy()
        available = self._states['available'].loc[nodes, seq].to_numpy()
        total = self._states['total'].loc[nodes, seq].to_numpy()
        states = torch.tensor(np.nan_to_num(np.stack([speed, available, total], axis=-1)), dtype=torch.float32)

        net = self.create_dgl_graph_from_networkx(self._net.subgraph(nodes), nodes)

        return attr, states[:, :self.n_hist, :self.in_dim], states[:, self.n_hist:, :self.out_dim], net

    def __len__(self):
        return len(self._partitions) * len(self.sequence)

    @staticmethod
    def create_dgl_graph_from_networkx(net: nx.DiGraph, nodes: List[int]):
        fro, to = zip(*net.edges())
        fro = list(map(lambda node: nodes.index(node), fro))
        to = list(map(lambda node: nodes.index(node), to))
        ids = list(range(len(nodes)))
        fro, to = zip(*sorted(set(zip(fro + ids, to + ids))))
        return dgl.graph((fro, to), num_nodes=len(nodes))

    @property
    def sequence(self):
        trn, val, tst = self.ratio
        _sum = sum(self.ratio)
        if self.phase == 'train':
            start, end = 0, int(trn / _sum * len(self._sequence))
        elif self.phase == 'val':
            start, end = int(trn / _sum * len(self._sequence)), int((trn + val) / _sum * len(self._sequence))
        else:  # self.phase == 'test'
            start, end = int((trn + val) / _sum * len(self._sequence)), len(self._sequence)
        return self._sequence[start:end]

    @staticmethod
    @timing
    def init_road_states(road_states_path='/home/huxiao/data/bj_data/roads-20180801-20180831.parquet'):
        roads = pd.read_parquet(road_states_path, )
        return {
            'speed': roads.speed.unstack('timestamps'),
            'available': roads.available.unstack('timestamps'),
            'total': roads.total.unstack('timestamps'),
        }

    @staticmethod
    @timing
    def init_road_attributes(road_net_path='/home/huxiao/data/bj_data/bj_roads/bj_roads.shp'):
        columns_used = ['id', 'kind', 'width', 'direction', 'toll', 'length', 'speedclass', 'lanenum']
        road_net = gpd.read_file(road_net_path)[columns_used]

        main_road = road_net.kind.apply(lambda kind: any(map(lambda k: 1 <= int(k[:2], base=16) <= 6, kind.split('|'))))
        road_net = road_net[main_road]

        road_net['id'] = pd.to_numeric(road_net.id)
        road_net = road_net.set_index('id')

        road_net['width'] = pd.to_numeric(road_net.width)
        road_net['direction'] = pd.to_numeric(road_net.direction)
        road_net['toll'] = pd.to_numeric(road_net.toll)
        road_net['length'] = pd.to_numeric(road_net.length)
        road_net['speedclass'] = pd.to_numeric(road_net.speedclass)
        road_net['lanenum'] = pd.to_numeric(road_net.lanenum)

        road_net = pd.merge(road_net, road_net.kind.str.get_dummies(sep='|'), left_index=True, right_index=True)
        road_net = road_net.drop(columns='kind')

        return road_net

    @staticmethod
    @timing
    def init_road_net(road_net_path='/home/huxiao/data/bj_data/bj_roads/bj_roads.shp'):
        used_columns = ['id', 'snodeid', 'enodeid', 'kind']
        roads = gpd.read_file(road_net_path)[used_columns]
        roads = roads[roads.kind.apply(lambda kind: any(map(lambda k: 1 <= int(k[:2], base=16) <= 6, kind.split('|'))))]
        road_segments, road_interactions = set(), set()
        prevs, nexts = collections.defaultdict(set), collections.defaultdict(set)
        road_net = nx.DiGraph()

        for _, road in roads.iterrows():
            nid, sid, eid = int(road.id), int(road.snodeid), int(road.enodeid)
            prevs[eid].add(nid)
            nexts[sid].add(nid)
            road_interactions.add(sid)
            road_interactions.add(eid)
            road_segments.add(nid)
            road_net.add_node(nid)

        for ri in road_interactions:
            road_net.add_edges_from(itertools.product(prevs[ri], nexts[ri]))

        return road_net


def bj_collate_fn(batch):
    assert len(batch) > 0
    attr, states_x, states_y, nets = zip(*batch)
    return (torch.cat(attr), torch.cat(states_x), dgl.batch(nets)), torch.cat(states_y)
