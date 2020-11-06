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

            road_states, id_to_rs, rs_to_id, id_to_ts, ts_to_id = cls.init_road_states(road_states_path)
            road_attributes = cls.init_road_attributes(road_net_path, list(rs_to_id))
            road_net = cls.init_road_net(road_net_path, rs_to_id)

            partitions = graph_partition(road_net, n_partitions or int(len(road_net.nodes) ** 0.5))
            sequence = cls.get_valid_sequence(ts_to_id, n_hist, n_pred)

            cls._states = road_states
            cls._id_to_rs = id_to_rs
            cls._rs_to_id = rs_to_id
            cls._id_to_ts = id_to_ts
            cls._ts_to_id = ts_to_id
            cls._attributes = road_attributes
            cls._net: nx.DiGraph = road_net
            cls._partitions = partitions
            cls._sequence = sequence
            cls.class_inited = True

    @staticmethod
    def get_valid_sequence(ts_to_id, n_hist, n_pred):
        time_series = pd.Series(ts_to_id)
        time_interval = min(j - i for i, j in zip(time_series.index[:-1], time_series.index[1:]))
        sequence = list()
        for ti in time_series.index:
            seq = time_series[ti: ti + time_interval * (n_hist + n_pred - 1)]
            if len(seq) == (n_hist + n_pred):
                sequence.append(list(seq))
        return sequence

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

        attr = torch.tensor(self._attributes[nodes], dtype=torch.float32)

        states = torch.tensor(self._states[nodes, :][:, seq], dtype=torch.float32)

        net = self.create_dgl_graph_from_networkx(self._net.subgraph(nodes), nodes)

        return attr, states[:, :self.n_hist, :self.in_dim], states[:, self.n_hist:, :self.out_dim], net

    def __len__(self):
        return len(self._partitions) * len(self.sequence)

    @staticmethod
    def create_dgl_graph_from_networkx(net: nx.DiGraph, nodes: List[int]):
        fro, to = zip(*net.edges())
        nodes = {node: i for i, node in enumerate(nodes)}
        ids = list(range(len(nodes)))
        fro = torch.tensor(list(map(lambda node: nodes[node], fro)) + ids)
        to = torch.tensor(list(map(lambda node: nodes[node], to)) + ids)
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
    def init_road_states(road_states_path):
        roads = pd.read_parquet(road_states_path)

        _, n_feat = roads.shape
        timestamps, segments = roads.index.levels

        roads = roads.unstack('timestamps')
        states = np.reshape(roads.to_numpy(dtype=np.float32), (len(segments), n_feat, len(timestamps)))

        id_to_rs, rs_to_id = {i: rs for i, rs in enumerate(segments)}, {rs: i for i, rs in enumerate(segments)}
        id_to_ts, ts_to_id = {i: ts for i, ts in enumerate(timestamps)}, {ts: i for i, ts in enumerate(timestamps)}

        return np.nan_to_num(states.transpose((0, 2, 1))), id_to_rs, rs_to_id, id_to_ts, ts_to_id

    @staticmethod
    @timing
    def init_road_attributes(road_net_path, valid_road_segments):
        columns_used = ['id', 'kind', 'width', 'direction', 'toll', 'length', 'speedclass', 'lanenum']
        roads = gpd.read_file(road_net_path)[columns_used]

        roads['id'] = pd.to_numeric(roads.id)
        roads = roads.set_index('id').loc[valid_road_segments]

        main_road = roads.kind.apply(lambda kind: any(map(lambda k: 1 <= int(k[:2], base=16) <= 6, kind.split('|'))))
        roads = roads[main_road]

        roads['width'] = pd.to_numeric(roads.width)
        roads['direction'] = pd.to_numeric(roads.direction)
        roads['toll'] = pd.to_numeric(roads.toll)
        roads['length'] = pd.to_numeric(roads.length)
        roads['speedclass'] = pd.to_numeric(roads.speedclass)
        roads['lanenum'] = pd.to_numeric(roads.lanenum)

        roads = pd.merge(roads, roads.kind.str.get_dummies(sep='|'), left_index=True, right_index=True)
        roads = roads.drop(columns='kind')

        return roads.to_numpy()

    @staticmethod
    @timing
    def init_road_net(road_net_path, rs_to_id):
        used_columns = ['id', 'snodeid', 'enodeid', 'kind']
        roads = gpd.read_file(road_net_path)[used_columns]

        roads['id'] = pd.to_numeric(roads.id)
        roads = roads.set_index('id').loc[list(rs_to_id)]

        roads = roads[roads.kind.apply(lambda kind: any(map(lambda k: 1 <= int(k[:2], base=16) <= 6, kind.split('|'))))]

        road_segments, road_interactions = set(), set()
        prevs, nexts = collections.defaultdict(set), collections.defaultdict(set)

        road_net = nx.DiGraph()

        for nid, road in roads.iterrows():
            sid, eid = int(road.snodeid), int(road.enodeid)
            prevs[eid].add(nid)
            nexts[sid].add(nid)
            road_interactions.add(sid)
            road_interactions.add(eid)
            road_segments.add(nid)
            road_net.add_node(nid)

        for ri in road_interactions:
            road_net.add_edges_from(itertools.product(prevs[ri], nexts[ri]))

        return nx.relabel_nodes(road_net.subgraph(rs_to_id.keys()), rs_to_id)


def bj_collate_fn(batch):
    assert len(batch) > 0
    attr, states_x, states_y, nets = zip(*batch)
    return (torch.cat(attr), torch.cat(states_x), dgl.batch(nets)), torch.cat(states_y)
