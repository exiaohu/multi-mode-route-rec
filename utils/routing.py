import random
import time
from collections import defaultdict
from heapq import heappop, heappush
from typing import List, Tuple, Any

from scipy.spatial import KDTree

from .gen_nets import GeneralNode

__all__ = ['EdgesLookup']


class EdgesLookup:
    def __init__(self, edges: List[Tuple[GeneralNode, GeneralNode, Any]], weight=None):
        """
        initialize edges' attributes
        :param edges: iterable of tuple as (`source node`, `target node`, `attribute(s)`)
        :param weight: function, given attribute(s) and a timestamp, it calculates a time-dependent cost.
        """
        edge_dict = {(fro, to): attr for fro, to, attr in edges}

        lookup, nodes = defaultdict(list), set()
        for s, t, a in edges:
            lookup[s].append((a, t))
            nodes.add(s)
            nodes.add(t)

        nodes = list(nodes)
        kdtree = KDTree(list(map(lambda it: it.point, nodes)))

        self.weight = weight
        self.lookup = lookup
        self.masked_edges = []

        self.nodes = nodes
        self.nodes_kdtree = kdtree

        self.edge_dict = edge_dict

    def nearest_node(self, p):
        d, i = self.nodes_kdtree.query(p)
        return self.nodes[i]

    def get(self, n_ts, default):
        """
        get neighbors and dynamic costs to them.
        :param n_ts: tuple as (node key, timestamp)
        :param default:
        :return: iterable of tuple as (cost, neighbor key)
        """
        node, timestamp = n_ts
        try:
            neighbors = []
            for attr, neighbor in self.lookup[node]:
                if (node, neighbor) not in self.masked_edges:
                    neighbors.append((self.weight(attr, timestamp), neighbor))

            return neighbors
        except KeyError:
            return default

    def shortest_path(self, f, t, ts, weight=None):
        tmp_weight = self.weight
        if weight is not None:
            self.weight = weight

        assert self.weight is not None, 'Must specify a weight matrix.'

        res = self._time_dependent_dijkstra(f, t, ts)

        self.weight = tmp_weight

        return res

    def get_attr(self, fro, to, default):
        return self.edge_dict.get((fro, to), default)

    def _time_dependent_dijkstra(self, f, t, ts):
        # dist records the min value of each node in heap.
        q, seen, dist = [(0, f, (), ts)], set(), {f: 0}
        while q:
            (cost, v1, path, ts) = heappop(q)
            if v1 in seen:
                continue

            seen.add(v1)
            path += ((v1, ts),)
            if v1 == t:
                return cost, path

            for c, v2 in self.get((v1, ts), ()):
                if v2 in seen:
                    continue
                # Not every edge will be calculated. Edges which can improve the value of node in heap will be useful.
                if v2 not in dist or cost + c < dist[v2]:
                    dist[v2] = cost + c
                    heappush(q, (cost + c, v2, path, ts + c))
        return float("inf"), ()

    def k_shortest_path(self, f, t, ts, k=3, weight=None):
        cost, path = self.shortest_path(f, t, ts, weight)

        paths, potential_paths = [(cost, path)], list()

        if not path:
            return paths

        for k in range(1, k):
            prev_pth = paths[-1][1]
            for i in range(0, len(prev_pth) - 1):
                node_spur, cost_spur = prev_pth[i]
                path_root, path_cost = zip(*prev_pth[:i + 1])
                self.masked_edges = set()
                for _, curr_path in paths:
                    curr_path, _ = zip(*curr_path)
                    if len(curr_path) > i and path_root == curr_path[:i + 1]:
                        self.masked_edges.add((curr_path[i], curr_path[i + 1]))

                cost_remain, path_spur = self.shortest_path(node_spur, t, cost_spur, weight)
                if cost_remain < float('inf'):
                    path_spur, cost_end = zip(*path_spur)
                    path_total = path_root[:-1] + path_spur
                    cost_iter = path_cost[:-1] + cost_end
                    potential_k = (cost_remain + cost_spur - ts, tuple(zip(path_total, cost_iter)))

                    if not (potential_k in potential_paths):
                        potential_paths.append(potential_k)

                self.masked_edges = set()

            if len(potential_paths):
                potential_paths = sorted(potential_paths, key=lambda _i: _i[0])
                paths.append(potential_paths[0])
                potential_paths.pop(0)
            else:
                break

        return paths


def gen_dynamic_edges(road_net, speeds):
    speeds = speeds.speed.unstack(level=-1)
    fill_na_with = speeds.mean().mean()  # fill NaN with mean speed
    speeds = speeds.fillna(fill_na_with).to_dict('list')

    return EdgesLookup([(GeneralNode(None, o, None, None), GeneralNode(None, d, None, None), {
        'length': float(road_net.edges[(o, d)]['length']),
        'speed': speeds.get(int(road_net.edges[(o, d)]['id']), [fill_na_with] * 96)
    }) for o, d in road_net.edges], lambda attr, ts: attr['length'] / attr['speed'][int(ts) // 900] * 3600)


def test():
    import networkx as nx
    import pandas as pd

    bj_roads = r'data/test/bj_small_roads_sample/bj_small_roads_sample.shp'
    bj_speeds = r'data/roads-20180801-20180831.parquet'

    road_net = nx.read_shp(bj_roads)
    edges = EdgesLookup([(
        GeneralNode(None, o, None, None),
        GeneralNode(None, d, None, None),
        float(road_net.edges[(o, d)]['length'])) for o, d in road_net.edges
    ], lambda a, _: a)
    since = time.perf_counter()
    res = edges.k_shortest_path(
        GeneralNode(None, (116.331194, 39.957388), None, None),
        GeneralNode(None, (116.417038, 40.002861), None, None), 0
    )
    print('since', time.perf_counter() - since)
    print(*map(lambda i: (i[0], i[1][:5]), res), sep='\n')

    edges = gen_dynamic_edges(road_net, pd.read_parquet(bj_speeds))
    for _ in range(3):
        ts = random.randint(0, 24 * 60 * 60)
        since = time.perf_counter()
        res = edges.k_shortest_path(
            GeneralNode(None, (116.331194, 39.957388), None, None),
            GeneralNode(None, (116.417038, 40.002861), None, None), ts
        )
        print('since', time.perf_counter() - since)
        print(*map(lambda i: (i[0], i[1][:5]), res), sep='\n')


if __name__ == '__main__':
    test()
