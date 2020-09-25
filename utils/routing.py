import random
import time
from collections import defaultdict
from heapq import heappop, heappush


class EdgesLookup:
    def __init__(self, edges, weight):
        """
        initialize edges' attributes
        :param edges: iterable of tuple as (`source node key`, `target node key`, `attribute(s)`)
        :param weight: function, given attribute(s) and a timestamp, it calculates a time-dependent cost.
        """
        lookup = defaultdict(list)
        for s, t, a in edges:
            lookup[s].append((a, t))

        self.weight = weight
        self.lookup = lookup

    def get(self, n_ts, default):
        """
        get neighbors and dynamic costs to them.
        :param n_ts: tuple as (node key, timestamp)
        :param default:
        :return: iterable of tuple as (cost, neighbor key)
        """
        node, timestamp = n_ts
        try:
            return map(lambda args: (self.weight(args[0], timestamp), args[1]), self.lookup[node])
        except KeyError:
            return default

    def time_dependent_dijkstra(self, f, t, ts):
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
        return float("inf")


def gen_dynamic_edges(road_net, speeds):
    speeds = speeds.speeds.unstack(level=-1)
    fill_na_with = speeds.mean().mean()  # fill NaN with mean speed
    speeds = speeds.fillna(fill_na_with).to_dict('list')

    return EdgesLookup(((*e, {
        'length': float(road_net.edges[e]['length']),
        'speed': speeds.get(int(road_net.edges[e]['id']), [fill_na_with] * 96)
    }) for e in road_net.edges), lambda attr, ts: attr['length'] / attr['speed'][int(ts) // 900] * 3600)


def test():
    import networkx as nx
    import pandas as pd

    bj_roads = r'/home/huxiao/data/bj_data/bj_small_roads_sample/bj_small_roads_sample.shp'
    bj_speeds = r'/home/huxiao/data/bj_data/taxi_road_speeds-20180801.parquet'

    road_net = nx.read_shp(bj_roads)
    edges = EdgesLookup(((*e, float(road_net.edges[e]['length'])) for e in road_net.edges), lambda a, _: a)
    since = time.perf_counter()
    res = edges.time_dependent_dijkstra((116.331194, 39.957388), (116.417038, 40.002861), 0)
    print('since', time.perf_counter() - since)
    print(res)

    for _ in range(3):
        ts = random.randint(0, 24 * 60 * 60)
        speeds = pd.read_parquet(bj_speeds)
        edges = gen_dynamic_edges(road_net, speeds)
        since = time.perf_counter()
        res = edges.time_dependent_dijkstra((116.331194, 39.957388), (116.417038, 40.002861), ts)
        print('since', time.perf_counter() - since)
        print(res)


if __name__ == '__main__':
    test()
