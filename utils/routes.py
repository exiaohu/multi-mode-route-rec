import datetime
import json
import pickle
import random
from collections import namedtuple, defaultdict
from heapq import heappush, heappop
from itertools import count, chain, product, combinations
from typing import List

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from shapely import wkt, ops
from shapely.geometry import shape, Point, LineString

from utils.geotool import realdis

RoutingResult = namedtuple(
    'RoutingResult', [
        'info',  # 成功状态说明
        'origin',  # 起点坐标
        'destination',  # 终点坐标
        'timestamp',  # 出发时间
        'plans'  # 换乘方案列表
    ]
)

RoutingPlan = namedtuple(
    'RoutingPlan', [
        'cost',  # 此换乘方案价格，单位：元
        'time',  # 预期时间，单位：秒
        'distance',  # 此换乘方案全程距离，单位：米
        'walking_distance',  # 此方案总步行距离，单位：米
        'transit_distance',  # 此方案公交行驶距离，单位：米
        'taxi_distance',  # 此方案出租车行驶距离，单位：米
        'path',  # 此换乘方案的路径坐标列表
        'segments'  # 换乘路段列表，以每次换乘动结束作为分段点，将整个换乘方案分隔成若干 Segment（换乘路段）
    ]
)

TrafficNode = namedtuple('TrafficNode', ['xy', 'route'])


def taxi_price(d):
    start_price = 11
    acc_price = max(d - 2000, 0.) / 1000 * 2.4
    return start_price + acc_price


def weight_function(graph, weight):
    if callable(weight):
        return weight
    # If the weight keyword argument is not callable, we assume it is a
    # string representing the edge attribute containing the weight of
    # the edge.
    if graph.is_multigraph():
        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
    return lambda u, v, data: data.get(weight, 1)


def time_dependent_astar_path(graph, source: TrafficNode, target: TrafficNode,
                              heuristic=None, weight="weight", stime=0.0) -> List[TrafficNode]:
    """Returns a list of nodes in a shortest path between source and target
    using the time dependent A* ("A-star") algorithm.

    There may be more than one shortest path.  This returns only one.

    Parameters
    ----------
    graph : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    heuristic : a function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.

    weight : a function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.
       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly four
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge, and a cost from
       the source node to the start point of the edge. The function
       must return a number.

    stime : a float
        If it is provided, that is the pre-accumulated cost of the route.
    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.
    """
    if source not in graph or target not in graph:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(_, __):
            return 0

    push = heappush
    pop = heappop
    weight = weight_function(graph, weight)

    # The queue stores priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guaranteed unique for all nodes in the graph.
    c = count()
    queue = [(0, next(c), source, stime, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}

    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path

        if curnode in explored:
            # Do not override the parent of starting node
            if explored[curnode] is None:
                continue

            # Skip bad paths that were enqueued before finding a better one
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        for neighbor, w in graph[curnode].items():
            ncost = dist + weight(curnode, neighbor, w, dist)
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                # if qcost <= ncost, a less costly path from the
                # neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                # to the queue
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target)
            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")


class TrafficNetwork:
    def __init__(self, graph: nx.Graph, weight_func=None, heuristic_func=None):
        def dist(a, b, _=None, __=None):
            (x1, y1), (x2, y2) = a.xy, b.xy

            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        def heuristic(a, b):
            (x1, y1), (x2, y2) = a.xy, b.xy

            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        self.weight_func = weight_func or dist
        self.heuristic_func = heuristic_func or heuristic
        self.graph = graph
        self.nodes = sorted(graph.nodes)
        self.search_nodes = KDTree([n.xy for n in self.nodes])

    def nearest_node(self, p, k=1):
        _, i = self.search_nodes.query(p, k=k)
        return self.nodes[i] if isinstance(i, int) else [self.nodes[ii] for ii in i]

    def shortest_path(self, src, dst, stime=0.0, **kwargs):
        assert src in self.nodes and dst in self.nodes

        return time_dependent_astar_path(
            self.graph, src, dst,
            kwargs.get('heuristic_func', None) or self.heuristic_func,
            kwargs.get('weight_func', None) or self.weight_func,
            stime=stime
        )


def subway(shentie):
    shentie = json.load(open(shentie))

    station2lines, station2name, links = defaultdict(set), dict(), list()
    for L in shentie['l']:
        for st in L['st']:
            lng, lat = map(float, st['sl'].split(','))
            station2lines[(lng, lat)].add(L['kn'])
            station2name[(lng, lat)] = st['n']

        for st1, st2 in zip(L['st'][:-1], L['st'][1:]):
            lng1, lat1 = map(float, st1['sl'].split(','))
            lng2, lat2 = map(float, st2['sl'].split(','))
            links.append((
                TrafficNode(xy=(lng1, lat1), route=L['kn']),
                TrafficNode(xy=(lng2, lat2), route=L['kn']), {
                    'from': st1['n'],
                    'to': st2['n'],
                    'length': realdis(lng1, lat1, lng2, lat2),
                    'route': L['kn']
                })
            )

    for station, lines in station2lines.items():
        for line1, line2 in combinations(lines, 2):
            links.append((
                TrafficNode(xy=station, route=line1),
                TrafficNode(xy=station, route=line2), {
                    'from': station2name[station],
                    'to': station2name[station],
                    'length': 1000,
                    'route': 'transit'
                })
            )

    return TrafficNetwork(nx.Graph(links), heuristic_func=lambda u, v: 0)


def bus(route_path, stop_path, route_stop_path):
    routes = pd.read_csv(route_path, usecols=['route_id', 'route_name', 'basic_price', 'line_name']).dropna()
    routes = routes[routes.line_name.str.find('地铁') < 0]
    stops = pd.read_csv(stop_path, usecols=['stop_id', 'stop_name', 'wkt'])
    route_stop = pd.read_csv(route_stop_path, usecols=['route_id', 'stop_id', 'stop_index'])

    data = stops.merge(route_stop, left_on='stop_id', right_on='stop_id') \
        .merge(routes, left_on='route_id', right_on='route_id')
    data['wkt'] = data.wkt.transform(wkt.loads)
    data['stop'] = data.apply(lambda _i: TrafficNode(xy=(_i.wkt.x, _i.wkt.y), route=_i.route_name), axis=1)
    stop2name = data[['stop', 'stop_name']].groupby('stop').first().stop_name.to_dict()
    route2price = {name: price for name, price in zip(routes.route_name, routes.basic_price)}

    def parse_edges(s):
        _s = [_i[0] for _i in sorted(zip(s.stop, s.stop_index), key=lambda _i: _i[-1])]
        return {(fro, to): {
            'from': stop2name[fro], 'to': stop2name[to],
            'length': realdis(*fro.xy, *to.xy),
            'route': s.route_name.iloc[0]
        } for fro, to in zip(_s[:-1], _s[1:])}

    edges = {k: v for d in data.groupby('route_id').apply(parse_edges) for k, v in d.items()}

    edgesa, stops = np.asarray([(f.xy, t.xy) for f, t in edges]), sorted(set(chain(*edges)))
    median = np.median(((edgesa[:, 0] - edgesa[:, 1]) ** 2).sum(-1) ** 0.5)
    for i, j in KDTree([s.xy for s in stops]).query_pairs(median * 2):
        for _fro, _to in [(stops[i], stops[j]), (stops[j], stops[i])]:
            if (_fro, _to) not in edges.keys():
                edges[(_fro, _to)] = {
                    'from': stop2name[_fro],
                    'to': stop2name[_to],
                    'length': realdis(*_fro.xy, *_to.xy),
                    'route': 'transit'
                }

    return TrafficNetwork(
        nx.DiGraph((u, v, a) for (u, v), a in edges.items()),
        lambda _, __, a, ___: max(a['length'] * 2, 1000) if a['route'] == 'transit' else a['length'],  # 换乘惩罚
        lambda u, v: 0. if u.route == v.route != 'transit' else 1.
    ), route2price


def road_net(path, predict_speed=None):
    def ha():
        """
        :return: a function takes exactly two positional arguments:
            a number indicating the POSIX time and the road_id, and
            return a number representing the predicted speed.
        """
        lsha_path = 'data/link_speed_ha.pickle'
        try:
            lsha = pickle.load(open(lsha_path, 'rb'))
        except OSError:
            print(f'WARNING: the link speed historical average values not exist at [{lsha_path}].')
            return lambda _, __: 1.

        default = sum(lsha.values()) / len(lsha)

        def _predict_speed(timestamp, road_id):
            timestamp = (round(timestamp) % (60 * 60 * 24)) // (5 * 60) + 1
            return lsha.get((timestamp, int(road_id)), default) or default

        return _predict_speed

    predict_speed = predict_speed or ha()

    roads = gpd.read_file(path)[['link_id', 'length', 'road_name', 'dir', 'geometry']]
    roads['fro'] = roads.apply(lambda _r: TrafficNode(xy=_r.geometry.coords[0], route=''), axis=1)
    roads['to'] = roads.apply(lambda _r: TrafficNode(xy=_r.geometry.coords[-1], route=''), axis=1)

    edges = list()
    for r in roads.itertuples(False):
        edges.append((r.fro, r.to, {
            'link_id': r.link_id,
            'from': r.fro.route,
            'to': r.to.route,
            'length': r.length,
            'route': r.road_name,
            'wkt': r.geometry
        }))
        if r.dir == 0:
            edges.append((r.to, r.fro, {
                'link_id': r.link_id,
                'from': r.to.route,
                'to': r.fro.route,
                'length': r.length,
                'route': r.road_name,
                'wkt': LineString(r.geometry.coords[::-1])
            }))

    return TrafficNetwork(nx.DiGraph(edges), lambda u, v, d, c: d['length'] / predict_speed(c, int(d['link_id'])) * 3.6)


class RoutePlanner:
    SUBWAY_SPEED = 50 * 1000 / 3600  # 假定地铁速度50公里每小时
    BUS_SPEED = 30 * 1000 / 3600  # 假定公交速度30公里每小时
    WALK_SPEED = 10 * 1000 / 3600  # 步行速度10公里每小时

    INFO_SUCCESS = 'success'
    INFO_UNREACH = 'error: not reachable.'
    INFO_TOO_CLOSE = 'error: origin and destination are too close.'

    def __init__(
            self,
            subway_path='data/shentie.json',
            bus_route_path='/home/buaa/data/base_info/route.csv',
            bus_stop_path='/home/buaa/data/base_info/stop.csv',
            bus_route_stop_path='/home/buaa/data/base_info/route_stop.csv',
            road_path='/home/buaa/data/t_common_base_link_2020M4_sz_#98350/t_common_base_link_98350.shp',
            sz_path='data/shenzhen.geojson'
    ):
        self.subway = subway(subway_path)
        self.bus, self.bus_price = bus(bus_route_path, bus_stop_path, bus_route_stop_path)
        self.road_net = road_net(road_path)

        self.shenzhen = max(shape(json.load(open(sz_path)).get('features')[0].get('geometry')), key=lambda a: a.area)

    def check_params(self, org, dst, timestamp, total):
        assert len(org) == 2 and all(map(lambda x: isinstance(x, float), org)) and self.shenzhen.contains(Point(org))
        assert len(dst) == 2 and all(map(lambda x: isinstance(x, float), dst)) and self.shenzhen.contains(Point(dst))
        assert isinstance(timestamp, float) or isinstance(timestamp, datetime.datetime)
        assert isinstance(total, int) and total > 0

    def find_car_plans(self, org, dst, timestamp, total=3):
        self.check_params(org, dst, timestamp, total)
        net = rp.road_net

        org_nnr = self.road_net.nearest_node(org, 3)
        dst_nnr = self.road_net.nearest_node(dst, 3)
        if len(set(org_nnr).union(dst_nnr)) < len(org_nnr) + len(dst_nnr):
            return RoutingResult(self.INFO_TOO_CLOSE, org, dst, timestamp, [])

        car_plans = list()
        for fro, to in product(org_nnr, dst_nnr):
            try:
                pth = self.road_net.shortest_path(fro, to, timestamp)
                ctime = timestamp
                for u, v in zip(pth[:-1], pth[1:]):
                    ctime += self.road_net.weight_func(u, v, self.road_net.graph[u][v], ctime)
                dist = sum(self.road_net.graph[u][v]['length'] for u, v in zip(pth[:-1], pth[1:]))
                wdis = realdis(*fro.xy, *org) + realdis(*to.xy, *dst)

                plan = RoutingPlan(
                    cost=taxi_price(dist),
                    time=ctime - timestamp,
                    distance=dist + wdis,
                    walking_distance=wdis,
                    transit_distance=0,
                    taxi_distance=dist,
                    path=list(
                        ops.linemerge([net.graph[u][v]['wkt'] for u, v in zip(pth[:-1], pth[1:])]).coords),
                    segments=None
                )
                car_plans.append(plan)
            except nx.NetworkXNoPath:
                pass
        if len(car_plans) > 0:
            car_plans = sorted(car_plans, key=lambda item: item.time)[:total]
            res = RoutingResult(self.INFO_SUCCESS, org, dst, timestamp, car_plans)
        else:
            return RoutingResult(self.INFO_UNREACH, org, dst, timestamp, [])

        return res

    @staticmethod
    def __get_plan(net, org, fro, to, dst, timestamp, get_price, link_speed, walk_speed):
        def clean_path(p):
            cur_route, ps = 'transit', list()
            for u, v in zip(p[:-1], p[1:]):
                r = net.graph[u][v]['route']
                if r != 'transit' and cur_route == r:
                    ps[-1].append(v)
                elif r != 'transit' and cur_route != r:
                    ps.append([u, v])
                else:
                    pass
                cur_route = r

            return ps

        paths = clean_path(net.shortest_path(fro, to, timestamp))
        tdis = sum(net.graph[u][v]['length'] for path in paths for u, v in zip(path[:-1], path[1:]))
        wdis = sum(realdis(*up[-1].xy, *vp[0].xy) for up, vp in zip(paths[:-1], paths[1:]))
        wdis += realdis(*org, *paths[0][0].xy) + realdis(*dst, *paths[-1][-1].xy)
        price = sum(get_price(net.graph[path[0]][path[1]]['route']) for path in paths)
        segments = list()
        for path in paths:
            segment = [net.graph[path[0]][path[1]]['from']]
            for u, v in zip(path[:-1], path[1:]):
                segment.append(net.graph[u][v]['to'])
            segments.append((net.graph[path[0]][path[1]]['route'], segment))

        return RoutingPlan(
            cost=price,
            time=tdis / link_speed + wdis / walk_speed,
            distance=wdis + tdis,
            walking_distance=wdis,
            transit_distance=tdis,
            taxi_distance=0,
            path=[[n.xy for n in p] for p in paths],
            segments=segments
        )

    @classmethod
    def __find_plans(cls, net, get_price, link_speed, walk_speed, org, dst, timestamp, total=3):

        org_nnr = net.nearest_node(org, 3)
        dst_nnr = net.nearest_node(dst, 3)
        if len(set(org_nnr).union(dst_nnr)) < len(org_nnr) + len(dst_nnr):
            return RoutingResult(cls.INFO_TOO_CLOSE, org, dst, timestamp, [])

        plans = list()
        for fro, to in product(org_nnr, dst_nnr):
            try:
                plans.append(cls.__get_plan(net, org, fro, to, dst, timestamp, get_price, link_speed, walk_speed))
            except nx.NetworkXNoPath:
                pass
        if len(plans) > 0:
            plans = sorted(plans, key=lambda item: item.time)[:total]
            res = RoutingResult(cls.INFO_SUCCESS, org, dst, timestamp, plans)
        else:
            return RoutingResult(cls.INFO_UNREACH, org, dst, timestamp, [])

        return res

    def find_bus_plans(self, org, dst, timestamp, total=3):
        self.check_params(org, dst, timestamp, total)
        net, get_price = self.bus, lambda ln: self.bus_price[ln]

        return self.__find_plans(net, get_price, self.BUS_SPEED, self.WALK_SPEED, org, dst, timestamp, total)

    def find_subway_plans(self, org, dst, timestamp, total=3):
        self.check_params(org, dst, timestamp, total)
        net, get_price = self.subway, lambda ln: 6

        return self.__find_plans(net, get_price, self.SUBWAY_SPEED, self.WALK_SPEED, org, dst, timestamp, total)

    def __call__(self, org, dst, timestamp, total=3):
        # 考虑换乘模式：
        #   单模式：公交、地铁、出租
        #   双模式：公交+地铁、地铁+出租
        #   三模式：公交+地铁+出租
        res = self.find_car_plans(org, dst, timestamp, total)
        if res.info != self.INFO_SUCCESS:
            return res

        res.plans.extend(self.find_bus_plans(org, dst, timestamp, total).plans)
        res.plans.extend(self.find_subway_plans(org, dst, timestamp, total).plans)

        subway_org, subway_dst = self.subway.nearest_node(org, 3), self.subway.nearest_node(dst, 3)
        if len(set(subway_org).union(subway_dst)) < len(subway_org) + len(subway_dst):
            return res

        for so, sd in product(subway_org, subway_dst):
            step0 = self.find_car_plans(org, so.xy, timestamp).plans + self.find_bus_plans(org, so.xy, timestamp).plans
            time0 = sum(p.time for p in step0) / len(step0)

            price_func, ss, ws = lambda ln: 6, self.SUBWAY_SPEED, self.WALK_SPEED
            step1 = [self.__get_plan(self.subway, so.xy, so, sd, sd.xy, timestamp, price_func, ss, ws)]
            time1 = step1[0].time

            since = timestamp + time0 + time1
            step2 = self.find_car_plans(sd.xy, dst, since).plans + self.find_bus_plans(sd.xy, dst, since).plans

            res.plans.extend(product(step0, step1, step2))

        _key = lambda pl: pl.time if isinstance(pl, RoutingPlan) else sum(p.time for p in pl)
        res.plans = sorted(res.plans, key=_key)[:total]

        return res


def test(region, method):
    def random_point_within(poly):
        min_x, min_y, max_x, max_y = poly.bounds
        while True:
            x, y = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
            if poly.contains(Point([x, y])):
                return x, y

    src, dst = random_point_within(region), random_point_within(region)
    t = datetime.datetime.utcnow().timestamp() + random.randint(-24 * 60 * 60, +24 * 60 * 60)

    return method(src, dst, t)


if __name__ == '__main__':
    import time

    rp = RoutePlanner()
    # plans = rp((114.23126585001434, 22.653067607112828), (114.1725139159855, 22.596286689331684), .0)

    since = time.perf_counter()
    [test(rp.shenzhen, rp) for _ in range(10000)]
    print((time.perf_counter() - since) / 10000)
