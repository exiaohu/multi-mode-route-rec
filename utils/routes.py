import json
import pickle
from heapq import heappush, heappop
from itertools import count, chain

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from shapely import wkt


def weight_function(graph, weight):
    if callable(weight):
        return weight
    # If the weight keyword argument is not callable, we assume it is a
    # string representing the edge attribute containing the weight of
    # the edge.
    if graph.is_multigraph():
        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
    return lambda u, v, data: data.get(weight, 1)


def time_dependent_astar_path(graph, source, target, heuristic=None, weight="weight", stime=0.0):
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
            (x1, y1), (x2, y2) = a, b

            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        def heuristic(a, b):
            (x1, y1), (x2, y2) = a, b

            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        self.weight_func = weight_func or dist
        self.heuristic_func = heuristic_func or heuristic
        self.graph = graph
        self.nodes = sorted(graph.nodes)
        self.search_nodes = KDTree(self.nodes)

    def nearest_node(self, p, k=1):
        _, i = self.search_nodes.query(p, k=k)
        return self.nodes[i] if isinstance(i, int) else [self.nodes[ii] for ii in i]

    def shortest_path(self, src, dst, stime=0.0):
        assert src in self.nodes and dst in self.nodes

        return time_dependent_astar_path(self.graph, src, dst, self.heuristic_func, self.weight_func, stime=stime)


def subway(shentie='data/shentie.json'):
    shentie = json.load(open(shentie))

    stations, links = list(), list()
    for L in shentie['l']:
        for st in L['st']:
            lng, lat = map(float, st['sl'].split(','))
            stations.append((lng, lat))

        for st1, st2 in zip(L['st'][:-1], L['st'][1:]):
            lng1, lat1 = map(float, st1['sl'].split(','))
            lng2, lat2 = map(float, st2['sl'].split(','))
            links.append(((lng1, lat1), (lng2, lat2)))

    return TrafficNetwork(nx.Graph(links), heuristic_func=lambda u, v: 0)


def get_predict_speed(method='ha', **kwargs):
    """
    :param method:
    :param kwargs:
    :return: a function takes exactly two positional arguments:
        a number indicating the POSIX time and the road_id, and
        return a number representing the predicted speed.
    """
    if method == 'ha':
        lsha = pickle.load(open(kwargs.get('lsha_path', 'data/link_speed_ha.pickle'), 'rb'))
        default = sum(lsha.values()) / len(lsha)

        def predict_speed(timestamp, road_id):
            timestamp = (round(timestamp) % (60 * 60 * 24)) // (5 * 60) + 1
            return lsha.get((timestamp, int(road_id)), default) or default

        return predict_speed
    else:
        raise ValueError('Method other than')


def bus(
        route_path='/home/buaa/data/base_info/route.csv',
        stop_path='/home/buaa/data/base_info/stop.csv',
        route_stop_path='/home/buaa/data/base_info/route_stop.csv',
):
    routes = pd.read_csv(route_path, index_col='id', usecols=['id', 'route_id', 'route_name'])
    stops = pd.read_csv(stop_path, index_col='id', usecols=['id', 'stop_id', 'stop_name', 'wkt'])
    route_stop = pd.read_csv(route_stop_path, index_col='id', usecols=['id', 'route_id', 'stop_id', 'stop_index'])

    data = stops.merge(route_stop, left_on='stop_id', right_on='stop_id') \
        .merge(routes, left_on='route_id', right_on='route_id')
    data['wkt'] = data.wkt.transform(wkt.loads)
    data['point'] = data.wkt.transform(lambda i: (i.x, i.y))
    stop_to_route = data[['point', 'route_id']].groupby('point').agg(set).route_id.to_dict()

    def parse_edges(stops):
        stops = [i[0] for i in
                 sorted(zip(stops.wkt.transform(lambda p: (p.x, p.y)), stops.stop_index), key=lambda i: i[-1])]
        return set(zip(stops[:-1], stops[1:]))

    edges = set(chain(*data.groupby('route_id').apply(parse_edges)))

    edgesa, stops = np.asarray(list(edges)), sorted(set(chain(*edges)))
    median = np.median(((edgesa[:, 0] - edgesa[:, 1]) ** 2).sum(-1) ** 0.5)
    for i, j in KDTree(stops).query_pairs(median):
        edges.add((stops[i], stops[j]))
        edges.add((stops[j], stops[i]))

    def weight(u, v, _, __):
        (x1, y1), (x2, y2) = u, v
        return (2 - (len(stop_to_route[u].intersection(stop_to_route[v])) > 0)) * (x1 - x2) ** 2 + (y1 - y2) ** 2

    return TrafficNetwork(nx.DiGraph(edges), weight, lambda u, v: 0)


def road_net(
        path='/home/buaa/data/t_common_base_link_2020M4_sz_#98350/t_common_base_link_98350.shp',
        predict_speed=get_predict_speed()
):
    return TrafficNetwork(nx.read_shp(path),
                          lambda u, v, d, c: d['length'] / predict_speed(c, int(d['link_id'])) * 3.6)
