import math
import os
import pickle
from collections import namedtuple
from itertools import combinations

import geopandas as gpd
import networkx as nx
import pandas as pd
from scipy.spatial import KDTree
from shapely.geometry import MultiLineString, Point
from tqdm import tqdm

from .geometry_util import distance_btw_two_points_on_a_line, transform_coordinate_system as transform
from .helper import timing

GeneralNode = namedtuple('GeneralNode', ['modal', 'point', 'name', 'line'])

__all__ = ['GeneralNode', 'get_bus', 'get_subway', 'get_road', 'get_multi_modal']


def get_real_distance(pt1: Point, pt2: Point, line: MultiLineString, crs) -> float:
    assert line.distance(pt1) < 1e-8 and line.distance(pt2) < 1e-8
    line, pt1, pt2 = transform([line, pt1, pt2], source_cs=crs)
    return distance_btw_two_points_on_a_line(line, pt1, pt2)


@timing
def _subway(
        stops='data/beijing_geo/wgs_subway_stops.geojson',
        lines='data/beijing_geo/wgs_subway_lines.geojson',
        plans='data/beijing_subway/subway_plans.pickle',
):
    stops, lines = gpd.read_file(stops), gpd.read_file(lines)
    assert stops.crs == lines.crs

    plans = pickle.load(open(plans, 'rb'))

    crs = stops.crs

    data = pd.merge(stops, lines, left_on='line_name', right_on='name', suffixes=('_stop', '_line'))
    del stops, lines

    data = data.drop(columns='line_name')
    data = data.rename(columns={
        'name_stop': 'stop_name',
        'name_line': 'line_name',
        'sequence': 'stop_seq',
        'sp': 'stop_spell',
        'geometry_stop': 'stop_geom',
        'geometry_line': 'line_geom'
    })

    nodes = set(data.apply(
        lambda i: GeneralNode('subway', (i.stop_geom.x, i.stop_geom.y), i.stop_name, i.line_name), axis=1))
    stop_geom_lookup = {node.name: node for node in nodes}
    line_geom_lookup = data.groupby('line_name').agg({'stop_name': list, 'stop_seq': list, 'line_geom': 'first'})

    net = nx.DiGraph()
    # 增加站点，作为图的结点
    net.add_nodes_from(nodes)

    # 增加边
    for si, sj in tqdm(combinations(nodes, 2)):
        if si.name != sj.name and (si.name, sj.name) not in [('T2航站楼', 'T3航站楼'), ('T3航站楼', 'T2航站楼')]:
            plan = plans[(si.name, sj.name)]
            try:
                dist = sum(get_real_distance(
                    Point(stop_geom_lookup[line[0][1]].point),
                    Point(stop_geom_lookup[line[-1][1]].point),
                    line_geom_lookup.loc[line[0][0]].line_geom,
                    crs
                ) for line in plan['plan'])

                net.add_edge(si, sj, distance=dist, **plan['cost'])
            except AttributeError:
                print('', si, sj, '', sep='\n')

    return net


def get_subway(saved='data/multi-net/subway.gpickle'):
    if os.path.exists(saved) and os.path.isfile(saved):
        return nx.read_gpickle(saved)
    else:
        net = _subway()
        nx.write_gpickle(net, saved)
        return net


@timing
def _bus(stops='data/beijing_geo/wgs_bus_stops.geojson', lines='data/beijing_geo/wgs_bus_lines.geojson'):
    bus_speed = 18 * 1000 / 60  # 18公里每小时

    stops, lines = gpd.read_file(stops), gpd.read_file(lines)
    assert stops.crs == lines.crs
    crs = stops.crs

    data = pd.merge(stops, lines, left_on='line_id', right_on='id', suffixes=('_stop', '_line'))
    del stops, lines

    data = data.drop(columns=['id_stop', 'line_id', 'id_line', 'type', 'bounds'])
    data = data.rename(columns={
        'name_stop': 'stop_name',
        'sequence': 'stop_seq',
        'geometry_stop': 'stop_geom',
        'name_line': 'line_name',
        'start_stop': 'start_stop',
        'end_stop': 'end_stop',
        'distance': 'line_dist',
        'geometry_line': 'line_geom'
    })

    nodes = list(set(data.apply(
        lambda it: GeneralNode('bus', (it.stop_geom.x, it.stop_geom.y), it.stop_name, it.line_name), axis=1
    )))
    lines = data.groupby('line_name').agg({
        'stop_name': list,
        'stop_seq': list,
        'stop_geom': list,
        'start_stop': 'first',
        'end_stop': 'first',
        'line_dist': 'first',
        'line_geom': 'first'
    })
    lines['stops'] = lines.apply(
        lambda line: [GeneralNode('bus', (_geom.x, _geom.y), _name, line.name) for _name, _, _geom in
                      sorted(zip(line.stop_name, line.stop_seq, line.stop_geom), key=lambda it: int(it[1]))], axis=1
    )
    lines = lines.drop(columns=['stop_name', 'stop_seq', 'stop_geom'])

    net = nx.DiGraph()
    # 增加站点，作为图的结点
    net.add_nodes_from(nodes)

    # 增加非换乘边
    for name, info in tqdm(lines.iterrows()):
        # 10公里（含）内2元。
        # 10公里以上部分，每增加1元可乘坐5公里。
        stops, geom = info.stops, info.line_geom
        for w in range(1, len(stops)):
            for i in range(len(stops) - w):
                j = i + w
                srt, end = stops[i], stops[j]
                if w == 1:
                    dist = get_real_distance(Point(srt.point), Point(end.point), geom, crs=crs)
                else:
                    itr = stops[i + 1]
                    dist = net.get_edge_data(srt, itr).get('distance') + net.get_edge_data(itr, end).get('distance')
                net.add_edge(srt, end, **{
                    'distance': dist,
                    'time': dist / bus_speed,
                    'price': math.ceil((dist - 10) / 5) + 2,
                    'transfer_time': 0,
                    'n_stations': w,
                    'line': name,
                    'plan': stops[i: j + 1]
                })
    # 增加公交换乘边
    coords = list(map(lambda it: (it.x, it.y), map(lambda it: transform(Point(it.point), source_cs=crs), nodes)))
    kdtree = KDTree(coords)
    for i, j in tqdm(kdtree.query_pairs(500)):  # any pair of nodes within 500 meters
        ix, iy = coords[i]
        jx, jy = coords[j]
        dist = ((ix - jx) ** 2 + (iy - jy) ** 2) ** .5
        net.add_edge(nodes[i], nodes[j], **{
            'distance': dist,
            'time': 5,
            'price': 0,
            'transfer_time': 1,
            'plan': [nodes[i].line, nodes[j].line]
        })

    return net


def get_bus(saved='data/multi-net/bus.gpickle'):
    if os.path.exists(saved) and os.path.isfile(saved):
        return nx.read_gpickle(saved)
    else:
        net = _bus()
        nx.write_gpickle(net, saved)
        return net


@timing
def _road(road='data/beijing_geo/wgs_roadnet.geojson'):
    walk_speed = 5 * 1000 / 60  # 人类步行速度约为 5 公里每小时

    # walk, drive or taxi
    road = gpd.read_file(road)
    road['id'] = pd.to_numeric(road.id)
    road['width'] = pd.to_numeric(road.width)
    road['direction'] = pd.to_numeric(road.direction)
    road['toll'] = pd.to_numeric(road.toll)
    road['snodeid'] = pd.to_numeric(road.snodeid)
    road['enodeid'] = pd.to_numeric(road.enodeid)
    road['length'] = pd.to_numeric(road['length'])
    road['l'] = road.to_crs(epsg=3857).length
    road['speedclass'] = pd.to_numeric(road.speedclass)
    road['lanenum'] = pd.to_numeric(road.lanenum)

    road = road.set_index('id')

    net = nx.DiGraph()
    for i, r in tqdm(road.iterrows()):
        is_motor = any(map(lambda k: 1 <= int(k[:2], base=16) <= 6, r.kind.split('|')))
        geoms = list(r.geometry.geoms)

        assert len(geoms) == 1

        walk_start = GeneralNode('walk', geoms[0].coords[0], r.snodeid, None)
        walk_end = GeneralNode('walk', geoms[0].coords[-1], r.enodeid, None)

        walk_edge = {
            'distance': r.l,
            'time': r.l / walk_speed,
            'price': 0,
            'transfer_time': 0,
            'id': i,
            'geometry': r.geometry
        }

        net.add_edge(walk_start, walk_end, **walk_edge)
        net.add_edge(walk_end, walk_start, **walk_edge)

        if is_motor:
            motor_start = GeneralNode('motor', geoms[0].coords[0], r.snodeid, None)
            motor_end = GeneralNode('motor', geoms[0].coords[-1], r.enodeid, None)
            motor_edge = {
                'distance': r.l,
                'time': 'dynamic',
                'price': 0,
                'transfer_time': 0,
                'id': i,
                'geometry': r.geometry
            }
            net.add_edge(motor_start, motor_end, **motor_edge)
            net.add_edge(motor_end, motor_start, **motor_edge)

            switch_edge = {
                'distance': 0,
                'time': 5,
                'price': 0,
                'transfer_time': 1,
            }

            net.add_edge(motor_start, walk_start, **switch_edge)
            net.add_edge(walk_start, motor_start, **switch_edge)
            net.add_edge(motor_end, walk_end, **switch_edge)
            net.add_edge(walk_end, motor_end, **switch_edge)

    return net


def get_road(saved='data/multi-net/road.gpickle'):
    if os.path.exists(saved) and os.path.isfile(saved):
        return nx.read_gpickle(saved)
    else:
        net = _road()
        nx.write_gpickle(net, saved)
        return net


@timing
def _combine_nets(*nets: nx.DiGraph):
    net = nx.compose_all(nets)

    ns = [list(this.nodes) for this in nets]
    cs = [list(map(lambda it: (it.x, it.y), map(lambda it: transform(Point(it.point)), n))) for n in ns]

    for (t, tn, tc), (o, on, oc) in tqdm(combinations(zip(nets, ns, cs), 2)):
        bkdtree, skdtree = KDTree(tc), KDTree(oc)
        for i, js in tqdm(enumerate(bkdtree.query_ball_tree(skdtree, 500))):  # any pair of nodes within 500 meters
            for j in js:
                if tn[i].modal != on[j].modal:
                    ix, iy = tc[i]
                    jx, jy = oc[j]
                    dist = ((ix - jx) ** 2 + (iy - jy) ** 2) ** .5
                    net.add_edge(tn[i], on[j], **{
                        'distance': dist,
                        'time': 5,
                        'price': 0,
                        'transfer_time': 1,
                    })
    return net


def _multi_modal():
    return _combine_nets(get_bus(), get_subway(), get_road())


def get_multi_modal(saved='data/multi-net/multi-modal.gpickle'):
    if os.path.exists(saved) and os.path.isfile(saved):
        return nx.read_gpickle(saved)
    else:
        net = _multi_modal()
        nx.write_gpickle(net, saved)
        return net


if __name__ == '__main__':
    print('generate multi-modal graph')
    saved = 'data/multi-net/multi-modal.gpickle'
    nx.write_gpickle(_multi_modal(), saved)
    print('multi-modal graph saved at', saved)
