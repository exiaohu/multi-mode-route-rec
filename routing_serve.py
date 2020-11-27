import json
import pickle
from datetime import datetime
from itertools import chain

import numpy as np
from flask import Flask, jsonify, request
from shapely.geometry import Point, Polygon

from utils.multi_modal import get_edges_by_modals, GeneralNode

app = Flask(__name__)

MAX_LNG, MIN_LNG = 116.495, 116.265
MAX_LAT, MIN_LAT = 39.995, 39.820

vld_area = Polygon(((MAX_LNG, MAX_LAT), (MAX_LNG, MIN_LAT), (MIN_LNG, MIN_LAT), (MIN_LNG, MAX_LAT), (MAX_LNG, MAX_LAT)))
vld_modals = ('walking', 'driving', 'taxi', 'public')
vld_pref = ('default', 'distance', 'time', 'price', 'transfer_time')


def get_dynamic_info():
    di = pickle.load(open('data/dynamic/dynamic_road_net.pickle', 'rb'))
    di['timestamps'] = di.timestamps.astype(np.int64)

    mv = di.speed.mean()
    iv = min(a - b for a, b in zip(sorted(set(di.timestamps))[1:], sorted(set(di.timestamps))[:-1]))
    return di.set_index(['part_id', 'timestamps']), mv, iv


dynamic_info, missing_value, interval = get_dynamic_info()
partitions = pickle.load(open('data/partitions.pickle', 'rb'))


def error(msg='Some error happened.'):
    return jsonify({'success': True, 'errorMessage': msg})


def ok(data=None):
    return jsonify({'success': True, 'data': data})


def check_params(args):
    assert 'origin_location' in args.keys() and 'dest_location' in args.keys(), f'No origin or destination specified.'

    origin = Point(json.loads(args['origin_location'])['lng'], json.loads(args['origin_location'])['lat'])
    dest = Point(json.loads(args['dest_location'])['lng'], json.loads(args['dest_location'])['lat'])
    assert vld_area.contains(origin) and vld_area.contains(dest), \
        f'Origin ({origin}) or dest ({dest}) is not inside {vld_area}'

    modals = args.getlist('modals') or vld_modals
    assert len(set(modals).difference(vld_modals)) == 0, f'Modals {modals} is not a subset of {vld_modals}'

    timestamp = datetime.fromtimestamp(int(args['timestamp'])) if 'timestamp' in args.keys() else datetime.now()
    timestamp = timestamp.replace(2018, 8)  # mapping the current time to Aug, 2018

    pref = args.get('preference', 'default')
    assert pref in vld_pref, f'Preference must one of {vld_pref}'

    total = args.get('total', default=3, type=int)

    return origin, dest, get_edges_by_modals(modals), timestamp.timestamp(), pref, total


def gen_plans(cost, _path, edges):
    assert cost < float('inf'), f'无法找到有效路线。'

    summary, path = dict(descriptions='无描述。', costs=dict(distance=0, time=0, price=0, transfer_time=0)), list()
    cur = dict(type=None, path=dict(coordinates=list()))

    prev_node = None
    for n, t in _path:
        if cur['type'] is None:
            cur['type'] = n.modal

        if cur['type'] == n.modal:
            try:
                if prev_node is not None:
                    line = edges.get_attr(prev_node, n, dict())['geometry']
                    cur['path']['coordinates'].extend(
                        list(chain(*([coord for coord in geom.coords] for geom in line.geoms)))[1:])
                else:
                    cur['path']['coordinates'].append(n.point)
            except KeyError:
                cur['path']['coordinates'].append(n.point)
        else:
            path.append(cur)
            cur = dict(type=n.modal, path=dict(coordinates=[n.point]))

        prev_node = n

    def get_attr(n1, n2, ts):
        attr = edges.get_attr(n1, n2, {'distance': 0, 'time': 0, 'price': 0, 'transfer_time': 0})
        if attr['time'] == 'dynamic':
            attr['time'] = edges.weight(attr, ts)
        return float(attr['distance']), float(attr['time']), float(attr['price']), float(attr['transfer_time'])

    dists, times, prices, trans = zip(*(get_attr(n1, n2, ts) for (n1, ts), (n2, _) in zip(_path[:-1], _path[1:])))
    summary['costs'].update(distance=sum(dists) / 1000, time=sum(times), price=sum(prices), transfer_time=sum(trans))

    if cur['type'] is not None:
        path.append(cur)

    return dict(summary=summary, path=path)


def dynamic_weight(attr, ts):
    try:
        return float(attr['time'])
    except ValueError:
        assert attr['time'] == 'dynamic'

        rid, rl = int(attr['id']), float(attr['distance'])
        try:
            speed = dynamic_info.loc[partitions[rid], ts // interval * interval].speed
        except KeyError:
            speed = missing_value

        return rl / speed * 3.6


@app.route('/api/routing', methods=['GET'])
def routing():
    try:
        org, dst, edges, timestamp, pref, total = check_params(request.args)
    except AssertionError as e:
        return error(e.args[0] if len(e.args) > 0 else None)

    ond, dnd = GeneralNode('walk', (org.x, org.y), None, None), GeneralNode('walk', (dst.x, dst.y), None, None)
    o, d = edges.nearest_node(org), edges.nearest_node(dst)
    paths = edges.k_shortest_path(o, d, timestamp, k=total, weight=lambda attr, ts: float(attr['time']))

    try:
        return ok(dict(plans=[gen_plans(cost, ((ond, None),) + pth + ((dnd, None),), edges) for cost, pth in paths]))
    except AssertionError as e:
        return error(e.args[0] if len(e.args) > 0 else None)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='2500', debug=True)
