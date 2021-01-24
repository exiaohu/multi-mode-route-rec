import datetime
import numpy as np
import random
import time
from shapely.geometry import Point

from utils.routes import RoutePlanner, road_net


def random_point_within(poly):
    min_x, min_y, max_x, max_y = poly.bounds
    while True:
        x, y = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
        if poly.contains(Point([x, y])):
            return x, y

def test(region, method):

    src, dst = random_point_within(region), random_point_within(region)
    t = datetime.datetime.utcnow().timestamp() + random.randint(-24 * 60 * 60, +24 * 60 * 60)

    try:
        since = time.perf_counter()
        method(src, dst, t)
        return time.perf_counter() - since
    except:
        return None

rp = RoutePlanner()
rp.road_net = road_net(
    '/home/buaa/data/t_common_base_link_2020M4_sz_#98350/t_common_base_link_98350.shp',
    heuristic_func=lambda _, __: 0
)

def test_all(rp):
    res = list()
    while True:
        r = test(rp.shenzhen, rp)
        if r is not None:
            res.append(r)
            print(len(res), flush=True)

        if len(res) >= 100:
            break
    return res

res = np.array(test_all(rp))
print(res.mean(), res.std(), flush=True)
