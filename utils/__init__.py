import math

from .hexagon import Layout, layout_pointy, Point, pixel_to_hex, hex_round

hexagon2id_lookup = dict()
id2hexagon_lookup = dict()

unit_length = 0.001
l, t, r, b = 113.751447, 22.396362, 114.627798, 22.858504
layout = Layout(
    orientation=layout_pointy,
    size=Point(math.sqrt(3.) * unit_length, 2. * unit_length),
    origin=Point((l + r) / 2, (t + b) / 2)
)


def coord_to_hexagon_id(p_x, p_y) -> int:
    res = hex_round(pixel_to_hex(layout, Point(x=p_x, y=p_y)))
    if res not in hexagon2id_lookup:
        hid = len(hexagon2id_lookup)
        hexagon2id_lookup[res] = hid
        id2hexagon_lookup[hid] = res
    return hexagon2id_lookup[res]

# def dijkstra(edges, f, t):
#     g = defaultdict(list)
#     for l, r, c in edges:
#         g[l].append((c, r))
#
#     # dist records the min value of each node in heap.
#     q, seen, dist = [(0, f, ())], set(), {f: 0}
#     while q:
#         (cost, v1, path) = heappop(q)
#         if v1 in seen:
#             continue
#
#         seen.add(v1)
#         path += (v1,)
#         if v1 == t:
#             return cost, path
#
#         for c, v2 in g.get(v1, ()):
#             if v2 in seen:
#                 continue
#             # Not every edge will be calculated. Edges which can improve the value of node in heap will be useful.
#             if v2 not in dist or cost + c < dist[v2]:
#                 dist[v2] = cost + c
#                 heappush(q, (cost + c, v2, path))
#     return float("inf")
