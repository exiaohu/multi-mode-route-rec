import json

import geopandas as gpd
from shapely.geometry import Point, LineString

sws = json.load(open('../data/beijing_subway.json'))

lines, stations = list(), list()
for line in sws['l']:
    l = list()
    for i, stop in enumerate(line['st']):
        s = Point(*map(float, stop['sl'].split(',')))
        stations.append({
            'name': stop['n'],
            'line_name': line['kn'],
            'geometry': s,
            'sequence': i,
            **stop
        })
        l.append(s)
    lines.append({
        'name': line['kn'],
        'geometry': LineString(l)
    })

gpd.GeoDataFrame(lines, crs='EPSG:4326').to_file('../data/beijing_subway/bj_subway_lines.shp', encoding='utf-8')
gpd.GeoDataFrame(stations, crs='EPSG:4326').to_file('../data/beijing_subway/bj_subway_stops.shp', encoding='utf-8')
