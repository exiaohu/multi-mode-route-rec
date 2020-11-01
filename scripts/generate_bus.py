import json
from typing import List

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, Point
from tqdm import tqdm


def request_poi(keyword: str, _city='北京'):
    amap_key = '720197cbb21e8b83eefc7a1df0070b20'
    poi_search_url = 'https://restapi.amap.com/v3/place/text'
    if not keyword.endswith('(公交站)'):
        keyword = keyword + '(公交站)'
    res = requests.get(poi_search_url, params={
        'key': amap_key,
        'keywords': keyword,
        'types': '公交车站',
        'city': _city,
        'citylimit': False
    }).json()

    assert res['status'] == '1', f'Request to {keyword} in {_city} failed.'
    if not res['pois'][0]['name'] == keyword:
        print(f"{res['pois'][0]['name']}({res['pois'][0]['location']}) and keyword {keyword} are not exactly matched.")

    return {
        'location': res['pois'][0]['location'],
        'amap_id': res['pois'][0]['id']
    }


class LinesLookup(object):
    def __init__(self, names: List[str]):
        self.keys = names
        self.lookup = dict()

        for name in tqdm(names):
            if len(self[name]) < 2:
                res = self.request_busline(name)
                for _line in res:
                    self.lookup[_line['name']] = _line

    def __getitem__(self, line_no):
        res = list()
        for key in self.lookup:
            name = (key[:key.find('(')] if '(' in key else key).strip()
            if name == line_no:
                res.append(self.lookup[key])

        return res

    @staticmethod
    def request_busline(line_no: str, _city='北京'):
        amap_key = 'bc6bab115306dde0c6b07db965106b7a'
        busline_url = 'https://restapi.amap.com/v3/bus/linename'
        res = requests.get(busline_url, params={
            'key': amap_key,
            's': 'rsv3',
            'extensions': 'all',
            'output': 'json',
            'city': _city,
            'offset': 50,
            'keywords': line_no,
            'platform': 'JS'
        }).json()

        assert res['status'] == '1', f'Request to {line_no} in {_city} failed.'
        return res['buslines']


# if __name__ == '__main__':
bus_lines_path = '/home/huxiao/data/bj_data/公交线路信息.xlsx'
metro_lines_path = '/home/huxiao/data/bj_data/轨道线路信息.xlsx'

bus_lines = set(pd.read_excel(bus_lines_path).线路名称)
metro_lines = set(pd.read_excel(metro_lines_path).线路名称)

ll = LinesLookup(
    list(map(lambda n: str(n) if str(n).endswith('路') or str(n).endswith('线') else str(n) + '路', bus_lines)) + list(
        metro_lines))

with(open('data/bj_public_traffic.json', 'w+')) as f:
    json.dump(ll.lookup, f)
# stops = pd.read_excel(bus_stops_path).to_dict('records')
# extras = {stop: dict() for stop in set(map(lambda item: item['站点名称'], stops))}
# for name in tqdm(extras):
#     try:
#         if 'location' not in extras[name] or 'amap_id' not in extras[name]:
#             extra = request_poi(name)
#             extras[name].update(**extra)
#     except AssertionError as e:
#         print(e)
#     except IndexError as e:
#         print(name, e)

lines = list(ll.lookup.values())

line_cols = [
    'id', 'type', 'name', 'polyline', 'citycode', 'start_stop',
    'end_stop', 'timedesc', 'loop', 'status', 'direc', 'company',
    'distance', 'basic_price', 'total_price', 'bounds'
]
stop_cols = [
    'id', 'name', 'sequence'
]

geo_lines = {col: list() for col in line_cols + ['geometry']}
geo_stops = {col: list() for col in stop_cols + ['geometry', 'line_id']}

for line in lines:
    for col in line_cols:
        geo_lines[col].append(str(line[col]))

    geo_lines['geometry'].append(
        LineString([Point([float(c) for c in p.split(',')]) for p in line['polyline'].split(';')]))

    for stop in line['busstops']:
        for col in stop_cols:
            geo_stops[col].append(str(stop[col]))

        geo_stops['geometry'].append(Point([float(c) for c in stop['location'].split(',')]))
        geo_stops['line_id'].append(str(line['id']))

gpd.GeoDataFrame(geo_lines, crs='EPSG:4326').to_file('data/geo_lines/geo_lines.shp', encoding='utf-8')
gpd.GeoDataFrame(geo_stops, crs='EPSG:4326').to_file('data/geo_stops/geo_stops.shp', encoding='utf-8')
