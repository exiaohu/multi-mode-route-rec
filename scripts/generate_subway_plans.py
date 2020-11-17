import json
import pickle
from itertools import chain, permutations
from xml.etree import ElementTree as ET

import geopandas as gpd
import requests
from tqdm import tqdm


def parse_code_to_name(p='../data/beijing_subway/beijing.xml'):
    root = ET.parse(p)
    code_to_name = {int(l.get('lcode')): l.get('lb') for l in root.findall('l')}
    return code_to_name


def get_plan(start, end, code_to_name):
    url = r'https://map.bjsubway.com/searchstartend?={}&end={}'
    res = requests.get(url, {'start': start, 'end': end}).json()
    if res['result'] != 'success':
        raise ValueError(f'Get plan from {start} to {end} failed.')

    plans, price = json.loads(res['fangan']), res['price']

    extext = ''
    assert len(plans) > 0
    plan = plans[0]
    del plans

    time, lines = plan['m'], plan['p']

    for item in chain(*lines):
        item[0] = code_to_name[int(item[0])]

    for line in lines:
        if extext != '':
            extext += '换乘'

        extext += f'{line[0][0]}（{line[0][1]}－{line[-1][1]}，{len(line)}站），'

    extext = f'【地铁方案】{lines[0][0][1]}－{lines[-1][-1][1]}：{extext}预计时间{time}分钟，票价{price}元。'

    return extext, lines, {
        'time': time,
        'price': price,
        'transfer_time': len(lines) - 1,
        'n_stations': sum(map(len, lines)) - len(lines)
    }


def get_subway_stops(stops='../data/beijing_geo/wgs_subway_stops.geojson'):
    return set(gpd.read_file(stops).name)


def get_subway_lines(lines='../data/beijing_geo/wgs_subway_lines.geojson'):
    return set(gpd.read_file(lines).name)


def main():
    code_to_name = parse_code_to_name()

    stops = get_subway_stops()
    stops.remove('T2航站楼')
    stops.add('2号航站楼')
    stops.remove('T3航站楼')
    stops.add('3号航站楼')

    plans = dict()
    for i, j in tqdm(list(permutations(stops, 2))):
        if i == j:
            continue
        try:
            desc, plan, cost = get_plan(i, j, code_to_name)
            plans[(i, j)] = {
                'description': desc,
                'plan': plan,
                'cost': cost
            }
        except ValueError as e:
            print(e)

    pickle.dump(plans, open('../data/beijing_subway/ori_subway_plans.pickle', 'wb+'))


def clean_data():
    line_mapping = {
        '10号线': '地铁10号线',
        '13号线': '地铁13号线',
        '14号线(东)': '地铁14号线东段',
        '14号线(西)': '地铁14号线西段',
        '15号线': '地铁15号线',
        '16号线': '地铁16号线',
        '1号线': '地铁1号线',
        '2号线': '地铁2号线',
        '4号线大兴线': '地铁4号线大兴线',
        '5号线': '地铁5号线',
        '6号线': '地铁6号线',
        '7号线': '地铁7号线',
        '8号线北': '地铁8号线',
        '8号线南': '地铁8号线南段',
        '9号线': '地铁9号线',
        'S1线': 'S1线',
        '亦庄线': '地铁亦庄线',
        '八通线': '地铁八通线',
        '大兴机场线': '北京大兴国际机场线',
        '房山线': '地铁房山线',
        '昌平线': '地铁昌平线',
        '燕房线': '地铁燕房线',
        '西郊线': '西郊线',
        '首都机场线': '首都机场线'
    }
    stop_mapping = {
        '2号航站楼': 'T2航站楼',
        '3号航站楼': 'T3航站楼',
    }

    plans = pickle.load(open('../data/beijing_subway/ori_subway_plans.pickle', 'rb'))
    for s in tqdm(chain(*chain(*[p['plan'] for p in plans.values()]))):
        s[0] = line_mapping[s[0]]
        if s[1] in stop_mapping.keys():
            s[1] = stop_mapping[s[1]]

    for start, end in filter(lambda se: se[0] in stop_mapping.keys() or se[1] in stop_mapping.keys(), plans.keys()):
        plan = plans.pop((start, end))

        if start in stop_mapping.keys():
            start = stop_mapping[start]
        if end in stop_mapping.keys():
            end = stop_mapping[end]

        plans[(start, end)] = plan

    pickle.dump(plans, open('../data/beijing_subway/subway_plans.pickle', 'wb+'))


if __name__ == '__main__':
    main()
    clean_data()
