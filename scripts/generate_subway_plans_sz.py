from itertools import permutations

import pickle
import requests
from tqdm import tqdm


def get_stations():
    url = r'https://www.szmc.net/styles/index/sz-subway/mcdata/shentie.json'
    shentie = requests.get(url).json()

    stations = list()
    for l in shentie['l']:
        for st in l['st']:
            stations.append((st['poiid'], st['n']))

    return stations


def get_plan(dsid, dsname, asid, asname):
    url = r'https://www.szmc.net/algorithm/Ticketing/MinTimeJson.do'
    res = requests.post(url, {
        'departureStation': dsid,
        'arriveStation': asid,
        'departureStationName': dsname,
        'arriveStationName': asname,
        'ridingType': 0
    }).json()

    return res, {
        'time': res['useTime'],  # 秒数
        'price': res['ticketPrice'],  # 元
        'transfer_time': res['times'],  # 次
    }


def main():
    stations = get_stations()
    plans = dict()
    for (dsid, dsname), (asid, asname) in tqdm(permutations(stations, 2)):
        plan, costs = get_plan(dsid, dsname, asid, asname)
        plans[dsname, asname] = plan

    pickle.dump(plans, open('data/subway_plans.pickle', 'wb+'))


if __name__ == '__main__':
    main()
