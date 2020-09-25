import itertools
import os
import pickle
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from coord_convert.transform import wgs2gcj
from shapely.geometry import mapping, shape, LineString
from tqdm import tqdm


def wgs2gcj_for_iterable(lngs, lats):
    rlngs, rlats = list(), list()
    for lng, lat in zip(lngs, lats):
        rlng, rlat = wgs2gcj(lng, lat)
        rlngs.append(rlng)
        rlats.append(rlat)
    return rlngs, rlats


def parse_to_int(v) -> np.int32:
    try:
        return np.int32(v)
    except ValueError:
        return np.int32()


def parse_to_float(v) -> np.float64:
    try:
        return np.float64(v)
    except ValueError:
        return np.nan


MAX_LNG, MIN_LNG = 116.495, 116.265
MAX_LAT, MIN_LAT = 39.995, 39.820

# 时间间隔阈值 100s
time_interval_threshold = 100

bj_taxi_folder = '/home/LAB/data/transport data/beijing_taxi_GPS'
date_strs = ['20180801', '20180802', '20180803', '20180804', '20180805', '20180808', '20180809', '20180810', '20180811',
             '20180812', '20180813', '20180814', '20180815', '20180816', '20180817', '20180818', '20180819', '20180820',
             '20180821', '20180822', '20180823', '20180824', '20180825', '20180826', '20180827', '20180828', '20180829',
             '20180830', '20180831']
# date_strs = ['20180801']
name_pattern = r'\d+_\d+.txt'
column_names = ['gen_date', 'gen_time', 'city_id', 'vehicle_id', 'lng', 'lat',
                'speed', 'angle', 'load_state', 'st_validate', 'recv_dt']
column_used = ['gen_date', 'gen_time', 'vehicle_id', 'lng', 'lat', 'speed']
data_converters = {
    'vehicle_id': parse_to_int,
    'lng': parse_to_float,
    'lat': parse_to_float,
    'speed': parse_to_float
}

crs_lookup = {'GCJ02': '', 'WGS84': 'EPSG:4326'}

for date_str in date_strs[5:]:
    data = list()
    data_dir = os.path.join(bj_taxi_folder, date_str)
    print('data collecting...', data_dir)
    for name in tqdm(os.listdir(data_dir)):
        if not re.fullmatch(name_pattern, name):
            continue
        fp = os.path.join(bj_taxi_folder, date_str, name)
        try:
            datum = pd.read_csv(fp, encoding='ascii', header=None, usecols=column_used, names=column_names,
                                error_bad_lines=False, infer_datetime_format=True,
                                converters=data_converters, parse_dates={'gen_dt': ['gen_date', 'gen_time']})
        except UnicodeDecodeError:
            datum = pd.read_csv(fp, encoding='latin1', header=None, usecols=column_used, names=column_names,
                                error_bad_lines=False, infer_datetime_format=True,
                                converters=data_converters, parse_dates={'gen_dt': ['gen_date', 'gen_time']})

        datum.dropna(axis=0, how='any', inplace=True)
        datum['gen_dt'] = pd.to_datetime(datum.gen_dt, format='%Y%m%d %H%M%S', errors='coerce')
        datum = datum[~pd.isnull(datum.gen_dt)]
        datum = datum[(datum.lng >= MIN_LNG) & (datum.lng <= MAX_LNG)
                      & (datum.lat >= MIN_LAT) & (datum.lat <= MAX_LAT)]
        data.append(datum)

    print('data shuffling...', data_dir)
    data = pd.concat(data, axis=0, ignore_index=True)
    data = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(*wgs2gcj_for_iterable(data.lng, data.lat)),
        crs=crs_lookup['WGS84']
    )
    data.drop(columns=['lng', 'lat'], inplace=True)
    data.sort_values('gen_dt', inplace=True)


    def parse_vehicle(geometry, dt, speed):
        res, cur = list(), list()
        for p, t, s in zip(geometry, dt, speed):
            item = {'p': mapping(p), 't': t.timestamp(), 's': s}
            if len(cur) == 0 or item['t'] - cur[-1]['t'] <= 100:
                cur.append(item)
            else:
                cur = [item]
        if len(cur) > 0:
            res.append(cur)
        return res


    print('data parsing...', data_dir)
    data = data.groupby(['vehicle_id']).apply(
        lambda vehicle: parse_vehicle(vehicle.geometry, vehicle.gen_dt, vehicle.speed)
    )
    data = data.to_dict()
    # CHECKPOINT 1
    # json.dump(data, open(f'bj_traj_parsed-{date_str}.json', 'w'))
    pickle.dump(data, open(f'bj_traj_parsed/bj_traj_parsed-{date_str}.pickle', 'wb+'))
    print('parsed trajectories saved as', f'bj_traj_parsed-{date_str}.pickle')

    data_list = [data[key] for key in sorted(data)]
    gdata = gpd.GeoSeries({i: LineString(shape(p['p']) for p in traj)
                           for i, traj in enumerate(itertools.chain(*data_list)) if len(traj) > 1})
    gdata = gpd.GeoDataFrame({'geometry': gdata}, crs=crs_lookup['WGS84'])
    gdata.index.rename('id', inplace=True)
    # # CHECKPOINT 2
    gdata.to_file(f'trajs/trajs-{date_str}.shp')
    print('parsed trajectories saved as', f'trajs/trajs-{date_str}.shp')
