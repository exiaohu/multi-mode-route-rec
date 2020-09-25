import json
import os
import re
import shutil

import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm

from utils import coord_to_hexagon_id, id2hexagon_lookup

chunksize = 1000

taxi_dir = '/mnt/windows-D/dateshenzhen/'
to_path = '../data/total_data.hdf5'


def parse_raw(raw, index):
    if isinstance(raw, str):
        with open(raw) as raw_file:
            raw = json.load(raw_file)

    header = ['vehicle_id', 'loc_time', 'lng', 'lat', 'speed', 'angle']
    rows = list(map(lambda row: row['row'], raw['page']['content']))
    _data = pd.DataFrame(data=rows, columns=header)
    _data.loc_time = pd.to_datetime(_data.loc_time, unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
    _data.lng, _data.lat = _data.lng.astype(float), _data.lat.astype(float)
    _data.speed = _data.speed.astype(float)
    _data.to_parquet(f'../data/taxi-{index}.parquet', index=False)


def convert_raw_to_dataframe(condition, index):
    for name in tqdm(sorted(filter(condition, os.listdir(taxi_dir)))):
        parse_raw(os.path.join(taxi_dir, name), index=index)


def aggregate(glob, freq):
    ddf = dd.read_parquet(glob, columns=['vehicle_id', 'loc_time', 'lng', 'lat', 'speed'])
    print(ddf.partitions())
    ddf['hex'] = ddf.apply(lambda row: coord_to_hexagon_id(row.lng, row.lat),
                           axis='columns',
                           meta=pd.Series(dtype='int', name='hex'))
    mean_lt_zero = dd.Aggregation(
        'mean_lt_zero',
        lambda s: (s.apply(lambda v: sum(v > 0)), s.sum()),
        lambda c, s: (c.sum(), s.sum()),
        lambda c, s: s / c
    )
    count_unique = dd.Aggregation(
        'count_unique',
        lambda s: s.apply(lambda v: list(set(v))),
        lambda s: s._selected_obj.groupby(level=list(range(s._selected_obj.index.nlevels))).sum(),
        lambda s: s.apply(lambda v: len(set(v)))
    )
    _data = ddf.groupby([pd.Grouper(key='loc_time', freq=freq), 'hex'])
    return _data.aggregate({'vehicle_id': count_unique, 'speed': mean_lt_zero}).compute()


if __name__ == '__main__':
    # convert_raw_to_dataframe(
    #     condition=lambda name: int(re.findall(r'\d+', name)[0]) == 3,
    #     index=lambda name: int(re.findall(r'\d+', name)[2])
    # )

    # data = aggregate(['../data/taxi-1552147200.parquet', '../data/taxi-1552255200.parquet'], freq='30T')
    # data = aggregate(['../data/taxi-1552257000.parquet', '../data/taxi-1552255200.parquet'], freq='30T')
    data = aggregate('../data/taxi-*.parquet', freq='5T')
    print(data.head())
    shutil.rmtree('../data/hexagon.json', ignore_errors=True)
    pd.Series(id2hexagon_lookup).to_json('../data/hexagon.json')
    shutil.rmtree('../data/data.parquet', ignore_errors=True)
    data.to_parquet('../data/data.parquet')

    # for name in os.listdir('../data'):
    #     if re.fullmatch(r'taxi-\d+.parquet', name):
    #         data = pd.read_parquet(os.path.join('../data', name))
    #         print(name, data.columns, data.shape)
