import itertools
import os
import pickle
from datetime import datetime

from tqdm import tqdm

# date_strs = ['20180808']
date_strs = ['20180801', '20180802', '20180803', '20180804', '20180805', '20180808', '20180809', '20180810', '20180811',
             '20180812', '20180813', '20180814', '20180815', '20180816', '20180817', '20180818', '20180819', '20180820',
             '20180821', '20180822', '20180823', '20180824', '20180825', '20180826', '20180827', '20180828', '20180829',
             '20180830', '20180831']

parsed_trajs_folder = r'/home/huxiao/data/bj_data/bj_traj_parsed'
match_result_folder = r'/home/huxiao/data/bj_data/mr_cleaned'


def get_records():
    import pandas as pd
    vehicles, road_segments, timestamps, speeds, angles, load_states = list(), list(), list(), list(), list(), list()
    for date_str in tqdm(date_strs):
        mr = pd.read_csv(os.path.join(match_result_folder, f'mr-{date_str}.csv'),
                         sep=';', index_col='id', usecols=['id', 'opath'])

        data = pickle.load(open(os.path.join(parsed_trajs_folder, f'bj_traj_parsed-{date_str}.pickle'), 'rb'))
        data_list = [data[key] for key in sorted(data)]
        for vehicle in data:
            for datums in data[vehicle]:
                for datum in datums:
                    datum['v'] = vehicle
        trajs = {i: datum for i, datum in enumerate(itertools.chain(*data_list)) if len(datum) > 1}

        mr['trajs'] = pd.Series(trajs)
        mr.dropna(how='any', inplace=True)
        for _, row in mr.iterrows():
            if len(row.opath.split(',')) != len(row.trajs):
                print(
                    'vehicle', row['vehicle_id'],
                    'opath', len(row.opath.split(',')),
                    'trajs', len(row.trajs)
                )
                continue
            # ogeom = wkt.loads(row.ogeom).coords.xy
            road_segments.extend(map(int, row.opath.split(',')))
            vehicles.extend(i['v'] for i in row.trajs)
            timestamps.extend(i['t'] for i in row.trajs)
            speeds.extend(i['s'] for i in row.trajs)
            angles.extend(i['a'] for i in row.trajs)
            load_states.extend(i['ls'] for i in row.trajs)

    return pd.DataFrame({
        'vehicles': vehicles,
        'segments': road_segments,
        'timestamps': timestamps,
        'speeds': speeds,
        'angles': angles,
        'load_num': load_states,
        'all_num': load_states
    })


def generate_roads():
    import pandas as pd

    records = pd.read_parquet(f'/home/huxiao/data/bj_data/records-{date_strs[0]}-{date_strs[-1]}.parquet')
    records['timestamps'] = pd.to_datetime(records['timestamps'], unit='s')

    def records_to_roads(group: pd.DataFrame) -> pd.Series:
        # vehicles, speeds, angles, load_num, all_num
        avail = (group[['load_num', 'vehicles']].groupby('vehicles').mean() < 0.5).load_num
        return pd.Series({
            'speed': group.speeds[group.speeds > 0].mean(),
            'available': avail.sum(),
            'total': avail.count()
        })

    tqdm.pandas()
    roads = records.groupby([pd.Grouper(key='timestamps', freq='15T'), 'segments']).progress_apply(records_to_roads)
    roads.loc[datetime(2018, 8, 1): datetime(2018, 8, 31)].to_parquet(f'roads-{date_strs[0]}-{date_strs[-1]}.parquet')


def parallelized_generate_roads():
    import pandas as pd
    import dask.dataframe as dd

    records = dd.read_parquet(f'/home/huxiao/data/bj_data/records-{date_strs[0]}-{date_strs[-1]}.parquet')
    records['timestamps'] = records.timestamps.dt.to_period('15T')

    def records_to_roads(group: pd.DataFrame) -> pd.Series:
        # vehicles, speeds, angles, load_num, all_num
        avail = (group[['load_num', 'vehicles']].groupby('vehicles').mean() < 0.5).load_num
        return pd.Series({
            'speed': group.speeds[group.speeds > 0].mean(),
            'available': avail.sum(),
            'total': avail.count()
        })

    roads = records.groupby(['timestamps', 'segments']).apply(records_to_roads, meta={
        'speed': 'f8',
        'available': 'f8',
        'total': 'f8',
    })
    roads.loc[datetime(2018, 8, 1): datetime(2018, 8, 31)].to_parquet(f'roads-{date_strs[0]}-{date_strs[-1]}.parquet')


# get_records().to_parquet(f'/home/huxiao/data/bj_data/records-{date_strs[0]}-{date_strs[-1]}.parquet')
if __name__ == '__main__':
    generate_roads()
