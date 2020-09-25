import itertools
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

# date_strs = ['20180808', '20180809', '20180810', '20180811', '20180812', '20180813', '20180814']
date_strs = ['20180801', '20180802', '20180803', '20180804', '20180805', '20180808', '20180809', '20180810', '20180811',
             '20180812', '20180813', '20180814', '20180815', '20180816', '20180817', '20180818', '20180819', '20180820',
             '20180821', '20180822', '20180823', '20180824', '20180825', '20180826', '20180827', '20180828', '20180829',
             '20180830', '20180831']

parsed_trajs_folder = r'/home/huxiao/data/bj_data/bj_traj_parsed'
match_result_folder = r'/home/huxiao/data/bj_data/mr'

road_segments, timestamps, speeds = list(), list(), list()
for date_str in tqdm(date_strs):
    data = pickle.load(open(os.path.join(parsed_trajs_folder, f'bj_traj_parsed-{date_str}.pickle'), 'rb'))
    data_list = [data[key] for key in sorted(data)]

    mr = pd.read_csv(os.path.join(match_result_folder, f'mr-{date_str}.csv'),
                     sep=';', index_col='id', usecols=['id', 'opath'])

    trajs = {i: traj for i, traj in enumerate(itertools.chain(*data_list)) if len(traj) > 1}

    mr['trajs'] = pd.Series(trajs)
    mr.dropna(how='any', inplace=True)
    for _, row in mr.iterrows():
        # ogeom = wkt.loads(row.ogeom).coords.xy
        rss = list(map(int, row.opath.split(',')))
        tss = list(map(lambda i: i['t'], row.trajs))
        sss = list(map(lambda i: i['s'], row.trajs))
        for rs, ts, ss in zip(rss, tss, sss):
            road_segments.append(rs)
            timestamps.append(ts)
            speeds.append(ss)

roads = pd.DataFrame({'segments': road_segments, 'timestamps': timestamps, 'speeds': speeds})
roads['timestamps'] = pd.to_datetime(roads['timestamps'], unit='s')
groupped = roads.groupby([pd.Grouper(key='timestamps', freq='15T'), 'segments']).aggregate(
    {'speeds': lambda s: s.sum() / (s > 0).sum() if (s > 0).sum() > 0 else np.nan})

groupped.loc['201808'].to_parquet(f'taxi_road_speeds-{date_strs[0]}-{date_strs[-1]}.parquet')
