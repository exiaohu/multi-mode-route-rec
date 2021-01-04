import os
import pandas as pd
import pickle
from tqdm import tqdm

link_speed_dir = r'/home/buaa/data/link_speed'

# FDATE 日期 yyyymmdd
# PERIOD 5分钟时间片 1-288
# LINK_FID 路链ID
# FROM_NODE link的起始node
# TO_NODE link的到达node
# SPEED 车速 km/h
# ,fdate,period,link_fid,from_node,to_node,speed

records = dict()
for name in tqdm(sorted(os.listdir(link_speed_dir))):
    fullname = os.path.join(link_speed_dir, name)
    datum = pd.read_csv(fullname, usecols=['period', 'link_fid', 'speed'])

    for d in datum.itertuples(False):
        period, link_id, speed = int(d.period), int(d.link_fid), float(d.speed)
        if speed > 0:
            speed_sum, speed_counter = records.get((period, link_id), (0., 0))
            records[(period, link_id)] = speed_sum + speed, speed_counter + 1

for record_key in records:
    speed_sum, speed_counter = records[record_key]
    records[record_key] = speed_sum / speed_counter

pickle.dump(records, open('data/link_speed_ha.pickle', 'wb+'))
