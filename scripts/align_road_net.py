from copy import deepcopy

import geopandas as gpd
from tqdm import tqdm

road_net = gpd.read_file('/home/huxiao/data/bj_data/bj_small_roads_sample/bj_small_roads_sample.shp')

data = gpd.GeoDataFrame(crs=road_net.crs)


def mapping_row(row):
    global data
    direction, snode_id, enode_id = row['direction'], row['snodeid'], row['enodeid']
    if direction == '0' or direction == '1':
        new_row = deepcopy(row)
        new_row['snodeid'], new_row['enode_id'] = enode_id, snode_id
        data = data.append(row, ignore_index=True)
        data = data.append(new_row, ignore_index=True)
    elif direction == '2':
        data = data.append(row, ignore_index=True)
    elif direction == '3':
        new_row = deepcopy(row)
        new_row['snodeid'], new_row['enode_id'] = enode_id, snode_id
        data = data.append(new_row, ignore_index=True)


for _, row in tqdm(road_net.iterrows()):
    mapping_row(row)

data.to_file('/home/huxiao/data/bj_data/bj_small_roads_sample/bj_small_roads_sample_cleaned.shp')
