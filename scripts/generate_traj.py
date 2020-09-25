import os
import pandas as pd
import osr
import shapefile  # 使用pyshp

MAXLNG = 116.496266
MINLNG = 116.271389

MAXLAT = 39.989214
MINLAT = 39.832525

data_address = "tra_small.shp"
shp_file = shapefile.Writer(data_address)

# 创建两个字段
shp_file.field('id')
shp_file.field('geom')  # 'SECOND_FLD'为字段名称，C代表数据类型为字符串， 长度为40

TaxiFolder = '/home/LAB/data/transport data/beijing_taxi_GPS'

datestr = '20180801'

files = os.listdir(os.path.join(TaxiFolder, datestr))

dfs = []

for fname in files[0:1]:
    if '.txt' not in fname:
        continue
    print(fname, end='\t')
    full_fname = os.path.join(TaxiFolder, datestr, fname)
    df_i = pd.read_csv(full_fname, encoding='ascii', header=None, error_bad_lines=False)
    dfs.append(df_i)
    # print(df[3].value_counts())
    # break

df = pd.concat(dfs, axis=0, ignore_index=True)

df = df[~(df.isnull().T.any())]

df = df[(df[4] >= MINLNG) & (df[4] <= MAXLNG) & (df[5] >= MINLAT) & (df[5] <= MAXLAT)]

df[11] = df.groupby(3)[3].transform('count')
df = df.sort_values([11, 3, 0, 1], ascending=[0, 1, 1, 1])

cur = 0

data = df.values

old_lat = None

old_lng = None

old_driver_id = None

cur_geo_list = []

# fw_tra = open('tra_test.csv','w')

# fw_tra.write("id;geom\n")

tra_id = 1

for row in data:
    if row[6] == 0:
        if len(cur_geo_list) >= 3:
            # shp_line = str(tra_id)+";"+"LINESTRING("+(",".join([" ".join([str(num) for num in geo[3:5]]) for geo in cur_geo_list]))+")"
            shp_file.line([[geo[2:4] for geo in cur_geo_list]])
            shp_file.record(tra_id, 'polyline')
            tra_id += 1
            cur_geo_list = []
        old_driver_id = None
        continue
    driver_id = row[3]
    if driver_id == old_driver_id:
        cur_geo_list.append(row[0:2].tolist() + row[4:6].tolist())
    else:
        if len(cur_geo_list) >= 3:
            # shp_line = str(tra_id)+";"+"LINESTRING("+(",".join([" ".join([str(num) for num in geo[3:5]]) for geo in cur_geo_list]))+")"
            shp_file.line([[geo[2:4] for geo in cur_geo_list]])
            shp_file.record(tra_id, 'polyline')
            tra_id += 1
            cur_geo_list = []
        old_driver_id = driver_id

if len(cur_geo_list) >= 3:
    shp_line = str(tra_id) + ";" + "LINESTRING(" + (
        ",".join([" ".join([str(num) for num in geo[3:5]]) for geo in cur_geo_list])) + ")"
    shp_file.line([[geo[2:4] for geo in cur_geo_list]])
    shp_file.record(tra_id, 'polyline')
    tra_id += 1

# fw_tra.close()
shp_file.close()

proj = osr.SpatialReference()
proj.ImportFromEPSG(4326)  # 4326-GCS_WGS_1984; 4490- GCS_China_Geodetic_Coordinate_System_2000
wkt = proj.ExportToWkt()
# 写入投影
f = open(data_address.replace(".shp", ".prj"), 'w')
f.write(wkt)
f.close()

