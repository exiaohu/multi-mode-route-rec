import geopandas as gpd
import pandas as pd

if __name__ == '__main__':
    lines = gpd.read_file('/home/huxiao/data/bj_data/北京公交线路/BeijingBusLines.shp', encoding='GB2312')
    stops = gpd.read_file('/home/huxiao/data/bj_data/北京公交线路/BeijingBusStops.shp', encoding='GB2312')

    lines['line'] = lines.name.apply(lambda name: name[:name.find('(')] if '(' in name else name)
    stops['line'] = stops.line.apply(lambda name: name[:name.find('(')] if '(' in name else name)

    res = pd.merge(lines, stops, on='line', suffixes=('_line', '_stop'))
