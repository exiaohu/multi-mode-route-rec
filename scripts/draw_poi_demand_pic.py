import pickle
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
font = FontProperties()
font.set_size('xx-small')

colormap = 'tab20'

pdm = 'data/draw-data/poi_demand_match.pickle'
pdm = pickle.load(open(pdm, 'rb'))

src, dst = defaultdict(set), defaultdict(set)
for (d, p, t), v in pdm.items():
    if d == 'src':
        src[p].add((t, v))
    elif d == 'dst':
        dst[p].add((t, v))

src = {k: [i[1] for i in sorted(v, key=lambda a: a[0])] for k, v in src.items()}
dst = {k: [i[1] for i in sorted(v, key=lambda a: a[0])] for k, v in dst.items()}
cols = set(src.keys()).intersection(dst.keys())
src = pd.DataFrame(src, index=range(1, 24), columns=cols)
dst = pd.DataFrame(dst, index=range(1, 24), columns=cols)


def draw(name, data, ftype):
    figure = plt.figure()
    data.plot(kind=ftype, colormap=colormap, ax=figure.gca())
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.xlabel(name)
    plt.savefig(f'data/plot/{name}.png', bbox_extra_artists=(lgd,), bbox_inches='tight')


# draw('Pickup', src, 'area')
# draw('Dropout', dst, 'area')
for col in cols:
    draw(col, pd.DataFrame({
        'Pickup': src[col],
        'Dropout': dst[col]
    }), 'line')
