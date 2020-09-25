import datetime
import pickle

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from torch.nn.utils import rnn
from torch.utils.data import Dataset

from model import E2ELSTM, E2EMLP
from utils.data import get_dataloaders
from utils.loss import get_loss
from utils.train import train_model, SimpleTrainer, get_optimizer, test_model

data = pickle.load(open(r'data/DS_CPATH_TIME_Y.pk', 'rb'))['DS_CPATH_TIME_Y']
speeds = pd.read_parquet(r'/home/huxiao/data/bj_data/taxi_road_speeds-20180801-20180831.parquet')
road_net = gpd.read_file('/home/huxiao/data/bj_data/bj_small_roads_sample/bj_small_roads_sample.shp')


class DSCpathTtimeYDataset(Dataset):
    def __init__(
            self,
            _data,
            _speeds: pd.DataFrame,
            _net: gpd.GeoDataFrame,
            his_length=datetime.timedelta(days=1),
            time_len=96):
        _speeds = _speeds.speeds.unstack(level=-1).asfreq('15Min')
        fill_nan_with = _speeds.mean().mean()
        _speeds.fillna(fill_nan_with, inplace=True)

        _net = _net.set_index('id')
        _net = _net.drop(columns=_net.columns.difference(['length']))

        self.net = _net
        self.speeds = _speeds
        self.time_len = time_len
        self.his_length = his_length
        self.data = _data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seq, s_time, y = self.data[item]
        seq = [int(v) for v in seq]  # [L]

        s_time = datetime.datetime.fromtimestamp(s_time)  # [T]
        selected_times = slice(s_time - self.his_length, s_time - datetime.timedelta(seconds=1))

        x = list()  # with shape [T, L]
        for node in seq:
            try:
                x.append(self.speeds.loc[selected_times, node].values)
            except KeyError:
                x.append(None)
        real_len = [len(v) for v in x if v is not None][0]
        x = [v if v is not None else np.array([np.nan] * real_len, dtype=np.float32) for v in x]
        x = np.stack(x, axis=-1).astype(np.float32)
        x[np.isnan(x)] = np.nanmean(x)

        assert not np.any(np.isnan(x))

        zeros_shp = (self.time_len - x.shape[0], *x.shape[1:])
        x = np.concatenate([np.zeros(zeros_shp, dtype=np.float32), x], axis=0)

        assert x.shape == (self.time_len, len(seq)), f'expected shape of {(self.time_len, len(seq))}, but got {x.shape}'

        length = self.get_length_of_seq(seq)

        return np.concatenate([x, length], axis=0).transpose((1, 0)), y, length.sum()

    def get_length_of_seq(self, seq) -> np.ndarray:
        return np.array([float(self.net.loc[str(i), 'length']) for i in seq], dtype=np.float32).reshape(1, len(seq))


def collate_variable_seq(batch):
    xs, ys, ls = zip(*batch)
    xs = [torch.tensor(x) for x in xs]
    ys = [torch.tensor([y]) for y in ys]
    ls = [torch.tensor([l]) for l in ls]
    return rnn.pack_sequence(xs, False), torch.stack(ys), torch.stack(ls)


total_size = len(data)

train_size = int(total_size * 0.7)
test_size = int(total_size * 0.15)
valid_size = total_size - train_size - test_size

datasets = {
    'train': DSCpathTtimeYDataset(data[:train_size], speeds, road_net),
    'val': DSCpathTtimeYDataset(data[train_size:train_size + valid_size], speeds, road_net),
    'test': DSCpathTtimeYDataset(data[train_size + valid_size:], speeds, road_net)
}

model = E2EMLP(96 + 1, [256], 1)
# model = E2ELSTM(96 + 1, 32, [256, 512], 1)
trainer = SimpleTrainer(
    model,
    loss=get_loss('MaskedMAELoss'),
    device=torch.device('cuda:1'),
    optimizer=get_optimizer('Adam', model.parameters(), lr=0.0001),
    max_grad_norm=5
)
dls = get_dataloaders(datasets, batch_size=64, collate_fn=collate_variable_seq)
train_model(
    dls,
    folder='run',
    trainer=trainer,
    scheduler=None,
    epochs=100,
    early_stop_steps=None
)
test_model(
    dls['test'],
    trainer=trainer,
    folder='run',
)
