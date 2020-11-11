import json
import numpy as np

from utils.bj_dataset import BJSpeedDataset
from utils.evaluate import masked_rmse_np, masked_mape_np, masked_mae_np
from utils.helper import JsonEncoder


def naive_historical_average_predict(road_states, _):
    _, t = road_states.shape

    train_ratio, test_ratio = (6 + 2, 2)
    divider = int(t * (train_ratio / (train_ratio + test_ratio)))

    road_states[road_states == 0.] = np.nan

    preds = np.nanmean(road_states[:, :divider], axis=1)

    targets, predictions = list(), list()
    for i in range(divider, t):
        targets.append(np.nan_to_num(road_states[:, i]))
        predictions.append(np.nan_to_num(preds))

    targets, predictions = np.concatenate(targets), np.concatenate(predictions)
    return targets, predictions


def historical_average_predict(road_states, id_to_ts):
    train_ratio, test_ratio = (6 + 2, 2)

    road_states[road_states == 0.] = np.nan

    data = list(road_states.transpose((1, 0)))

    divider, end = int(len(data) * (train_ratio / (train_ratio + test_ratio))), len(data)

    hours_preds = {i: list() for i in range(24)}
    for i in range(0, divider):
        hours_preds[id_to_ts[i].hour].append(data[i])

    for i in range(24):
        hours_preds[i] = np.nanmean(np.stack(hours_preds[i], axis=0), axis=0)

    targets, predictions = list(), list()
    for i in range(divider, end):
        targets.append(np.nan_to_num(data[i]))
        predictions.append(np.nan_to_num(hours_preds[id_to_ts[i].hour]))

    targets, predictions = np.concatenate(targets), np.concatenate(predictions)
    return targets, predictions


def eval_historical_average(road_states, id_to_ts):
    # y_test, y_predict = historical_average_predict(road_states, id_to_ts)
    y_test, y_predict = naive_historical_average_predict(road_states, id_to_ts)
    rmse = masked_rmse_np(preds=y_predict, labels=y_test, null_val=0)
    mape = masked_mape_np(preds=y_predict, labels=y_test, null_val=0)
    mae = masked_mae_np(preds=y_predict, labels=y_test, null_val=0)
    print('Historical Average')
    print('\t\t\t'.join(['Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
    for horizon in range(12):
        line = 'HA\t\t\t%d\t\t\t%.9f\t\t\t%.9f\t\t\t%.9f' % (horizon, rmse, mape, mae)
        print(line)
    return {
        'mae': {i: mae for i in range(12)},
        'rmse': {i: rmse for i in range(12)},
        'mape': {i: mape for i in range(12)}
    }


if __name__ == '__main__':
    ds = BJSpeedDataset('train')

    print('*' * 100, 'speed', '*' * 100)
    speed_res = eval_historical_average(ds._states[..., 0], ds._id_to_ts)
    print('*' * 100, 'avail', '*' * 100)
    avail_res = eval_historical_average(ds._states[..., 1], ds._id_to_ts)
    print('*' * 100, 'END', '*' * 100)

    results = {'speed': speed_res, 'available': avail_res}
    print(json.dumps(results, indent=4, cls=JsonEncoder))
    json.dump(results, open('run/ha_results.json', 'w+'), indent=4, cls=JsonEncoder)
