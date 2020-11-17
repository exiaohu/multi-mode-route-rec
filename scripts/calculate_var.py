import json
import numpy as np
from numpy import ma
from statsmodels.tsa.vector_ar.var_model import VAR

from utils.bj_dataset import BJSpeedDataset
from utils.data import ZScoreScaler
from utils.evaluate import masked_rmse_np, masked_mape_np, masked_mae_np
from utils.helper import JsonEncoder


def var_predict(data, n_forwards=(1, 3), n_lags=4, test_ratio=0.2):
    """
    Multivariate time series forecasting using Vector Auto-Regressive Model.
    :param data: numpy.ndarray of shape [T, N].
    :param n_forwards: a tuple of horizons.
    :param n_lags: the order of the VAR model.
    :param test_ratio:
    :return: [list of prediction in different horizon], dt_test
    """
    # data += np.random.standard_normal(data.shape)

    n_sample, _ = data.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    data_train, data_test = data[:n_train], data[n_train:]

    to_remove = ~np.all(data_train[1:] == data_train[:-1], axis=0)
    data, data_train, data_test = data[:, to_remove], data_train[:, to_remove], data_test[:, to_remove]
    _, n_output = data.shape

    scaler = ZScoreScaler(mean=np.nanmean(data_train), std=np.nanstd(data_train))
    var_model = VAR(scaler.transform(data_train, nan_val=0.0))
    var_result = var_model.fit(n_lags)
    max_n_forwards = np.max(n_forwards)
    # Do forecasting.
    result = np.zeros(shape=(len(n_forwards), n_test, n_output))
    start = n_train - n_lags - max_n_forwards + 1
    for input_ind in range(start, n_sample - n_lags):
        datum = scaler.transform(data[input_ind: input_ind + n_lags], 0.0)
        prediction = var_result.forecast(datum, max_n_forwards)
        for i, n_forward in enumerate(n_forwards):
            result_ind = input_ind - n_train + n_lags + n_forward - 1
            if 0 <= result_ind < n_test:
                result[i, result_ind, :] = prediction[n_forward - 1, :]

    data_predicts = []
    for i, n_forward in enumerate(n_forwards):
        data_predict = scaler.inverse_transform(result[i], nan_val=0.0)
        data_predicts.append(data_predict)
    return data_predicts, data_test


def eval_var(data, n_lags=3):
    n_forwards = list(range(1, 13))
    data = data[:, ~np.all(data[1:] == data[:-1], axis=0)]  # delete constant columns
    y_predicts, y_tests = var_predict(data, n_forwards=n_forwards, n_lags=n_lags, test_ratio=0.2)

    print('VAR (lag=%d)' % n_lags)
    print('Model\t\t\tHorizon\t\t\tRMSE\t\t\tMAPE\t\t\tMAE')
    maes, rmses, mapes = dict(), dict(), dict()
    for i, horizon in enumerate(n_forwards):
        rmse = masked_rmse_np(preds=y_predicts[i], labels=y_tests, null_val=0.)
        mape = masked_mape_np(preds=y_predicts[i], labels=y_tests, null_val=0.)
        mae = masked_mae_np(preds=y_predicts[i], labels=y_tests, null_val=0.)
        print('VAR\t\t\t%d\t\t\t%.9f\t\t\t%.9f\t\t\t%.9f' % (horizon, rmse, mape, mae))

        maes.update({horizon - 1: mae})
        mapes.update({horizon - 1: mape})
        rmses.update({horizon - 1: rmse})

    return {'mae': maes, 'rmse': rmses, 'mape': mapes}


if __name__ == '__main__':
    fill_with_mean = False
    ds = BJSpeedDataset('train')
    roads, road_net, id_to_ts = ds._states, ds._net, sorted(ds._id_to_ts.values())

    if fill_with_mean:
        roads[roads == 0.] = np.nan
        fillva = ma.array(roads, mask=np.isnan(roads)).mean(axis=1)
        n, d = fillva.shape
        fillva = np.reshape(fillva, (n, 1, d))
        roads = np.where(np.isnan(roads), fillva, roads)
    results = dict()
    for n_lags in [1]:
        print('*' * 100, 'speed', '*' * 100)
        results['speed'] = eval_var(roads[..., 0].transpose((1, 0)), n_lags=n_lags)

        print('*' * 100, 'avail', '*' * 100)
        results['available'] = eval_var(roads[..., 1].transpose((1, 0)), n_lags=n_lags)

        print('*' * 100, 'END', '*' * 100)

        print(json.dumps(results, indent=4, cls=JsonEncoder))
        json.dump(results, open('run/var_results.json', 'w+'), indent=4, cls=JsonEncoder)
