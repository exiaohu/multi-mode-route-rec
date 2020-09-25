import numpy as np


def evaluate(predictions: np.ndarray, targets: np.ndarray):
    """
    evaluate model performance
    :param predictions: [n_samples, 12, n_nodes, n_features]
    :param targets: np.ndarray, shape [n_samples, 12, n_nodes, n_features]
    :return: a dict [str -> float]
    """
    return {
        'mae': mae_np(predictions, targets),
        'rmse': rmse_np(predictions, targets),
        'mape': mape_np(predictions, targets),
        'loss': masked_mae_np(predictions, targets, null_val=0.0)
    }


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return np.mean(mse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def rmse_np(y_pred, y_true):
    return np.mean(np.square(y_true - y_pred)) ** 0.5


def mape_np(y_pred, y_true):
    return np.mean(np.abs(y_true - y_pred) / y_true)


def mae_np(y_pred, y_true):
    return np.mean(np.abs(y_true - y_pred))
