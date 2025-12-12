import numpy as np


def np_mae(predict: np.ndarray, label: np.ndarray) -> np.ndarray:
    mask = label > 0
    diff = np.abs(predict[mask] - label[mask])
    loss = np.mean(diff)
    return loss


def np_rmse(predict: np.ndarray, label: np.ndarray) -> np.ndarray:
    mask = label > 0
    diff = predict[mask] - label[mask]
    loss = np.sqrt(np.mean(diff**2))
    return loss


def np_mare(predict: np.ndarray, label: np.ndarray) -> np.ndarray:
    predict = predict[label > 0]
    label = label[label > 0]
    loss = np.mean(np.abs((predict - label) / label))
    return loss
