import numpy as np
import torch as t


def RSE(pred, true):
    return t.sqrt(t.sum((true - pred) ** 2)) / t.sqrt(t.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = t.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return t.mean(t.abs(pred - true))


def MSE(pred, true):
    return t.mean((pred - true) ** 2)


def RMSE(pred, true):
    return t.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return t.mean(t.abs((pred - true) / true))


def MSPE(pred, true):
    return t.mean(t.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
