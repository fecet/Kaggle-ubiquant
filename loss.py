import tensorflow as tf
import numpy as np

from scipy.stats import pearsonr


def pearson_corr(a_true, a_pred):
    score = pearsonr(a_pred, a_true)[0]
    return score


def _mape(a_true, a_pred):
    # weights = 1/a_true*mask
    res = tf.abs(a_pred - a_true) / (a_true + 1e-8)
    # res=es*res
    # nonzero=tf.math.count_nonzero(res)
    # nonzero=tf.cast(nonzero,tf.float32)
    return 100 * res


def _smape(a_true, a_pred):
    # weights = 1/a_true*mask
    res = tf.abs(a_pred - a_true) / (tf.abs(a_true) + tf.abs(a_pred) + 1e-8)
    # res=es*res
    # nonzero=tf.math.count_nonzero(res)
    # nonzero=tf.cast(nonzero,tf.float32)
    return 200 * res


def mape(a_true, a_pred):
    return tf.reduce_mean(_mape(a_true, a_pred))


def es_mape(a_true, a_pred):
    horizon = a_true.shape[-1]
    es = 0.7 ** tf.range(horizon, 0.0, delta=-1)
    es = es / tf.reduce_sum(es)
    res = es * _mape(a_true, a_pred)
    return tf.reduce_mean(res) * horizon


def last_mape(a_true, a_pred):
    # a_true=a_true[...,-1]
    # a_pred=a_pred[...,-1]
    horizon = a_true.shape[-1]
    a = np.zeros(horizon)
    a[-1] = 1
    res = a * _mape(a_true, a_pred)
    return tf.reduce_mean(res) * horizon


def smape(a_true, a_pred):
    return tf.reduce_mean(_smape(a_true, a_pred))


def es_smape(a_true, a_pred):
    horizon = a_true.shape[-1]
    es = 0.7 ** tf.range(horizon, 0.0, delta=-1)
    es = es / tf.reduce_sum(es)
    res = es * _smape(a_true, a_pred)
    return tf.reduce_sum(res)


def last_smape(a_true, a_pred):
    # a_true=a_true[...,-1]
    # a_pred=a_pred[...,-1]
    a = np.zeros(10)
    a[-1] = 1
    res = a * _smape(a_true, a_pred)
    return tf.reduce_sum(res)


LOSS = {
    "smape": smape,
    "last_smape": last_smape,
    "es_smape": es_smape,
    "mape": mape,
    "last_mape": last_mape,
    "es_mape": es_mape,
    "corr": pearson_corr
}
