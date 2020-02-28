import numpy as np
from numba import jit


@jit
def auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc_score = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc_score += y_i * nfalse
    auc_score /= (nfalse * (n - nfalse))
    return auc_score


def eval_auc(preds, dtrain):
    """
    For LightGBM eval func
    """
    labels = dtrain.get_label()
    return 'auc', auc(labels, preds), True
