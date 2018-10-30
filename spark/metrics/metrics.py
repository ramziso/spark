import numpy as np
import pandas as pd
from sklearn import metrics

def _rank_data_on_panda(preds):
    ranks = pd.DataFrame(preds)
    return ranks.rank(axis=1, ascending = False).values.astype(np.int8) - 1

def top_n_accuracy(preds, truths, n):
    # very naive approach
    preds = _rank_data_on_panda(preds)
    best_n = np.argsort(preds, axis=1)[:, -n:]
    ts = np.argmax(truths, axis=1)
    successes = 0
    for i in range(ts.shape[0]):
        if ts[i] in best_n[i, :]:
            successes +=1
    return float(successes)/ts.shape[0]

def log_loss(preds, truths):
    return metrics.log_loss(truths, preds)