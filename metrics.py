import numpy as np

import sklearn.metrics

def calculate_metrics(ytrue, ypred):
    auc_roc = sklearn.metrics.roc_auc_score(ytrue, ypred)
    auc_pr = sklearn.metrics.average_precision_score(ytrue, ypred)
    acc = sklearn.metrics.accuracy_score(ytrue, ypred >= 0.5)
    bacc = sklearn.metrics.balanced_accuracy_score(ytrue, ypred >= 0.5)
    rand_probs = ypred.copy()
    np.random.shuffle(rand_probs)
    auc_pr_null = sklearn.metrics.average_precision_score(ytrue, rand_probs)

    return {
        "auc_roc" : auc_roc,
        "auc_pr" : auc_pr,
        "bacc" : bacc,
        "acc" : acc, 
        "auc_pr_null" : auc_pr_null
    }
