import numpy as np
from sklearn.metrics import fbeta_score

def make_default_targets(y, target_names):
    default_targets = []
    for target in target_names[y]:
        # at-on => on-on
        # from-about => about-about
        s,t = target.split('-')
        default = '-'.join([t, t])
        default_targets.append(
                np.where(target_names == default)[0][0])
    return default_targets

def predict_for_fbeta(y_hat_proba, default_targets, threshold=0.5, threshold_type='margin'):
    n = y_hat_proba.shape[0]
    y_hat_for_fbeta = np.zeros(n, dtype=np.int)

    if threshold_type not in ['margin', 'value']:
        raise ValueError('threshold_type must be either "margin" or "value"')

    for i in np.arange(n):
        most, next_most = np.argsort(y_hat_proba[i, :])[[-2,-1]]
        if threshold_type == 'margin':
            if y_hat_proba[i, most] - y_hat_proba[i, next_most] > threshold:
                y_hat_for_fbeta[i] = most
            else:
                y_hat_for_fbeta[i] = default_targets[most]
        elif threshold_type == 'value':
            if y_hat_proba[i, most] > threshold:
                y_hat_for_fbeta[i] = most
            else:
                y_hat_for_fbeta[i] = default_targets[most]

    return y_hat_for_fbeta
