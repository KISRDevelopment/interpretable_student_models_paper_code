import itertools
import json
import numpy as np
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_json(path, d):
    with open(path, 'w') as f:
        json.dump(d, f, indent=4)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def min_seq_len_filter(df, val):
    
    by_student = df.groupby('student').size()
    by_student = by_student[by_student >= val]
    new_df = df[df['student'].isin(set(by_student.index))]

    return new_df

def xe_loss(ytrue, ypred, mask):
    """
        ytrue: [n_batch, n_steps]
        ypred: [n_batch, n_steps]
        mask:  [n_batch, n_steps]
    """
    losses = -(ytrue * tf.math.log(ypred) + (1-ytrue) * tf.math.log(1-ypred))
    losses = losses * mask
    return tf.reduce_sum(losses) / tf.reduce_sum(mask)

def load_params(components, params):
    for component in components:
        for name, p in component.trainables:
            p.assign(params[name])
   
def save_params(components):
    params = {}
    for component in components:
        for name, p in component.trainables:
            params[name] = p.numpy()
    return params


def hyperparam_combs(d):
    
    keys = sorted(d.keys())
    sets = [d[k] for k in keys]
    
    return [ dict(zip(keys, s)) for s in itertools.product(*sets) ]
