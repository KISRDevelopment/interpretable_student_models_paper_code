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

def calc_padded_len(n, m):
    """
        Computes how much padding to add to a list of length n so that the
        length is a multiple of m
    """
    return int( np.ceil(n / m) * m )

def pad_to_multiple(seqs, multiple, padding_value):
    """
        padds the sequences length to the nearest multiple of the given number
    """
    new_seqs = []

    max_len = np.max([len(s) for s in seqs])
    padded_len = calc_padded_len(max_len, multiple)

    new_seqs = np.ones((len(seqs), padded_len)) * padding_value

    for i, seq in enumerate(seqs):
        new_seqs[i, :len(seq)] = seq
    
    return new_seqs


if __name__ == "__main__":
    seqs = [[0, 1, 1, 0], [0, 1]]

    print(pad_to_multiple(seqs, 3, -1))

