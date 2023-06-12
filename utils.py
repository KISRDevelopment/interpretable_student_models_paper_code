import itertools
import json
import numpy as np
from collections import defaultdict

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

def pad_to_max(seqs, padding_value):
    new_seqs = []

    max_len = np.max([len(s) for s in seqs])
    
    new_seqs = np.ones((len(seqs), max_len)) * padding_value

    for i, seq in enumerate(seqs):
        new_seqs[i, :len(seq)] = seq
    
    return new_seqs

def to_seqs(df):
    """
        Returns a dictionary of dictionaries.
            student => {
                kc => [],
                problem => [],
                correct => []
            }
    """
    seqs = defaultdict(lambda: {
        "kc" : [],
        "problem" : [],
        "correct" : []
    })
    for r in df.itertuples():
        seqs[r.student]["kc"].append(r.skill)
        seqs[r.student]["problem"].append(r.problem)
        seqs[r.student]["correct"].append(r.correct)
    
    for student, details in seqs.items():
        details['kc'] = np.array(details['kc'])
        details['problem'] = np.array(details['problem'])
        details['correct'] = np.array(details['correct'])
        
    return seqs

def prepare_batch(seqs):
    
    # need to compute max length because the indexing will be flat
    lens = [s['kc'].shape[0] for s in seqs]
    max_len = max(lens)
    
    # split sequences by KC
    subseqs = []
    offset = 0
    for seq in seqs:
        unique_kcs = np.unique(seq['kc'])
        for kc in unique_kcs:
            ix = seq['kc'] == kc 
            subseqs.append({
                "kc" : kc,
                "problem" : seq['problem'][ix],
                "correct" : seq['correct'][ix],
                "trial_id" : offset + np.where(ix)[0]
            })
        offset += max_len
    
    return subseqs, max_len

if __name__ == "__main__":
    #seqs = [[0, 1, 1, 0], [0, 1]]

    #print(pad_to_multiple(seqs, 3, -1))

    import pandas as pd 

    #seqs = to_seqs(df)
    
    seqs = [
        {
            "kc" :      np.array([0, 1, 0, 1]),
            "problem" : np.array([0, 0, 0, 0]),
            "correct":  np.array([0, 0, 0, 0]),
        },
        {
            "kc" :      np.array([0, 0, 0, 1, 1, 1, 1]),
            "problem" : np.array([0, 0, 0, 0, 0, 0, 0]),
            "correct":  np.array([0, 0, 0, 0, 0, 0, 0]),
        }
    ]

    subseqs, max_len = prepare_batch(seqs)

    print(subseqs)

    padded_trial_id = pad_to_multiple([s['trial_id'] for s in subseqs], multiple=8, padding_value=-1)

    print(padded_trial_id)

    flattened_trial_id = padded_trial_id.flatten()

    print(flattened_trial_id)

    # store result
    valid_trial_id = flattened_trial_id[flattened_trial_id > -1].astype(int)
    print(valid_trial_id.dtype)
    result = np.zeros(len(seqs)*max_len)
    result[valid_trial_id] = 1

    print(result)