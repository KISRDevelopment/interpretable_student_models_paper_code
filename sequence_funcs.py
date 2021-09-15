#
#   Sequence loading operations
#
import pandas as pd 
import numpy as np
from collections import defaultdict, namedtuple
import numpy.random as rng


def make_sequences(df, seq_identifier):
    """
        Generates list of sequences from data frame
    """
    seqs = defaultdict(list)

    for row in df.to_dict('records'):

        key = row[seq_identifier]
        seqs[key].append(row)
    
    return [s for _, s in seqs.items()] 

def pad_seqs(seqs, n_batch_trials):
    """
        Pads sequences to the required number of batch trials
    """
    # pad to n_batch_trials 
    for seq in seqs:
        to_pad = calc_padded_len(len(seq), n_batch_trials) - len(seq)
        if to_pad > 0:
            seq.extend([None] * to_pad)
        
    return seqs

def calc_padded_len(n, m):
    """
        Computes how much padding to add to a list of length n so that the
        length is a multiple of m
    """
    return int( np.ceil(n / m) * m )

def featurize_seqs(seqs, cols_to_monitor):
    """
        Generates featurized representation of sequences which
        has current and previous step features
    """
    featurized_seqs = []
    for seq in seqs:

        first_element = { ('prev_%s' % c): p for c,p in cols_to_monitor.items() }
        for c, _ in cols_to_monitor.items():
            first_element['curr_%s' % c] = seq[0][c]
        first_element['__padding__'] = False
        featurized_seq = [first_element]

        for e in seq[1:]:
            prev_element = featurized_seq[-1]
            element = { ('prev_%s' % c): prev_element['curr_%s' % c] for c,_ in cols_to_monitor.items() }
            if e is not None:
                for c, _ in cols_to_monitor.items():
                    element['curr_%s' % c] = e[c]
                element['__padding__'] = False
            else:
                for c, p in cols_to_monitor.items():
                    element['curr_%s' % c] = p
                element['__padding__'] = True 

            featurized_seq.append(element)
        
        featurized_seqs.append(featurized_seq)
    
    return featurized_seqs

def create_loader(seqs, n_batch_seqs, n_batch_trials, transformer, shuffle=True):
    
    if shuffle:
        rng.shuffle(seqs)

    n_seqs = len(seqs)

    for from_seq_id in range(0, n_seqs, n_batch_seqs):
        to_seq_id = from_seq_id + n_batch_seqs

        # grab the batch of sequences
        batch_seqs = seqs[from_seq_id:to_seq_id]

        # sort by length from shortest to longest to improve
        # training efficiency
        batch_seqs = sorted(batch_seqs, key=lambda s: len(s))

        # identify maximum sequence length
        max_seq_len = np.max([len(s) for s in batch_seqs])

        # iterate over batches of trials now
        for from_trial_id in range(0, max_seq_len, n_batch_trials):
            to_trial_id = from_trial_id + n_batch_trials

            # get eligible sequences
            subseqs = [s[from_trial_id:to_trial_id] for s in batch_seqs if to_trial_id <= len(s)]

            # transform them into final features
            features = transformer(subseqs)

            new_seqs = from_trial_id == 0
            
            yield features, new_seqs

def dummy_transformer(subseqs):

    n_batch = len(subseqs)
    n_trials = len(subseqs[0])

    features = np.zeros((n_batch, n_trials, 3))
    for s, seq in enumerate(subseqs):
        for t, elm in enumerate(seq):
            features[s, t, 0] = elm['prev_correct']
            features[s, t, 1] = elm['curr_correct']
            features[s, t, 2] = not elm['__padding__']
    
    return features

def create_kt_transformer(max_kc_idx):

    KTFeatures = namedtuple('KTFeatures', 'prev_corr prev_skill curr_corr curr_skill curr_mask')

    n_kcs = max_kc_idx + 1

    def transformer(subseqs):
        n_batch = len(subseqs)
        n_trials = len(subseqs[0])

        prev_corr = np.zeros((n_batch, n_trials), dtype=np.float32)
        prev_skill = np.zeros((n_batch, n_trials, n_kcs), dtype=np.float32)
        curr_corr = np.zeros((n_batch, n_trials), dtype=np.float32)
        curr_skill = np.zeros((n_batch, n_trials, n_kcs), dtype=np.float32)
        curr_mask = np.zeros((n_batch, n_trials), dtype=np.float32)

        for s, seq in enumerate(subseqs):
            for t, elm in enumerate(seq):
                prev_corr[s, t] = elm['prev_correct']
                curr_corr[s, t] = elm['curr_correct']

                curr_mask[s, t] = not elm['__padding__']

                prev_skill[s, t, elm['prev_skill']] = 1
                curr_skill[s, t, elm['curr_skill']] = 1
                
        return KTFeatures(prev_corr, prev_skill, curr_corr, curr_skill, curr_mask)
    
    return transformer

if __name__ == "__main__":

    rows = [
        { "student" : 0, "correct" : 1, "skill" : 0 },
        { "student" : 0, "correct" : 0, "skill" : 1 },
        { "student" : 1, "correct" : 0, "skill" : 1 },
        { "student" : 0, "correct" : 1, "skill" : 2 },
        { "student" : 1, "correct" : 0, "skill" : 0 },
        { "student" : 1, "correct" : 1, "skill" : 2 },
        { "student" : 1, "correct" : 1, "skill" : 2 },
        { "student" : 1, "correct" : 1, "skill" : 1 },
    ]
    df = pd.DataFrame(rows)

    seqs = make_sequences(df, 'student')
    for s in seqs:
        for e in s:
            print(e)
        print()

    seqs = pad_seqs(seqs, 4)

    for s in seqs:
        for e in s:
            print(e)
        print()
    
    featurized_seqs = featurize_seqs(seqs, {
        "correct" : 0,
        "skill" : 3
    })

    for s in featurized_seqs:
        for e in s:
            print(e)
        print()

    for features, new_seqs in create_loader(featurized_seqs, 2, 4, create_kt_transformer(3)):
        print(features)
        print(new_seqs)