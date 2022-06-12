#
#   Sequence operations
#
import numpy as np
from collections import defaultdict

def make_sequences(df, students):
    """
    Generates list of sequences from data frame
    """
    seqs = defaultdict(list)
    for i, row in enumerate(df.to_dict('records')):
        key = row['student']
        if key in students:
            row['__index__'] = i
            seqs[key].append(row)
    
    return [v for k,v in seqs.items()] 


def calc_padded_len(n, m):
    """
        Computes how much padding to add to a list of length n so that the
        length is a multiple of m
    """
    return int( np.ceil(n / m) * m )

def iterate_batched(seqs, n_batch_seqs, n_batch_trials):
    """
        Iterates in batches of size (n_batch_seqs, n_batch_trials)
    """
    n_seqs = len(seqs) 
    for from_seq_id in range(0, n_seqs, n_batch_seqs):
        to_seq_id = from_seq_id + n_batch_seqs

        # grab the batch of sequences
        batch_seqs = seqs[from_seq_id:to_seq_id]
        batch_seqs = pad_to_multiple(batch_seqs, n_batch_trials)
        batch_seqs = make_prev_curr_sequences(batch_seqs)

        # sort by length from shortest to longest to improve
        # training efficiency
        batch_seqs = sorted(batch_seqs, key=lambda s: len(s))

        # identify maximum sequence length
        max_batch_seq_len = len(batch_seqs[-1])
        
        # iterate over batches of trials now
        for from_trial_id in range(0, max_batch_seq_len, n_batch_trials):
            to_trial_id = from_trial_id + n_batch_trials

            # get eligible sequences that haven't finished
            subseqs = [s[from_trial_id:to_trial_id] for s in batch_seqs if to_trial_id <= len(s)]

            new_seqs = from_trial_id == 0
            
            yield subseqs, new_seqs

def calculate_deltat(seqs):
    """
        Calculates delta time in seconds between consecutive trials in each sequences
    """
    for seq in seqs:
        last_timestamp = seq[0]['timestamp']
        for elm in seq:
            elm['__deltat__'] = elm['timestamp'] - last_timestamp
            last_timestamp = elm['timestamp']
    
def pad_to_multiple(seqs, multiple):
    """
        padds the sequences length to the nearest multiple of the given number
    """
    new_seqs = []

    for seq in seqs:
        to_pad = calc_padded_len(len(seq), multiple) - len(seq)
        new_seq = seq[:]
        if to_pad > 0:
            new_seq.extend([None] * to_pad)
        new_seqs.append(new_seq)

    return new_seqs

def pad_to_max(seqs):
    new_seqs = []
    max_len = max([len(s) for s in seqs])
    for seq in seqs:
        to_pad = max_len - len(seq)
        new_seq = seq[:]
        if to_pad > 0:
            new_seq.extend([None] * to_pad)
        return new_seqs.append(new_seq)
    return new_seqs
    
def make_prev_curr_sequences(seqs):
    """
        Changes sequences so that each has (prev, curr) which will make
        the job of feature transformer much easier.
    """
    new_seqs = []
    for seq in seqs:
        new_seq = [ (None, seq[0]) ]
        for i in range(1, len(seq)):
            new_seq.append((seq[i-1], seq[i]))
        new_seqs.append(new_seq)
        
    return new_seqs

if __name__ == "__main__":

    # df = pd.read_csv("data/datasets/gervetetal_assistments17_first_attempt.csv")

    # featurizer = SequenceFeaturizer(df, 50)

    # for prev_features, curr_features, new_seqs in featurizer.iterate([1, 3], 10):
    #     print(curr_features)

    # r = split_long_seqs([[1,2,3],[4,5,6,7,8]], maxlen=3)
    # for subseq in r:
    #     print(subseq)
    
    # r = pad_to_maxlen([[1,2,3],[4,5,6,7,8]])
    # for subseq in r:
    #     print(subseq)

    # r = pad_to_multiple([[1,2,3,4,5,6,7,8,9,10],[4,5,6,7,8]], 10)
    # for subseq in r:
    #     print(subseq)
    # print()

    # for batch, newseqs in iterate_unbatched([[1,2,3,4,5,6,7,8,9,10],[4,5,6,7,8],[3,4], [5,6,7], [10]], 2):
    #     print(len(batch))
    #     print([len(s) for s in batch])
    #     print(newseqs)
    #     print()

    print("Iterate batched:")
    for batch, newseqs in iterate_batched([[1,2,3,4,5,6,7,8,9,10],[4,5,6,7,8],[3,4], [5,6,7], [10]], n_batch_seqs=2, n_batch_trials=3):
        print("Batch length: %d" % len(batch))
        print("Sequence lengths: ", [len(s) for s in batch])
        print(newseqs)
        print(batch)
        print()
