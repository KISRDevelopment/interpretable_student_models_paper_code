#
#   Sequence loading operations
#
import pandas as pd 
import numpy as np
from collections import defaultdict, namedtuple
import numpy.random as rng
import tensorflow as tf


# class SequenceFeaturizer:

#     def __init__(self, df):
#         self.df = df 
        
#         max_kc_id = np.max(df['skill'])
#         max_item_id = np.max(df['problem'])
        
#         self.n_kcs = max_kc_id + 1
#         self.n_items = max_item_id + 1

#         self._seqs = self._make_sequences()
#         self._calculate_deltat()
#         self._pad_sequences()
#         self._make_prev_curr_sequences()

#     def _make_sequences(self):
#         """
#         Generates list of sequences from data frame
#         """
#         seqs = defaultdict(list)

#         for i, row in enumerate(self.df.to_dict('records')):

#             key = row['student']
#             row['__index__'] = i
#             seqs[key].append(row)
        
#         return seqs 
    
#     def _pad_sequences(self):
#         """
#             Pads sequences to the required number of batch trials
#         """
#         # pad to n_batch_trials 
#         for _, seq in self._seqs.items():
#             to_pad = calc_padded_len(len(seq), self.n_batch_trials) - len(seq)
#             if to_pad > 0:
#                 seq.extend([None] * to_pad)
            
    
#     def _calculate_deltat(self):
#         """
#             Calculates delta time in seconds between consecutive trials in each sequences
#         """
#         for _, seq in self._seqs.items():
#             last_timestamp = seq[0]['timestamp']
#             for elm in seq:
#                 elm['__deltat__'] = elm['timestamp'] - last_timestamp
#                 last_timestamp = elm['timestamp']

#     def _make_prev_curr_sequences(self):
#         """
#             Changes sequences so that each has (prev, curr) which will make
#             the job of feature transformer much easier.
#         """
#         new_seqs = {}
#         for key, seq in self._seqs.items():
#             new_seq = [ (None, seq[0]) ]
#             for i in range(1, len(seq)):
#                 new_seq.append((seq[i-1], seq[i]))
#             new_seqs[key] = new_seq
        
#         self._seqs = new_seqs

#     def iterate(self, students, n_batch_seqs, n_batch_trials, max_seq_length, shuffle=True):

#         seqs = [self._seqs[s] for s in students]
#         seqs = split_long_seqs(seqs, max_seq_length)
            
#         if shuffle:
#             rng.shuffle(seqs)
        
#         n_seqs = len(seqs) 

#         for from_seq_id in range(0, n_seqs, n_batch_seqs):
#             to_seq_id = from_seq_id + n_batch_seqs

#             # grab the batch of sequences
#             batch_seqs = seqs[from_seq_id:to_seq_id]

#             # sort by length from shortest to longest to improve
#             # training efficiency
#             batch_seqs = sorted(batch_seqs, key=lambda s: len(s))

#             # identify maximum sequence length
#             max_seq_len = np.max([len(s) for s in batch_seqs])

#             # iterate over batches of trials now
#             for from_trial_id in range(0, max_seq_len, n_batch_trials):
#                 to_trial_id = from_trial_id + n_batch_trials

#                 # get eligible sequences that haven't finished
#                 subseqs = [s[from_trial_id:to_trial_id] for s in batch_seqs if to_trial_id <= len(s)]

#                 # transform them into final features
#                 prev_trial_features = self._transform(subseqs, True)
#                 curr_trial_features = self._transform(subseqs, False)

#                 new_seqs = from_trial_id == 0
                
#                 yield prev_trial_features, curr_trial_features, new_seqs

#     def _transform(self, subseqs, prev_trial=False):
#         n_batch = len(subseqs)
#         n_trials = len(subseqs[0])

#         correct = np.zeros((n_batch, n_trials), dtype=np.float32)
#         skill = np.zeros((n_batch, n_trials, self.n_kcs), dtype=np.float32)
#         item = np.zeros((n_batch, n_trials, self.n_items), dtype=np.float32)
#         deltat = np.zeros((n_batch, n_trials), dtype=np.float32)

#         included = np.zeros((n_batch, n_trials), dtype=np.float32)
#         trial_index = np.zeros((n_batch, n_trials), dtype=np.int)

#         tuple_idx = 0 if prev_trial else 1 

#         for s, seq in enumerate(subseqs):
#             for t, elm in enumerate(seq):
                
#                 trial = elm[tuple_idx]

#                 if trial is None:
#                     correct[s, t] = 0
#                     deltat[s, t] = 0
#                     included[s, t] = False 
#                     trial_index[s, t] = -1
#                 else:
#                     correct[s, t] = trial['correct']
#                     skill[s, t, trial['skill']] = 1
#                     item[s, t, trial['problem']] = 1
#                     deltat[s, t] = trial['__deltat__']
#                     included[s, t] = True 
#                     trial_index[s, t] = trial['__index__']
                
#         return TrialFeatures(
#             tf.convert_to_tensor(correct), 
#             tf.convert_to_tensor(skill), 
#             tf.convert_to_tensor(item), 
#             tf.convert_to_tensor(deltat), 
#             tf.convert_to_tensor(included), 
#             trial_index) 

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

def split_long_seqs(arr, maxlen):
    """
        Splits sequences that are greater than maxlen in length
    """
    new_seqs = []
    for seq in arr:
        if len(seq) <= maxlen:
            new_seqs.append(seq)
        else:
            rem_seq = seq 
            while len(rem_seq) > 0:
                subseq = rem_seq[:maxlen]
                new_seqs.append(subseq)
                rem_seq = rem_seq[maxlen:]
    return new_seqs

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
        max_batch_seq_len = np.max([len(s) for s in batch_seqs])

        # iterate over batches of trials now
        for from_trial_id in range(0, max_batch_seq_len, n_batch_trials):
            to_trial_id = from_trial_id + n_batch_trials

            # get eligible sequences that haven't finished
            subseqs = [s[from_trial_id:to_trial_id] for s in batch_seqs if to_trial_id <= len(s)]

            new_seqs = from_trial_id == 0
            
            yield subseqs, new_seqs

def iterate_unbatched(seqs, n_batch_seqs):
    """
        Iterates through sequences in batches but without batching across a sequence
        So a batch will constaint of full sequences
    """
    n_seqs = len(seqs) 

    for from_seq_id in range(0, n_seqs, n_batch_seqs):
        to_seq_id = from_seq_id + n_batch_seqs

        batch_seqs = seqs[from_seq_id:to_seq_id]
        batch_seqs = pad_to_maxlen(batch_seqs)
        batch_seqs = make_prev_curr_sequences(batch_seqs)
        
        yield batch_seqs, True

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
    
def pad_to_maxlen(seqs):
    """
        pads the sequences so that they all have the same length as longest sequence
    """
    maxlen = np.max([len(s) for s in seqs])
    new_seqs = []
    for i in range(len(seqs)):
        new_seqs.append(seqs[i] + ([None] * (maxlen - len(seqs[i]))))
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

    r = split_long_seqs([[1,2,3],[4,5,6,7,8]], maxlen=3)
    for subseq in r:
        print(subseq)
    
    r = pad_to_maxlen([[1,2,3],[4,5,6,7,8]])
    for subseq in r:
        print(subseq)

    r = pad_to_multiple([[1,2,3,4,5,6,7,8,9,10],[4,5,6,7,8]], 10)
    for subseq in r:
        print(subseq)
    print()

    for batch, newseqs in iterate_unbatched([[1,2,3,4,5,6,7,8,9,10],[4,5,6,7,8],[3,4], [5,6,7], [10]], 2):
        print(len(batch))
        print([len(s) for s in batch])
        print(newseqs)
        print()

    print("Iterate batched:")
    for batch, newseqs in iterate_batched([[1,2,3,4,5,6,7,8,9,10],[4,5,6,7,8],[3,4], [5,6,7], [10]], 2, 3):
        print(len(batch))
        print([len(s) for s in batch])
        print(newseqs)
        print(batch)
        print()
