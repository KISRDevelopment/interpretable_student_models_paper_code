
import numpy as np 
import itertools 
import torch as th 
import torch.nn.functional as F 
from torch import Tensor
from typing import List, Tuple
import torch.nn as nn 
from numba import jit

def main():
    
    # membership logits
    membership_logits = th.tensor([
        [3, -3, -3],        # 0
        [3, -3, -4],        # 1
        [-3, 4, -2],        # 2
        [-3, 5, -2],        # 3
        [-10, 0, 10],       # 4
        [10, 0, 0]          # 5
    ]).float()

    # sequence 
    problem_seq = th.tensor([
        [0, 1, 2, 0, 1, 2, 3]
    ])

    # only first 4 trials
    mask_ix = th.tensor([[1, 1, 1, 1, 1, 0, 0]]).float()

    membership_logprobs = F.log_softmax(membership_logits, 1)
    
    r = mean_logprob_same_at_lag(problem_seq, mask_ix, membership_logprobs, 2)
    print(r)

    r = nback_loss(problem_seq, mask_ix, membership_logprobs, [1, 2, 3, 4, 5])
    print(r)

@th.jit.script
def mean_logprob_same_at_lag(problem_seq: Tensor, mask_ix: Tensor, membership_logprobs: Tensor, lag: int):
    """
        Input:
            problem_seq: BxT
            mask_ix: BxT
            membership_logprobs: PxK
            lag: int
        Output:
            mean_logprob_same: B
                Average log P(a_i == a_i-lag)
    """

    # if lag is bigger than input size, return max loss
    if lag >= problem_seq.shape[1]:
        return th.ones(problem_seq.shape[0]) * -100.0
    
    #
    # probability that n-lag problems are the same
    #
    prev_problem = problem_seq[:, :-lag] # BxT-L
    curr_problem = problem_seq[:, lag:]  # BxT-L
    prev_logprobs = membership_logprobs[prev_problem] # BxT-LxK
    curr_logprobs = membership_logprobs[curr_problem] # BxT-LxK
    u = prev_logprobs + curr_logprobs # BxT-LxK
    logprob_same = th.logsumexp(u, dim=2) # BxT-L 
    
    # compute average logprob on valid trials, B
    mean_logprob_same = (logprob_same * mask_ix[:, lag:]).sum(1) / (mask_ix[:, lag:].sum(1) + 1e-6)
    
    # sequence length has to be twice the lag at minimum
    # so that we get one cycle at least
    seq_lens = mask_ix.sum(1) # B
    less_than_lag_ix = seq_lens < (2*lag)
    mean_logprob_same[less_than_lag_ix] = -100
    
    return mean_logprob_same

@th.jit.script
def nback_loss(problem_seq: Tensor, mask_ix: Tensor, membership_logprobs: Tensor, lags: List[int]):
    
    outputs = th.jit.annotate(List[Tensor], [])
    for lag in lags:
        # B
        mean_logprob_same = mean_logprob_same_at_lag(problem_seq, mask_ix, membership_logprobs, lag)
        outputs.append(mean_logprob_same)
    
    outputs = th.stack(outputs) # lags x B
    outputs = outputs.T # Bxlags 

    mfl, _ = th.min(-outputs, 1)

    return mfl.mean()
if __name__ == "__main__":
    main()