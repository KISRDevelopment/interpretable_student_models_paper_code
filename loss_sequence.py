
import numpy as np 
import itertools 
import torch as th 
import torch.nn.functional as F 
from torch import Tensor
from typing import List, Tuple
import torch.nn as nn 
from numba import jit
import joint_pmf 

def main():
    n_skills = 4
    n_batch = 50 
    timesteps = 100
    n_problems = 20

    problem_seq = np.tile(np.random.permutation(n_problems)[:timesteps][None,:], (n_batch, 1))
    problem_seq = th.tensor(problem_seq).long()
    

    ml = th.randn((n_problems, n_skills))

    #loss_layer = NbackLoss(n_skills, [1])
    #output = loss_layer(problem_seq, ml, th.ones_like(problem_seq).float())
    
    
    membership_logprobs = pmf(ml) # Bx2**n_skills
            
    print(nback_loss(problem_seq, th.ones_like(problem_seq).float(), membership_logprobs, [1, 2, 3, 10]))
    #print(nback_loss(problem_seq, th.ones_like(problem_seq).float(), membership_logprobs, 3))

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


    prev_problem = problem_seq[:, :-lag] # BxT-L
    curr_problem = problem_seq[:, lag:]  # BxT-L
    mask_ix = mask_ix[:, lag:] # BxT-L

    prev_logprobs = membership_logprobs[prev_problem] # BxT-LxK
    curr_logprobs = membership_logprobs[curr_problem] # BxT-LxK

    u = prev_logprobs + curr_logprobs # BxT-LxK

    logprob_same = th.logsumexp(u, dim=2) # BxT-L 

    # masked elements are not counted in the average logprob, B
    seq_lens = mask_ix.sum(1) # B
    mean_logprob_same = (logprob_same * mask_ix).sum(1) / seq_lens

    # set sequences whose length is less than the lag to maximum loss
    less_than_lag_ix = seq_lens <= lag 
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