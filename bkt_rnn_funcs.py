import numpy as np
import metrics
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from typing import List
import pandas as pd 
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import copy 
import json
import time 
import model_brute_force_bkt

def bkt_rnn(state, 
            prev_kc, 
            prev_E, 
            prev_R, 
            prev_y, 
            curr_kc, 
            curr_E,
            first_trial):
    """ 
        Implements standard BKT RNN dynamics

        state: predictive joint distrib log P(s_{t-1}=i,y_{1:(t-2)})
            shape: BxKxM
            where K is number of KCS and M is number of states
        prev_kc: KC encountered at previous time step
            shape: B
        prev_E: Previous log output matrix
            shape: BxMxO
            where O is number of outputs
        prev_R: Previous log transition matrix
            shape: BxMxM
            rows represent targets and columns represent source
        prev_y: Previous answer
            shape: B
        curr_kc: current KC
            shape: B
        curr_E: Current log output matrix
            shape: BxMxO
    """
    
    #
    # update state
    # 
    batch_idx = th.arange(state.shape[0])
    log_prev_obs = prev_E[batch_idx, :, prev_y] # BxM
    state[batch_idx, prev_kc, :] = th.logsumexp(log_prev_obs[:,None,:] + # Bx1xM
                     state[batch_idx,prev_kc,None,:] + # Bx1xM
                     prev_R, # BxMxM
                     dim=2) * (1 - first_trial) + first_trial * state[batch_idx, prev_kc, :]
        
    #
    # predict
    #
    log_py = th.logsumexp(log_obs + # BxMxO
                          state[batch_idx, curr_kc, :, None], # BxMx1, 
                          dim=1)  # BxO
    log_py = log_py - th.logsumexp(log_py, dim=1)
    
    return state, log_py

if __name__ == "__main__":

    n_chains = 1
    n_states = 2
    n_outputs = 2

    trans_logits = nn.Parameter(th.randn(n_chains, n_states, n_states))
    obs_logits = nn.Parameter(th.randn(n_chains, n_states, n_outputs))
    init_logits = nn.Parameter(th.randn(n_chains, n_states))

    log_alpha = F.log_softmax(init_logits, dim=1) # n_chains x n_states
    log_obs = F.log_softmax(obs_logits, dim=2) # n_chains x n_states x n_obs
    log_t = F.log_softmax(trans_logits, dim=1) # n_chains x n_states x n_states
    
    y = th.tensor([[0, 1, 1, 0, 1, 0, 0, 0, 1]]).long()
    n_batch = y.shape[0]
    state = th.tile(log_alpha, (n_batch, 1, 1))
    kc = th.zeros(n_batch).long()

    outputs = th.zeros((y.shape[0], y.shape[1], n_outputs))

    for i in range(y.shape[1]):
        prev_y = y[:, i-1] if i > 0 else y[:, 0]

        state, log_py = bkt_rnn(state=state, 
                                prev_kc=kc, 
                                prev_E=log_obs[kc, :, :], 
                                prev_R=log_t[kc, :, :], 
                                prev_y=prev_y, 
                                curr_kc=kc, 
                                curr_E=log_obs[kc, :, :],
                                first_trial=i == 0)
        outputs[:, i, :] = log_py

    print(th.exp(outputs))

    with th.no_grad():
        R = th.exp(log_t).numpy()
        E = th.exp(log_obs).numpy()
        I = th.exp(F.log_softmax(init_logits, dim=1)).numpy()   
        y1 = y.numpy() == 1
        y0 = y.numpy() == 0
        seq = np.vstack((y0, y1)).T 

        probs = model_brute_force_bkt.forward_bkt(seq, R[0, 1, 0], R[0, 0, 1], E[0, 0, 1], E[0, 1, 0], I[0, 1])
        print(probs)

    