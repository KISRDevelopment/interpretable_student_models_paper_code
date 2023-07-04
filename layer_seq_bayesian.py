import numpy as np 
import torch as th 
import torch.nn.functional as F 
import torch.nn as nn 
import torch.jit as jit
from torch import Tensor
from typing import List
import itertools

@jit.script
def seq_bayesian(logpred, ytrue):
    """
        Performs sequential Bayesian model averaging based on binary observations.
        Assumes there is B sequences of length T each and that
        there are A models predicting y_t given y_1..t-1.

        Input:
            logpred: BxAxTx2 representing p(y_t|y_1..t-1,alpha)
            ytrue: BxT
        Returns:
            logpreds: BxTx2
            final_posterior: BxA logP(alpha, y1...yt)
    """

    # logprobability of observations BxAxT
    logprob = ytrue[:,None,:] * logpred[:,:,:,1] + (1-ytrue[:,None,:]) * logpred[:,:,:,0]
        
    # calculate unnormed posterior over alphas
    # this represents p(alpha,y_1...yt-1)
    # p(alpha) ~ Uniform (i.e., for t=1)
    posteriors = logprob.cumsum(2)
    alpha_posterior = posteriors.roll(dims=2, shifts=1) # BxAxT
    alpha_posterior[:, :, 0] = 0.0 # uniform prior

    # compute unnormed predictions
    unnormed_preds = logpred + alpha_posterior[:,:,:,None] # BxAxTx2
    unnormed_preds = th.logsumexp(unnormed_preds, dim=1) # BxTx2

    # normalize BxTx2
    normed_preds = unnormed_preds - th.logsumexp(unnormed_preds, dim=2)[:,:,None]

    return normed_preds, posteriors # BxAxT

def main():

    ytrue = th.tensor([
        [0, 1, 1, 0, 1, 0]
    ]).float()

    pred_1 = th.tensor([
        [
            [0.2, 0.9, 0.9, 0.1, 0.9, 0.1],# strong model
            [0.2, 0.5, 0.6, 0.5, 0.2, 0.9],# rubbish model
            [0.3, 0.75, 0.8, 0.8, 0.5, 0.7], # good model
        ]
        
    ]) # BxAxT
    
    pred_0 = 1-pred_1

    logpred = th.concat((pred_0[:,:,:,None], pred_1[:,:,:,None]), dim=3).log() # BxAxTx2

    final_logpred, alpha_posterior = seq_bayesian(logpred, ytrue)

    print(final_logpred.exp())

    alpha_posterior = alpha_posterior[:,:,-1]
    alpha_logposterior = alpha_posterior - th.logsumexp(alpha_posterior, dim=1)[:,None] # BxA

    print(alpha_logposterior.exp())

if __name__ == "__main__":
    main()
    