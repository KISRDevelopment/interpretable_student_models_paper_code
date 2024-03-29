import numpy as np 

import torch as th 
import torch.nn.functional as F 
import torch.nn as nn 
import torch.jit as jit
from torch import Tensor
from typing import List
import itertools
import utils 

class FastBkt(jit.ScriptModule):

    def __init__(self, n, device):
        super(FastBkt, self).__init__()
        
        self.n = n 
        
        # all possible trajectories of length n
        trajectories = make_trajectories(n)

        # transition indecies to compute the probaiblity of a trajectory
        # 0: -pL, 1: pL, 2: pF, 3: 1-pF, 4: 1-pI0, 5: pI0 
        trans_ind = make_transition_indices(trajectories)

        # transition indecies to compute the posterior predictive over
        # the hidden state
        pred_ind = make_predictive_indices(trajectories)

        # move to torch
        self._trajectories = th.tensor(trajectories).float().to(device)
        self._trans_ind = th.tensor(trans_ind).long().to(device)
        self._pred_ind = th.tensor(pred_ind).long().to(device)
        self._device = device 

        self.temporal_batch_size = n * 8

    def pad(self, seqs, padding_value):
        return pad_to_multiple(seqs, multiple=self.n, padding_value=padding_value)
    
    @jit.script_method
    def forward(self, corr: Tensor, dynamics_logits: Tensor, obs_logits: Tensor) -> Tensor:
        
        # initial and state transition dynamics (B)
        logit_pL, logit_pF, logit_pI0 = dynamics_logits[:, 0], dynamics_logits[:, 1], dynamics_logits[:, 2]

        # transition probabilities that will be indexed into
        trans_logits = th.vstack((-logit_pL, logit_pL, logit_pF, -logit_pF, -logit_pI0, logit_pI0)).T # Bx5
        trans_logprobs = F.logsigmoid(trans_logits) # Bx5

        #
        # probability of each trajectory
        #

        # Bx2**Nx1 * Bx1x1
        initial_logprobs = self._trajectories[None,:, [0]] * trans_logprobs[:,None,[5]] + \
            (1-self._trajectories[None, :, [0]]) * trans_logprobs[:,None,[4]] # Bx2**Nx1
        
        # Bx2**Nx1
        logprob_h = trans_logprobs[:,self._trans_ind].sum(2)[:,:,None] + initial_logprobs
        
        return self.forward_(corr, dynamics_logits, obs_logits, logprob_h)

    @jit.script_method
    def forward_(self, corr: Tensor, dynamics_logits: Tensor, obs_logits: Tensor, logprob_h: Tensor) -> Tensor:
        """
            Input: 
                corr: trial sequence BxT
                dynamics_logits: BKT parameter logits (pL, pF, pI0) Bx3
                obs_logits: per-timestep logits (pG, pS) BxTx2
            Output:
                output_logprobs BxTx2

            Let 2**N = V
        """

        # initial and state transition dynamics (B)
        logit_pL, logit_pF, logit_pI0 = dynamics_logits[:, 0], dynamics_logits[:, 1], dynamics_logits[:, 2]

        # observation probabilities (BxT)
        logit_pG, logit_pS = obs_logits[:, :, 0], obs_logits[:, :, 1]

        # transition probabilities that will be indexed into
        trans_logits = th.vstack((-logit_pL, logit_pL, logit_pF, -logit_pF, -logit_pI0, logit_pI0)).T # Bx5
        trans_logprobs = F.logsigmoid(trans_logits) # Bx5

        # iterate over time 
        pred_corrects = th.jit.annotate(List[Tensor], [])
        pred_incorrects = th.jit.annotate(List[Tensor], [])
        for i in range(0, corr.shape[1], self.n):
            if (i > 0) and (i % self.temporal_batch_size == 0):
                logprob_h = logprob_h.detach()
            
            from_index = i
            to_index = from_index + self.n 

            # grab slice BxN
            corr_slice = corr[:, from_index:to_index]

            # grab slice of guess and slip probabilities BxN
            logit_pG_slice = logit_pG[:, from_index:to_index]
            logit_pS_slice = logit_pS[:, from_index:to_index]

            # probability of answering correctly Bx2**NxN
            # 1x2**NxN * Bx1xN
            obs_logits_correct = self._trajectories[None,:,:] * (-logit_pS_slice[:,None,:]) + (1-self._trajectories[None,:,:]) * logit_pG_slice[:,None,:]
            obs_logprob_correct = F.logsigmoid(obs_logits_correct) # Bx2**NxN

            # probability of answering incorrectly Bx2**NxN
            # print(logit_pS.shape)
            # print(self._trajectories.shape)
            obs_logits_incorrect = self._trajectories[None,:,:] * (logit_pS_slice[:,None,:]) + (1-self._trajectories[None,:,:]) * (-logit_pG_slice[:,None,:])
            obs_logprob_incorrect = F.logsigmoid(obs_logits_incorrect) # Bx2**NxN


            # likelikelihood of observing each observation under each trajectory Bx2**NxN
            # corr_slice[:,None,:]          Bx1xN
            # obs_logprob_correct           Bx2**NxN
            obs_loglik_given_h = corr_slice[:,None,:] * obs_logprob_correct + \
                (1-corr_slice[:,None,:]) * obs_logprob_incorrect

            # running likelihood of prior observations Bx2**NxN
            past_loglik_given_h = obs_loglik_given_h.cumsum(2).roll(dims=2, shifts=1)
            past_loglik_given_h[:,:,0] = 0.0

            # probability of correct/incorrect for each trajectory (weighted by trajectory weight)
            # Bx2**NxN + Bx2**NxN + Bx2**Nx1 = Bx2**NxN
            pred_correct_given_h = past_loglik_given_h + obs_logprob_correct + logprob_h
            pred_incorrect_given_h = past_loglik_given_h + obs_logprob_incorrect + logprob_h

            # unnormed probabilities of correctness and incorrectness BxN
            pred_correct = th.logsumexp(pred_correct_given_h, dim=1)
            pred_incorrect = th.logsumexp(pred_incorrect_given_h, dim=1)
            pred_corrects.append(pred_correct)
            pred_incorrects.append(pred_incorrect)

            #
            # new state prior for next iteration
            #
            seq_loglik_given_h = obs_loglik_given_h.sum(2) # Bx2**N
            seq_loglik = seq_loglik_given_h + logprob_h[:,:,0] # Bx2**N
            next_h_one_logprob = F.logsigmoid(trans_logits[:, self._pred_ind]) + seq_loglik # Bx2**N
            next_h_zero_logprob = F.logsigmoid(-trans_logits[:, self._pred_ind]) + seq_loglik  # Bx2**N
            next_h_one_logprob = th.logsumexp(next_h_one_logprob, dim=1)[:,None] # Bx1
            next_h_zero_logprob = th.logsumexp(next_h_zero_logprob, dim=1)[:,None] # Bx1
            
            # 1x2**N * Bx1 = Bx2**N
            initial_logprobs = self._trajectories[None,:,0] * next_h_one_logprob + (1-self._trajectories[None,:,0]) * next_h_zero_logprob
            
            # Bx2**Nx1 + Bx2**Nx1 = Bx2**Nx1
            logprob_h = trans_logprobs[:,self._trans_ind].sum(2)[:,:,None] + initial_logprobs[:,:,None]

                
        # BxT
        pred_corrects = th.concat(pred_corrects, dim=1)
        pred_incorrects = th.concat(pred_incorrects, dim=1)

        # BxTx2
        preds = th.concat((pred_incorrects[:,:,None], pred_corrects[:,:,None]), dim=2)
        
        # BxTx2 - BxTx1
        logprob_next = preds - th.logsumexp(preds, dim=2)[:,:,None]

        # BxTx2
        return logprob_next

def make_trajectories(n):
    """
        constructs all possible trajectories of binary state of
        a sequence of length n.
        Returns a matrix of size 2^n x n
    """
    trajectories = np.array(list(itertools.product(*[[0, 1]]*n)))
    
    return trajectories

def make_transition_indices(trajectories):
    """
        computes the transition indices
        Returns a matrix of size 2^n x n-1
        because it excludes the first trial.
    """

    convolved = np.zeros_like(trajectories)
    for i in range(trajectories.shape[0]):
        indices = np.convolve(trajectories[i,:], [1, 2], mode='same')
        convolved[i, :] = indices
    convolved = convolved[:,1:]

    return convolved

def make_predictive_indices(trajectories):
    """
        computes the indices to predict the transition from
        the last state.
    """
    target_state = np.ones(trajectories.shape[0])
    indices = trajectories[:,-1] * 2 + target_state
    return indices


def pad_to_multiple(seqs, multiple, padding_value):
    return th.tensor(utils.pad_to_multiple(seqs, multiple, padding_value))


def main():

    """ quick test of implementation against reference """
    
    # BKT parameters probabilities
    logit_pI0 = 0.3
    logit_pG = -0.4
    logit_pS = 1.0
    logit_pL = -1
    logit_pF = -2 

    # sequence
    obs = [0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
    seq = np.zeros((len(obs), 2))
    seq[np.arange(seq.shape[0]), obs] = 1
    
    sigmoid = lambda x: 1/(1+np.exp(-x))

    #
    # reference probabilities
    #
    import model_brute_force_bkt 
    probs = sigmoid(np.array([logit_pL, logit_pF, logit_pG, logit_pS, logit_pI0]))
    ref_bkt_prob_corr = model_brute_force_bkt.forward_bkt(seq, *probs)
    print(ref_bkt_prob_corr)

    #
    # BKT probabilities
    #
    dynamics_logits = th.tensor([[logit_pL, logit_pF, logit_pI0]])
    obs_logits = th.tensor([[logit_pG, logit_pS]])
    corr = th.tensor([obs])
    obs_logits = th.tile(obs_logits, (1, corr.shape[1], 1)) # BxTx2
    
    model = FastBkt(5, 'cpu:0')
    logpred = model(corr, dynamics_logits, obs_logits)
    print(logpred.exp()[0, :, 1].numpy())

    #
    # compare to RNN BKTwhen obs logits are individualized 
    #
    obs_logits = th.rand_like(obs_logits)

    print("Individualized OBS Probs:")
    print("Fast BKT:")
    logpred = model(corr, dynamics_logits, obs_logits)
    print(logpred.exp()[0, :, 1].numpy())
    print("RNN BKT:")

    import layer_bkt 
    model = layer_bkt.RnnBkt()
    logpred = model(corr, dynamics_logits, obs_logits)
    print(logpred.exp()[0, :, 1].numpy())
if __name__ == "__main__":
    main()
