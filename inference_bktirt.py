import numpy as np
import metrics
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from typing import List, Tuple

from torch.nn.utils.rnn import pad_sequence

class RnnBkt(jit.ScriptModule):
    
    def __init__(self):
        super().__init__()

    def pad(self, seqs, padding_value):
        return pad_sequence([th.tensor(s) for s in seqs], batch_first=True, padding_value=padding_value)
     
    @jit.script_method
    def forward(self, corr: Tensor, dynamics_logits: Tensor, obs_logits: Tensor) -> Tuple[Tensor, Tensor]:
        """
            Input: 
                corr: trial sequence BxT
                dynamics_logits: BKT parameter logits (pL, pF, pI0) Bx3
                obs_logits: per-timestep logits (pG, pS) BxTx2
            Output:
                output_logprobs BxTx2
                state_posteriors BxTxS P(h_t,y1...yt)
        """
        
        trans_logits, obs_logits, init_logits = get_logits(dynamics_logits, obs_logits)
        
        return self.forward_(corr, trans_logits, obs_logits, init_logits)
    
    @jit.script_method
    def smoother(self, corr: Tensor, dynamics_logits: Tensor, obs_logits: Tensor) -> Tensor:
        """
            Input: 
                corr: trial sequence BxT
                dynamics_logits: BKT parameter logits (pL, pF, pI0) Bx3
                obs_logits: per-timestep logits (pG, pS) BxTx2
            Output:
                smoothed_logprobs BxTxS P(h_t|y_1:T)
        """
        
        trans_logits, obs_logits, init_logits = get_logits(dynamics_logits, obs_logits)
        
        _, state_posteriors = self.forward_(corr, trans_logits, obs_logits, init_logits) # P(h_t,y_1:t) BxTxS
        
        backward_logprobs = self.backward_(corr, trans_logits, obs_logits, init_logits) # P(y_{t+1}:T|h_t) BxTxS
        
        smoothed_logprobs = state_posteriors + backward_logprobs
        smoothed_logprobs = smoothed_logprobs - th.logsumexp(smoothed_logprobs, dim=2)[:,:,None] # BxTxS P(h_t|y_1:T)

        return smoothed_logprobs
    
    @jit.script_method
    def forward_(self, 
        obs: Tensor,
        trans_logits: Tensor, 
        obs_logits: Tensor, 
        init_logits: Tensor) -> Tuple[Tensor, Tensor]:
        """
            Input:
                obs: [n_batch, t]
                trans_logits: [n_batch, n_states, n_states] (Target, Source)
                obs_logits: [n_batch, t, n_states, n_outputs]
                init_logits: [n_batch, n_states]
            output:
                logits: [n_batch, t, n_outputs]
                state_posteriors: [n_batch, t, n_states]  P(h_t,y1...yt)
        """

        outputs = th.jit.annotate(List[Tensor], [])
        
        n_batch, _ = obs.shape
        batch_idx = th.arange(n_batch)

        log_alpha = F.log_softmax(init_logits, dim=1) # n_batch x n_states
        log_obs = F.log_softmax(obs_logits, dim=3) # n_batch x t x n_states x n_obs
        log_t = F.log_softmax(trans_logits, dim=1) # n_batch x n_states x n_states
        
        state_posteriors = th.jit.annotate(List[Tensor], [])
        for i in range(0, obs.shape[1]):
            
            # predict
            # B X S X O + B X S X 1
            log_py = th.logsumexp(log_obs[:,i,:,:] + log_alpha[:, :, None], dim=1)  # B x O
            log_py = log_py - th.logsumexp(log_py, dim=1)[:,None]
            outputs += [log_py]

            # update
            curr_y = obs[:,i]
            log_py = log_obs[batch_idx, i, :, curr_y] # B x S
            
            # B x 1 X S + B x 1 x S + B x S x S
            state_posteriors += [log_py + log_alpha]
            log_alpha = th.logsumexp(log_py[:,None,:] + log_alpha[:,None,:] + log_t, dim=2)
        
        outputs = th.stack(outputs)
        outputs = th.transpose(outputs, 0, 1)
        
        state_posteriors = th.stack(state_posteriors)
        state_posteriors = th.transpose(state_posteriors, 0, 1)

        return outputs, state_posteriors

    @jit.script_method
    def backward_(self,
                  obs: Tensor,
                  trans_logits: Tensor, 
                  obs_logits: Tensor, 
                  init_logits: Tensor) -> Tensor:
        """
            Input:
                obs: [n_batch, t]
                trans_logits: [n_batch, n_states, n_states] (Target, Source)
                obs_logits: [n_batch, t, n_states, n_outputs]
                init_logits: [n_batch, n_states]
            output:
                backward_logprobs: [n_batch, t, n_states]  P(y_{t+1}:T|h_t)
        """
        backward_logprobs = th.jit.annotate(List[Tensor], [])
        
        n_batch, _ = obs.shape
        batch_idx = th.arange(n_batch)

        
        log_obs = F.log_softmax(obs_logits, dim=3) # n_batch x t x n_states x n_obs
        log_t = F.log_softmax(trans_logits, dim=1) # n_batch x n_states x n_states
        
        log_beta = th.zeros_like(init_logits) # n_batch x n_states P(y_{i+1}:T|h_i)

        for i in range(obs.shape[1]-1, -1, -1):
            backward_logprobs += [log_beta]

            # probability of current observation P(y_i|h_i)
            curr_y = obs[:,i]
            log_py = log_obs[batch_idx, i, :, curr_y] # B x S
            

            # B x S x 1 + B x S x 1 + B x S x S = BxS
            log_beta = th.logsumexp(log_py[:,:,None] + log_beta[:,:,None] + log_t, dim=1)

        backward_logprobs = th.stack(backward_logprobs)
        backward_logprobs = th.transpose(backward_logprobs, 0, 1)

        return th.flip(backward_logprobs, dims=(1,))
    
@jit.script
def get_logits(dynamics_logits, obs_logits):
    
    trans_logits = th.hstack((  dynamics_logits[:, [0]]*0, # 1-pL
                                dynamics_logits[:, [1]],  # pF
                                dynamics_logits[:, [0]],  # pL
                                dynamics_logits[:, [1]]*0)).reshape((-1, 2, 2)) # 1-pF (Latent KCs x 2 x 2)
    obs_logits = th.concat((obs_logits[:, :, [0]]*0, # 1-pG
                            obs_logits[:, :, [0]],  # pG
                            obs_logits[:, :, [1]],  # pS
                            obs_logits[:, :, [1]]*0), dim=2).reshape((obs_logits.shape[0], -1, 2, 2)) # 1-pS (Latent KCs x T x 2 x 2)
    init_logits = th.hstack((dynamics_logits[:, [2]]*0, 
                             dynamics_logits[:, [2]])) # (Latent KCs x 2)

    return trans_logits, obs_logits, init_logits

def main():

    """ quick test of implementation against reference """
    
    # # BKT parameters probabilities
    # logit_pI0 = 0.3
    # logit_pG = -0.4
    # logit_pS = -1
    # logit_pL = -1
    # logit_pF = -2 

    # # sequence
    # obs = [0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
    # seq = np.zeros((len(obs), 2))
    # seq[np.arange(seq.shape[0]), obs] = 1
    
    # sigmoid = lambda x: 1/(1+np.exp(-x))

    # #
    # # reference probabilities
    # #
    # import model_brute_force_bkt 
    # probs = sigmoid(np.array([logit_pL, logit_pF, logit_pG, logit_pS, logit_pI0]))
    # ref_bkt_prob_corr = model_brute_force_bkt.forward_bkt(seq, *probs)
    # print(ref_bkt_prob_corr)

    # #
    # # BKT probabilities
    # #

    # corr = th.tensor([obs])
    
    # dynamics_logits = th.tensor([[logit_pL, logit_pF, logit_pI0]])
    
    # obs_logits = th.tensor([[logit_pG, logit_pS]]) # Bx2
    # obs_logits = th.tile(obs_logits, (1, corr.shape[1], 1)) # BxTx2
    
    # model = RnnBkt()
    # logpred, state_posteriors = model(corr, dynamics_logits, obs_logits)
    # print(logpred.exp()[0, :, 1].numpy())

    # # BxTxS - BxTx1 = BxTxS
    # normed_state_posteriors = state_posteriors - th.logsumexp(state_posteriors, dim=2)[:,:, None]
    # print(obs)
    # print(normed_state_posteriors.exp()[0, :, :].numpy())

    # smoothed_state_logbprobs = model.smoother(corr, dynamics_logits, obs_logits)
    # print(smoothed_state_logbprobs.exp().numpy())

    obs = [0, 0, 1, 0, 0]
    
    logit_pI0 = 0.0
    logit_pG = np.log(0.1 / 0.9)
    logit_pS = np.log(0.2 / 0.8)
    logit_pL = np.log(0.3 / 0.7)
    logit_pF = np.log(0.3 / 0.7)
    corr = th.tensor([obs])
    
    dynamics_logits = th.tensor([[logit_pL, logit_pF, logit_pI0]])
    
    obs_logits = th.tensor([[logit_pG, logit_pS]]) # Bx2
    obs_logits = th.tile(obs_logits, (1, corr.shape[1], 1)) # BxTx2
    model = RnnBkt()
    smoothed_state_logbprobs = model.smoother(corr, dynamics_logits, obs_logits)
    print(smoothed_state_logbprobs.exp().numpy())

if __name__ == "__main__":
    main()