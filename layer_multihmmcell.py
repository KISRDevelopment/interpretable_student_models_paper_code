import numpy as np
import metrics
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from typing import List, Tuple


class MultiHmmCell(jit.ScriptModule):
    
    def __init__(self):
        super(MultiHmmCell, self).__init__()

    @jit.script_method
    def forward(self, obs: Tensor, 
        chain: Tensor, 
        log_trans: Tensor, 
        log_obs: Tensor, 
        init_logits: Tensor) -> Tensor:

        n_batch = obs.shape[0]
        log_alpha = F.log_softmax(init_logits, dim=1) # n_chains x n_states
        log_alpha = th.tile(log_alpha, (n_batch, 1, 1)) # batch x chains x states
        return self.forward_given_alpha(obs, chain, log_trans, log_obs, log_alpha)[0]
    
    @jit.script_method
    def forward_given_alpha(self, obs: Tensor, chain: Tensor, 
        log_t: Tensor, 
        log_obs: Tensor, 
        log_alpha: Tensor) -> Tuple[Tensor, Tensor]:
        """
            Input:
                obs: [n_batch, t]
                chain: [n_batch, t, n_chains]
                log_trans: [n_chains, n_states, n_states] (Target, Source)
                log_obs: [n_chains, n_states, n_outputs]
                log_alpha: [n_batch, n_chains, n_states]
            output:
                logits: [n_batch, t, n_outputs]
                log_alpha: [n_batch, n_chains, n_states]
        """

        n_chains, n_states, n_outputs = log_obs.shape 

        outputs = th.jit.annotate(List[Tensor], [])
        
        n_batch, _ = obs.shape
        batch_idx = th.arange(n_batch)

        
        #log_obs = F.log_softmax(obs_logits, dim=2) # n_chains x n_states x n_obs
        #log_t = F.log_softmax(trans_logits, dim=1) # n_chains x n_states x n_states
        
        # B X C X S
        for i in range(0, obs.shape[1]):
            curr_chain = chain[:,i,:] # B X C
            
            # predict
            a1 = (curr_chain[:,:,None, None] * log_obs[None,:,:,:]).sum(1) # B X S X O
            a2 = (curr_chain[:,:,None] * log_alpha).sum(1) # BXCX1 * BXCXS = BXS

            # B X S X O + B X S X 1
            log_py = th.logsumexp(a1 + a2[:,:,None], dim=1)  # B X O
            
            log_py = log_py - th.logsumexp(log_py, dim=1)[:,None]
            outputs += [log_py]

            # update
            curr_y = obs[:,i]
            a1 = th.permute(log_obs[:,:,curr_y], (2, 0, 1)) # B X C X S
            log_py = (a1 * curr_chain[:,:,None]).sum(1) # B X S
            

            a1 = (log_alpha * curr_chain[:,:,None]).sum(1) # BxCxS * BxCx1 = BxS
            a2 = (log_t[None,:,:,:] * curr_chain[:,:,None,None]).sum(1) # 1xCxSxS * BxCx1x1 = BxSxS
            a3 = th.logsumexp(log_py[:,None,:] + a1[:,None,:] + a2, dim=2)

            # B x 1 X S + B x 1 x S + B x S x S = B x S
            log_alpha = (1 - curr_chain[:,:,None]) * log_alpha + curr_chain[:,:,None] * a3[:,None,:]
        
        
        outputs = th.stack(outputs)
        outputs = th.transpose(outputs, 0, 1)
        
        return outputs, log_alpha
