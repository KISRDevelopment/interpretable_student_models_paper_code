import torch as th
import torch.nn as nn
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor
import numpy as np

class BKTCell(jit.ScriptModule):
    def __init__(self, n_kcs):
        super(BKTCell, self).__init__()
        self.n_kcs = n_kcs
        self.kc_logits = nn.Embedding(n_kcs, 5) # learning, forgetting, guessing, and slipping, initial logits
        
    @jit.script_method
    def forward(self, input: Tuple[Tensor, Tensor, Tensor], state: Tensor) -> Tuple[Tensor, Tensor]:
        """
            state: p(h_(t-1) = 1| y_1, ..., y_(t-2)) [n_batch, n_kcs]
            input: 
                previous KC [n_batch]
                current KC [n_batch]
                previous correct [n_batch]
        """
        
        prev_kc, curr_kc, prev_corr = input 
        
        batch_ix = th.arange(prev_kc.shape[0])
        
        prev_logits = self.kc_logits(prev_kc) # [n_batch, 5]
        prev_probs = th.sigmoid(prev_logits) # [n_batch, 5]
        curr_logits = self.kc_logits(curr_kc) #[n_batch, 5]
        curr_probs = th.sigmoid(curr_logits)
        
        # compute probability of correctness given h (0 or 1)
        p_correct_given_h = prev_probs[:, [2, 3]] # [n_batch, 2]
        
        # compute probability of previous steps' output given h [n_batch, 2]
        p_output_given_h = th.pow(p_correct_given_h, prev_corr[:,None]) * th.pow(1-p_correct_given_h, 1-prev_corr[:,None])
        
        # compute filtering distribution p(h_(t-1) = 1 | y_1 .. y_(t-1))
        skill_state = state[batch_ix, prev_kc] # [n_batch]
        # [n_batch]
        filtering = (p_output_given_h[:,1] * skill_state) / (p_output_given_h[:,0]*(1-skill_state) + p_output_given_h[:,1]*skill_state)
        
        # compute predictive distribution p(h_t=1|y_1...y_(t-1))
        p_learning = prev_probs[:,0]
        p_forgetting = prev_probs[:,1]
        predictive = p_learning * (1-filtering) + (1-p_forgetting) * filtering
        
        # update relevant entry
        state[batch_ix,prev_kc] = predictive
        
        # grab  predictive state at current time step
        curr_state = state[batch_ix, curr_kc] #[n_batch]
        
        p_correct_given_curr_h = curr_probs[:, [2, 3]] # [n_batch, 2]
        p_curr_correct = p_correct_given_curr_h[:,0] * (1-curr_state) + p_correct_given_curr_h[:,1] * curr_state
        
        return p_curr_correct, state
    
    @jit.script_method
    def forward_first_trial(self, curr_kc: Tensor) -> Tuple[Tensor, Tensor]:
        """
            curr_kc: [n_batch]
        """
        logits = self.kc_logits(th.arange(self.n_kcs))
        
        state = th.tile(th.sigmoid(logits[:,[4]].T), (curr_kc.shape[0], 1)) # [n_batch, n_kcs]
        
        return self.forward_first_trial_from_state(curr_kc, state)
    
    @jit.script_method
    def forward_first_trial_from_state(self, curr_kc: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
            curr_kc: [n_batch]
        """
        batch_ix = th.arange(curr_kc.shape[0])
        
        logits = self.kc_logits(th.arange(self.n_kcs))
        
        curr_logits = self.kc_logits(curr_kc) #[n_batch, 5]
        curr_probs = th.sigmoid(curr_logits)
        
        curr_state = state[batch_ix, curr_kc] #[n_batch]
        
        p_correct_given_curr_h = curr_probs[:, [2, 3]] # [n_batch, 2]
        p_curr_correct = p_correct_given_curr_h[:,0] * (1-curr_state) + p_correct_given_curr_h[:,1] * curr_state
        
        return p_curr_correct, state 
    
class BKTLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(BKTLayer, self).__init__()
        self.cell = cell(*cell_args)
        
    @jit.script_method
    def forward(self, prev_kc: Tensor, curr_kc: Tensor, prev_corr: Tensor) -> Tuple[Tensor, Tensor]:
        """
            prev_kc: [n_batch, t]
            curr_kc: [n_batch, t]
            prev_corr: [n_batch, t]
        """
        outputs = th.jit.annotate(List[Tensor], [])
        
        pc, state = self.cell.forward_first_trial(curr_kc[:,0])
        outputs += [pc]
        
        for i in range(1, prev_kc.shape[1]):
            pc, state = self.cell((prev_kc[:,i], curr_kc[:, i], prev_corr[:,i]), state)
            
            outputs += [pc]
        return th.stack(outputs).T, state

    @jit.script_method
    def forward_from_state(self, prev_kc: Tensor, curr_kc: Tensor, prev_corr: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
            prev_kc: [n_batch, t]
            curr_kc: [n_batch, t]
            prev_corr: [n_batch, t]
            state: [n_batch, n_kcs]
        """
        outputs = th.jit.annotate(List[Tensor], [])
        
        pc, state = self.cell.forward_first_trial_from_state(curr_kc[:,0], state)
        outputs += [pc]
        
        for i in range(1, prev_kc.shape[1]):
            pc, state = self.cell((prev_kc[:,i], curr_kc[:, i], prev_corr[:,i]), state)
            
            outputs += [pc]
        return th.stack(outputs).T, state

class BKTModel(nn.Module):
    def __init__(self, n_kcs):
        super(BKTModel, self).__init__()
        
        self.bktlayer = BKTLayer(BKTCell, n_kcs)

    def forward(self, prev_kc, curr_kc, prev_corr):
        probs, state = self.bktlayer(prev_kc, curr_kc, prev_corr)
        
        return probs, state

    def forward_from_state(self, prev_kc, curr_kc, prev_corr, state):
        probs, state = self.bktlayer.forward_from_state(prev_kc, curr_kc, prev_corr, state)
        
        return probs, state

        