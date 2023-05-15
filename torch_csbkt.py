
import numpy as np 
import itertools 
import torch as th 
import torch.nn.functional as F 
from torch import Tensor
from typing import List, Tuple
import torch.nn as nn 

def main():
    n_skills = 3

    decoder = make_state_decoder_matrix(n_skills)
    #print(decoder)
    assert decoder.shape[0] == 2**n_skills

    im = make_transition_indicator_matrix(decoder)
    print(im)


    logit_pL = th.logit(th.tensor([0.2, 0.5, 0.05]).float()[:,None])
    logit_pF = th.logit(th.tensor([0.1, 0.3, 0.05]).float()[:,None])

    A = make_transition_log_prob_matrix(im, logit_pL, logit_pF)
    print(A.exp())
    assert th.allclose(A.exp().sum(1), th.ones(A.shape[0]))

    logit_pi = th.logit(th.tensor([0.2, 0.5, 0.3])[:, None])
    initial_log_prob_vec = make_initial_log_prob_vector(decoder, logit_pi)
    print(initial_log_prob_vec.exp().sum())
    
    model = HmmCell(32, 2)
    n_batch = 4 
    timesteps = 7 

    trans_log_probs = A # [n_states, n_states]
    init_log_probs = th.tile(initial_log_prob_vec, (1, n_batch)).T # [n_batch, n_states]
    
    obs_probs = th.softmax(th.randn(2**n_skills, 2), axis=1) # [n_states, 2]
    obs_probs = th.tile(obs_probs[None, :, :], (timesteps, 1, 1))
    obs_probs = th.tile(obs_probs[None, :, :, :], (n_batch, 1, 1, 1))
    obs_log_probs = obs_probs.log()
    
    obs = np.random.binomial(1, 0.5, (n_batch, timesteps))
    obs = th.tensor(obs).long()

    n_problems = 20
    nido_layer = NIDOLayer(n_problems, decoder)

    problem_seq = np.tile(np.random.permutation(n_problems)[:timesteps][None,:], (n_batch, 1))
    problem_seq = th.tensor(problem_seq).long()

    obs_log_probs = nido_layer(problem_seq)
    result,state = model(obs, trans_log_probs, obs_log_probs, init_log_probs)
    print(result.exp())
    print(state.exp())


def make_state_decoder_matrix(k):
    """
        creates a lookup table that maps integer states into
        the binary states of each core skill.
        Arguments:
            k: the number of core skills
        
        Returns:
            decoder: (2**K, K) matrix (Pytorch)
    """
    r = list(itertools.product([0, 1], repeat=k))
    return th.tensor(r).float()

def make_transition_indicator_matrix(decoder):
    """
        Returns a matrix that specifies how each skill transitions 
        skill transitions (0 -> 0, 0 -> 1, 1 -> 0, 1 -> 1)
        
        Inputs:
            decoder: the integer->state combination mapping [n_total, n_skills]
        
        Output:
            transition indictor_mat: for each source and target states of the
            HMM, specifies how each of the core n_skills transitioned.
                [n_total, n_total, n_skills, 4]
                 Source,  Target
                 (Pytorch)
    """

    n_total = decoder.shape[0]
    n_skills = decoder.shape[1]

    im = np.zeros((n_total, n_total, n_skills, 4))
    
    for i in range(n_total):
        from_h = decoder[i, :]
        for j in range(n_total):
            to_h = decoder[j, :]
            diff = to_h - from_h
            for l in range(n_skills):
                im[i, j, l, 0] = (diff[l] == 0) & (from_h[l] == 0)
                im[i, j, l, 1] = diff[l] == 1
                im[i, j, l, 2] = diff[l] == -1
                im[i, j, l, 3] = (diff[l] == 0) & (from_h[l] == 1)
    
    return th.tensor(im).float()

@th.jit.script
def make_initial_log_prob_vector(decoder, logit_pi):
    """
        (Pytorch function)
        Creates the initial probability distribution over states of the full HMM.

        Inputs:
            decoder     [n_total, n_skills]
            logit_pi    [n_skills, 1]
        Output:
            log_prob of each state
    """

    logit_pi = logit_pi.T # [1, n_skills]

    logit_probs = decoder * logit_pi + (1-decoder) * (-logit_pi) # [n_total, n_skills]

    log_pinitial = F.logsigmoid(logit_probs)
    
    return log_pinitial.sum(1)[:,None]

@th.jit.script
def make_transition_log_prob_matrix(im, logit_pL, logit_pF):
    """
        (Pytorch function)

        Given the individual KC's learning and forgetting probabilities,
        this constructs the transition matrix for the full HMM.

        Inputs:
            im: transition_indicator_matrix  [n_total, n_total, n_skills, 4]
            logit_pL: logit of probability of learning    [n_skills,1]
            logit_pF: logit of probability of forgetting  [n_skills,1]
        
        Output:
            T: the log transition matrix of the full HMM [n_total, n_total]
    """
    
    # [n_skills, 4]
    log_prob_matrix = F.logsigmoid(th.concat((-logit_pL, logit_pL, logit_pF, -logit_pF), dim=1))

    # compute log probability for each skill for each state transition
    # [n_total, n_total, n_skills]
    U = (im * log_prob_matrix[None, None, :, :]).sum(3)

    # finally, compute summation to get the log transition matrix
    return U.sum(2) # [n_total, n_total]

class HmmCell(th.jit.ScriptModule):
    
    def __init__(self, n_states, n_outputs):
        super(HmmCell, self).__init__()
        
        self.n_states = n_states
        self.n_outputs = n_outputs
        
    @th.jit.script_method
    def forward(self, obs: Tensor,
                      trans_log_probs: Tensor, 
                      obs_log_probs: Tensor, 
                      init_log_probs: Tensor) -> Tuple[Tensor, Tensor]:
        """
            input:
                obs: [n_batch, t]
                trans_log_probs: [n_states, n_states] (Target, Source)
                obs_log_probs:   [n_batch, t, n_states, n_outputs]
                init_log_probs:  [n_batch, n_states]
            output:
                log_output_prob: [n_batch, t, n_outputs]
                log_alpha: [n_batch, n_states]
        """
        outputs = th.jit.annotate(List[Tensor], [])
        batch_idx = th.arange(init_log_probs.shape[0])

        # [n_batch, n_states]
        log_alpha = init_log_probs
        for i in range(0, obs.shape[1]):
            
            # predict
            # B X S X O + B X S X 1
            log_py = th.logsumexp(obs_log_probs[:,i,:,:] + log_alpha[:, :, None], dim=1)  #[n_batch, n_obs]
            log_py = log_py - th.logsumexp(log_py, dim=1)[:,None]
            outputs += [log_py]

            # update
            curr_y = obs[:,i]
            log_py = obs_log_probs[batch_idx, i, :, curr_y] # [n_batch, n_states]
            
            # B x 1 X S + B x 1 x S + 1 x S x S
            log_alpha = th.logsumexp(log_py[:,None,:] + log_alpha[:,None,:] + trans_log_probs[None,:,:], dim=2)
        
        outputs = th.stack(outputs)
        outputs = th.transpose(outputs, 0, 1)
        
        return outputs, log_alpha

class NIDOLayer(nn.Module):

    def __init__(self, n_problems, decoder):
        super().__init__()

        n_states, n_skills = decoder.shape 

        self.skill_offset = nn.Parameter(th.randn(n_skills))
        self.skill_slope = nn.Parameter(th.randn(n_skills))
        self.membership_logits = nn.Parameter(th.randn(n_problems, n_skills))
        
        self.decoder = decoder # [n_states, n_skills]

    def forward(self, problem_seq):
        """
            Input:
                problem_seq: [n_batch, t]
            Output:
                obs_log_probs: [n_batch, t, n_states, 2]
        """

        #
        # compute state logits
        # 1xK + (1xK) * (SxK) = SxK 
        # [n_states, n_skills]
        #
        state_logits = self.skill_offset[None,:] + self.skill_slope[None,:] * self.decoder

        #
        # compute problem logits
        #
        membership_probs = th.sigmoid(self.membership_logits) # [n_problems, n_skills]
        problem_logits = membership_probs @ state_logits.T # [n_problems, n_states]

        obs_correct_logits = problem_logits[problem_seq] # [n_batch, t, n_states]

        # [n_batch, t, n_states, 2]
        obs_log_probs = F.log_softmax(th.concat((-obs_correct_logits[:,:,:,None], obs_correct_logits[:,:,:,None]), dim=3), dim=3)

        return obs_log_probs


        
if __name__ == "__main__":
    main()