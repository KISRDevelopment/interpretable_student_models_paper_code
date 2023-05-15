#import tensorflow as tf
import numpy as np 
import itertools 
import torch as th 
import torch.nn.functional as F 

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


if __name__ == "__main__":
    main()