import numpy as np 
from scipy.stats import qmc

def initialize_params(n_kcs):
    log2_n_skills = int(np.ceil(np.log(n_kcs) / np.log(2)))

    sampler = qmc.Sobol(d=5, scramble=True)

    # pI, pL, pF, pG, pS
    probs = sampler.random_base2(m=log2_n_skills)[:n_kcs, :]
    
    # Target x Source
    idx = np.arange(n_kcs)
    trans_probs = np.zeros((n_kcs, 2, 2))
    trans_probs[idx, 1, 0] = probs[:, 1]
    trans_probs[idx, 0, 0] = 1 - probs[:, 1]
    trans_probs[idx, 1, 1] = 1 - probs[:, 2]
    trans_probs[idx, 0, 1] = probs[:, 2]

    # States x Outputs
    obs_probs = np.zeros((n_kcs, 2, 2))
    obs_probs[idx, 0, 0] = 1 - probs[:,3]
    obs_probs[idx, 0, 1] = probs[:,3]
    obs_probs[idx, 1, 0] = probs[:,4]
    obs_probs[idx, 1, 1] = 1 - probs[:, 4]

    initial_probs = np.zeros((n_kcs, 2))
    initial_probs[idx, 0] = 1 - probs[:, 0]
    initial_probs[idx, 1] = probs[:, 0]

    # transform to logits
    return np.log(trans_probs), np.log(obs_probs), np.log(initial_probs)
    
initialize_params(10)