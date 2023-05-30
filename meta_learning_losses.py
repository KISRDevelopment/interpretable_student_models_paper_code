import numpy as np 
import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.jit as jit
from torch import Tensor
from typing import List, Tuple
import numpy.random as rng

def main():
    
    #test_logprob_assignment()
    
    #test_get_kc_params()

    #test_logprob_sequence()
    
    test_loss()

def test_logprob_assignment():
    in_cluster = th.tensor([
        [
            [
                0, 0, 1, 1, 0, 1
            ],
            [
                1, 1, 0, 0, 1, 0
            ]
        ]
    ])
    n_batch, n_skills, n_trials = in_cluster.shape 

    membership_logits = th.tensor([
        [
            10, 10, 10, 10, 10, -10
        ]
    ]).float()
    
    obs_mask = th.ones_like(membership_logits)
    obs_mask[:, 2:4] = 0

    ll = logprob_assignment(membership_logits, in_cluster, obs_mask)
    print(ll)

def test_get_kc_params():
    n_batch = 1
    n_skills = 3

    kc_logits = th.randn((n_batch, 5))
    
    a,b,c = get_kc_params(kc_logits, n_skills)

    print(a.shape)
    print(b.shape)
    print(c.shape)

    print(a)
    print(b)
    print(c)

def test_logprob_sequence():

    n_trials = 5
    n_batch = 1
    n_skills = 3

    obs = th.tensor(rng.binomial(1, 0.5, (n_batch, n_trials)))
    
    chain = rng.randint(low=0, high=n_skills, size=(n_batch, n_trials))
    in_cluster = th.permute(F.one_hot(th.tensor(chain)), (0, 2, 1))
    
    kc_logits = th.randn((n_batch, 5))
    obs_mask = th.ones_like(obs)
    
    print(chain)
    ll = logprob_sequence(obs, in_cluster, kc_logits, obs_mask)
    print(ll)

def test_loss():

    n_trials = 50
    n_batch = 100
    n_skills = 20

    obs = th.tensor(rng.binomial(1, 0.5, (n_batch, n_trials)))
    
    chain = rng.randint(low=0, high=n_skills, size=(n_batch, n_trials))
    in_cluster = th.permute(F.one_hot(th.tensor(chain)), (0, 2, 1))
    
    kc_logits = th.randn((n_batch, 5))
    obs_mask = th.ones_like(obs)
    skill_mask = th.ones((n_batch, n_skills))

    membership_logits = th.randn((n_batch, n_trials))
    
    loss = loss_func(obs, membership_logits, kc_logits, in_cluster, obs_mask, skill_mask)
    print(loss)

def loss_func(obs, membership_logits, kc_logits, in_cluster, obs_mask, skill_mask):

    ll_obs = logprob_assignment(membership_logits, in_cluster, obs_mask) # BxK
    ll_seq = logprob_sequence(obs, in_cluster, kc_logits, obs_mask) # BxK

    # BxK
    joint_ll = (ll_obs + ll_seq) * skill_mask + (1-skill_mask) * -10

    mfl,_ = th.max(joint_ll, dim=1)
    
    return -mfl.mean()

def logprob_assignment(membership_logits, in_cluster, obs_mask):
    """
        Computes the logprobability of possible memberships, given the membership_logits.

        Input:
            membership_logits: probability of the cluster including the trial (BxT)
            in_cluster: possible clusters (BxKxT)
            obs_mask: which trials to include in calculation (BxT)
        Output:
            logprob (BxK)
    """
    membership_log_probs = F.logsigmoid(membership_logits) # BxT
    nonmembership_log_probs = F.logsigmoid(-membership_logits) # BxT
    
    # compute individual logprobs
    # Bx1xT * BxKxT = BxKxT
    ll = membership_log_probs[:,None,:] * in_cluster  + nonmembership_log_probs[:,None,:] * (1-in_cluster)

    # mask out irrelevant trials
    # Bx1xT * BxKxT = BxKxT
    ll = obs_mask[:,None,:] * ll

    # compute loglikelihood for each possible cluster
    # BxK / Bx1 = BxK
    ll = ll.sum(2) #/ obs_mask.sum(1)[:,None]

    return ll

def logprob_sequence(obs, in_cluster, kc_param_logits, obs_mask):
    """
        Computes the logprobability of observations belonging to the cluster 
        using BKT.

        Input:
            obs: the observation sequence (BxT)
            in_cluster: possible memberships (BxKxT)
            kc_param_logits: BKT parameter logits (Bx5)
            obs_mask: which trials to include in calculation (BxT)
        Output:
            logprob: log probability of the observations, for each membership (BxK)
    """

    n_skills = in_cluster.shape[1]

    trans_logits, obs_logits, init_logits = get_kc_params(kc_param_logits, n_skills)
    chain = th.permute(in_cluster, (0, 2, 1)) # BxTxK

    log_obs, _ = forward(obs, chain, trans_logits, obs_logits, init_logits) # BxTxO

    # BxT
    ll = log_obs[:,:,1] * obs + log_obs[:,:,0] * (1-obs)
    
    mask = obs_mask[:,:,None] * chain # BxTxK, which trials to include when calculating ll per skill
    log_obs_per_skill = ll[:,:,None] * mask # BxTx1 * BxTxK = BxTxK

    log_seq_per_skill = log_obs_per_skill.sum(1) # BxK

    return log_seq_per_skill

def get_kc_params(kc_logits, n_skills):
    """
        Input:
            kc_logits: KC parameter logits (Bx5)
        
        Output:
            trans_logits: Bxn_skillsx2x2
            obs_logits: Bxn_skillsx2x2
            init_logits: Bxn_skillsx2
    """

    trans_logits = th.hstack((-kc_logits[:, [0]], # 1-pL
                                  kc_logits[:, [1]],  # pF
                                  kc_logits[:, [0]],  # pL
                                  -kc_logits[:, [1]])).reshape((-1, 2, 2)) # 1-pF (B x 2 x 2)
    obs_logits = th.hstack((-kc_logits[:, [2]], # 1-pG
                                  kc_logits[:, [2]],  # pG
                                  kc_logits[:, [3]],  # pS
                                  -kc_logits[:, [3]])).reshape((-1, 2, 2)) # 1-pS (B x 2 x 2)
    init_logits = th.hstack((-kc_logits[:, [4]], kc_logits[:, [4]])) # (B x 2)

    trans_logits = th.tile(trans_logits[:, None, :, :], (1, n_skills, 1, 1))
    obs_logits = th.tile(obs_logits[:, None, :, :], (1, n_skills, 1, 1))
    init_logits = th.tile(init_logits[:, None, :], (1, n_skills, 1))

    return trans_logits, obs_logits, init_logits

    
@jit.script
def forward_given_alpha(obs: Tensor, chain: Tensor, 
        trans_logits: Tensor, 
        obs_logits: Tensor, 
        init_logits: Tensor,
        log_alpha: Tensor) -> Tuple[Tensor, Tensor]:
        """
            Input:
                obs: [n_batch, t]
                chain: [n_batch, t, n_chains]
                trans_logits: [n_batch, n_chains, n_states, n_states] (Target, Source)
                obs_logits: [n_batch, n_chains, n_states, n_outputs]
                init_logits: [n_batch, n_chains, n_states]
                log_alpha: [n_batch, n_chains, n_states]
            output:
                logits: [n_batch, t, n_outputs]
                log_alpha: [n_batch, n_chains, n_states]
        """

        n_batch, n_chains, n_states, n_outputs = obs_logits.shape 

        outputs = th.jit.annotate(List[Tensor], [])
        
        batch_idx = th.arange(n_batch)

        log_obs = F.log_softmax(obs_logits, dim=3) # batch x n_chains x n_states x n_obs
        log_t = F.log_softmax(trans_logits, dim=2) # batch x n_chains x n_states x n_states
        
        # B X C X S
        for i in range(0, obs.shape[1]):
            curr_chain = chain[:,i,:] # B X C
            
            # predict
            a1 = (curr_chain[:,:,None, None] * log_obs).sum(1) # B X S X O
            a2 = (curr_chain[:,:,None] * log_alpha).sum(1) # BXCX1 * BXCXS = BXS

            # B X S X O + B X S X 1
            log_py = th.logsumexp(a1 + a2[:,:,None], dim=1)  # B X O
            
            log_py = log_py - th.logsumexp(log_py, dim=1)[:,None]
            outputs += [log_py]

            # update
            curr_y = obs[:,i]
            a1 = log_obs[batch_idx, :,:,curr_y] # B X C X S
            
            log_py = (a1 * curr_chain[:,:,None]).sum(1) # B X S
            

            a1 = (log_alpha * curr_chain[:,:,None]).sum(1) # BxCxS * BxCx1 = BxS
            a2 = (log_t * curr_chain[:,:,None,None]).sum(1) # BxCxSxS * BxCx1x1 = BxSxS
            a3 = th.logsumexp(log_py[:,None,:] + a1[:,None,:] + a2, dim=2)

            # B x 1 X S + B x 1 x S + B x S x S = B x S
            log_alpha = (1 - curr_chain[:,:,None]) * log_alpha + curr_chain[:,:,None] * a3[:,None,:]
        
        
        outputs = th.stack(outputs)
        outputs = th.transpose(outputs, 0, 1)
        
        return outputs, log_alpha


@jit.script
def forward(obs: Tensor, chain: Tensor, 
        trans_logits: Tensor, 
        obs_logits: Tensor, 
        init_logits: Tensor) -> Tuple[Tensor, Tensor]:

        n_batch = obs.shape[0]
        log_alpha = F.log_softmax(init_logits, dim=2) # batch x n_chains x n_states
        
        return forward_given_alpha(obs, chain, trans_logits, obs_logits, init_logits, log_alpha)

if __name__ == "__main__":
    main()