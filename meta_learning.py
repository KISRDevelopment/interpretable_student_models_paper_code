import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 
import numpy.random as rng 
import pandas as pd 
from collections import defaultdict

import itertools
def main():

    test_loss()

    seqs = generate_dataset(100, 20, 10)

    obs, ytrue = to_arrays(seqs, 20)

    train(obs, ytrue)

def train(obs, ytrue):
    cfg = {
        "n_lstm_size" : 50,
        "lr" : 0.01,
        "epochs" : 50,
        "n_batch_size" : 10,
        "p_valid" : 0.2
    }
    model = KCDiscoverer(cfg['n_lstm_size'])
    optimizer = th.optim.NAdam(model.parameters(), lr=cfg['lr'])
    
    idx = rng.permutation(obs.shape[0])

    n_valid = int(cfg['p_valid'] * obs.shape[0])
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]
    
    train_obs = obs[train_idx, :]
    train_ytrue = ytrue[train_idx,:,:]

    valid_obs = th.tensor(obs[valid_idx, :]).float()
    valid_ytrue = th.tensor(ytrue[valid_idx, :, :]).float()

    print("Train size: %d, valid: %d" % (train_obs.shape[0], valid_obs.shape[0]))
    for e in range(cfg['epochs']):
        ix = rng.permutation(train_obs.shape[0])

        train_obs = train_obs[ix, :]
        train_ytrue = train_ytrue[ix, :, :]

        losses = []
        for offset in range(0, train_obs.shape[0], cfg['n_batch_size']):
            end = offset + cfg['n_batch_size']

            batch_obs = th.tensor(train_obs[offset:end, :]).float()
            batch_ytrue = th.tensor(train_ytrue[offset:end, :, :]).float()

            membership_logits = model(batch_obs)
            
            train_loss = min_filtering_loss(membership_logits, batch_ytrue)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            losses.append(train_loss.item())
        
        with th.no_grad():
            membership_logits = model(valid_obs)
            
            valid_loss = min_filtering_loss(membership_logits, valid_ytrue)
            
            print("%5d Train loss: %8.4f, Valid: %8.4f" % (e, np.mean(losses), valid_loss.numpy()))


def generate_dataset(n_students, n_skills, n_trials_per_skill):
    
    probs = generate_skill_params(n_skills)

    seqs = defaultdict(lambda: {
        "obs" : [],
        "skill" : []
    })


    pL = probs[:,1]
    pF = probs[:,2]
    pG = probs[:,3]
    pS = probs[:,4]

    for student in range(n_students):

        skill_seq = rng.permutation(np.tile(np.arange(n_skills), n_trials_per_skill))
        
        state = rng.binomial(1, probs[:,0])
        
        obs_seq = []
        for t in range(skill_seq.shape[0]):
            skill = skill_seq[t]

            pC = (1 - pS[skill]) * state[skill] + (1-state[skill]) * pG[skill] 
            
            ans = rng.binomial(1, pC)
            
            state[skill] = rng.binomial(1, (1 - pF[skill]) * state[skill] + (1-state[skill]) * pL[skill])
            obs_seq.append(ans)

        seqs[student]["obs"] = obs_seq
        seqs[student]["skill"] = skill_seq
    
    return seqs

def generate_skill_params(n_skills):
   
    # possible parameter values
    pIs = [0.1, 0.25, 0.5, 0.75, 0.9]
    pLs = [0.01, 0.05, 0.1, 0.2] 
    pFs = [0.01, 0.05, 0.1, 0.2]
    pGs = [0.1, 0.2, 0.3, 0.4]
    pSs = [0.1, 0.2, 0.3, 0.4]

    all_prob_combs = np.array(list(itertools.product(pIs, pLs, pFs, pGs, pSs)))

    print("Choosing from %d combinations with replacement" % all_prob_combs.shape[0])

    probs = all_prob_combs[rng.choice(all_prob_combs.shape[0], replace=True, size=n_skills), :]
    
    # n_skills x 5
    return probs

def to_arrays(seqs, n_skills):

    obs_seq = np.array([s['obs'] for _, s in seqs.items()])

    possible_answers_seqs = []
    for _, s in seqs.items():
        kc_seq = np.array(s['skill'])

        container = []
        for kc in range(n_skills):
            ix = kc_seq == kc 
            container.append(ix.astype(int))
        
        possible_answers_seqs.append(container)
    
    ytrue = np.array(possible_answers_seqs)
    
    return obs_seq, ytrue


def test_loss():

    membership_logits = th.tensor([[-10, 10, 10, 10, -10]]).float() 
    ytrue = th.tensor([
        [
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ]
    ]).float()

    print(min_filtering_loss(membership_logits, ytrue))


class KCDiscoverer(nn.Module):

    def __init__(self, n_lstm_size):
        super().__init__()
        self.cell = nn.LSTM(1, 
            n_lstm_size, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True)
        self.ff = nn.Linear(n_lstm_size*2, 1)
    def forward(self, obs_seq):
        """
            Input: 
                obs_seq: BxT
            Output:
                membership_logits: BxT
        """
        
        cell_output,_ = self.cell(obs_seq[:,:,None])
        logits = self.ff(cell_output)[:,:,0]
        return logits
        
def min_filtering_loss(membership_logits, ytrue):
    """
        Input:
            membership_logits: BxT
            ytrue: BxKxT
            where K is the number of valid answers
        Output:
            loss
    """
    membership_log_probs = F.logsigmoid(membership_logits) # BxT
    nonmembership_log_probs = F.logsigmoid(-membership_logits) # BxT
    
    # Bx1xT * BxKxT = BxKxT
    ll = membership_log_probs[:,None,:] * ytrue  + nonmembership_log_probs[:,None,:] * (1-ytrue)

    # BxT
    mfl,_ = th.max(ll, dim=1)

    return -mfl.mean()

if __name__ == "__main__":
    main()
