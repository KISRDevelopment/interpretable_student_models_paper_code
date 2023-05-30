import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_sequence

import numpy as np 
import numpy.random as rng 
import pandas as pd 
from collections import defaultdict

import sklearn.metrics
import copy 
import itertools

import meta_learning_losses
def main():

    #test_loss()

    train()

def train():
    cfg = {
        "n_lstm_size" : 10,
        "lr" : 0.5,
        "epochs" : 200,
        "n_batch_size" : 5,
        "n_train_seqs" : 80,
        "n_valid_seqs" : 20,
        "max_n_skills" : 10,
        "max_n_trials_per_skill" : 10,
        "patience" : 50
    }
    model = KCDiscoverer(cfg['n_lstm_size'])
    optimizer = th.optim.NAdam(model.parameters(), lr=cfg['lr'])
    

    # generate new set
    train_seqs = generate_dataset(cfg['n_train_seqs'], cfg['max_n_skills'], cfg['max_n_trials_per_skill'])
    valid_seqs = generate_dataset(cfg['n_valid_seqs'], cfg['max_n_skills'], cfg['max_n_trials_per_skill'])
        
    best_auc_roc = 0
    waited = 0
    best_state = None 
    for e in range(cfg['epochs']):
        
        losses = []
        for offset in range(0, len(train_seqs), cfg['n_batch_size']):
            end = offset + cfg['n_batch_size']

            batch_seqs = train_seqs[offset:end]

            batch_obs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0).long() # BxT
            
            obs_mask = (pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) >= 0).float() # BxT
            batch_skill_seqs = pad_sequence([th.tensor(s['skill']) for s in batch_seqs], batch_first=True, padding_value=-1)
            batch_ytrue, skill_mask = make_ytrue(batch_skill_seqs, cfg['max_n_skills']) # BxKxT

            membership_logits, kc_logits = model(batch_obs) # BxT and Bx5

            train_loss = meta_learning_losses.loss_func(batch_obs, membership_logits, kc_logits, batch_ytrue, obs_mask, skill_mask)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            losses.append(train_loss.item())
        
        with th.no_grad():
            valid_obs = pad_sequence([th.tensor(s['obs']) for s in valid_seqs], batch_first=True, padding_value=0).float() # BxT
            obs_mask = (pad_sequence([th.tensor(s['obs']) for s in valid_seqs], batch_first=True, padding_value=-1) >= 0).float() # BxT
            batch_skill_seqs = pad_sequence([th.tensor(s['skill']) for s in valid_seqs], batch_first=True, padding_value=-1)
            valid_ytrue, skill_mask = make_ytrue(batch_skill_seqs, cfg['max_n_skills']) # BxKxT

            membership_logits, _ = model(valid_obs)
            
            auc_roc = min_filtering_auc_roc(membership_logits.cpu().numpy(), obs_mask.cpu().numpy(), valid_ytrue.cpu().numpy(), skill_mask.cpu().numpy())
            
            if auc_roc > best_auc_roc:
                best_auc_roc = auc_roc
                waited = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                waited += 1
            
            print(membership_logits.min(), membership_logits.max())
            print("%5d Train loss: %8.4f, AUC-ROC: %0.2f %s" % (e, np.mean(losses), auc_roc, '***' if waited == 0 else ''))

            if waited >= cfg['patience']:
                break 
    
    model.load_state_dict(best_state)


def generate_dataset(n_students, max_n_skills, max_n_trials_per_skill):
    
    probs = generate_skill_params(max_n_skills)

    seqs = []

    pL = probs[:,1]
    pF = probs[:,2]
    pG = probs[:,3]
    pS = probs[:,4]

    for student in range(n_students):
        
        #
        # determine skills that were practiced
        #
        n_skills = rng.choice(max_n_skills) + 1
        skills = rng.choice(max_n_skills, size=n_skills, replace=False)

        #
        # generate skill sequence
        #
        skill_seq = []
        for skill in skills:
            # determine number of trials
            #n_skill_trials = rng.choice(max_n_trials_per_skill) + 1
            n_skill_trials = max_n_trials_per_skill
            skill_seq.extend([skill] * n_skill_trials)
        rng.shuffle(skill_seq)    
        
        #
        # generate BKT responses
        #
        state = rng.binomial(1, probs[:,0])
        obs_seq = []
        for t in range(len(skill_seq)):
            skill = skill_seq[t]

            pC = (1 - pS[skill]) * state[skill] + (1-state[skill]) * pG[skill] 
            
            ans = rng.binomial(1, pC)
            
            state[skill] = rng.binomial(1, (1 - pF[skill]) * state[skill] + (1-state[skill]) * pL[skill])
            obs_seq.append(ans)

        seqs.append({
            "obs" : obs_seq,
            "skill" : skill_seq
        })
        
    return seqs


def generate_skill_params(n_skills):
   
    # possible parameter values
    pIs = [0.1, 0.25, 0.5, 0.75, 0.9]
    pLs = [0.01, 0.05, 0.1, 0.2] 
    pFs = [0.01, 0.05, 0.1, 0.2]
    pGs = [0.1, 0.2, 0.3, 0.4]
    pSs = [0.1, 0.2, 0.3, 0.4]

    all_prob_combs = np.array(list(itertools.product(pIs, pLs, pFs, pGs, pSs)))

    #print("Choosing from %d combinations with replacement" % all_prob_combs.shape[0])

    probs = all_prob_combs[rng.choice(all_prob_combs.shape[0], replace=True, size=n_skills), :]
    
    # n_skills x 5
    return probs

def make_ytrue(skill_seqs, max_n_skills):
    """
        skill_seqs: BxT
    """
    
    ytrue = []
    mask = []
    for i in range(max_n_skills):
        ix = skill_seqs == i # BxT
        ytrue.append(ix[:, None, :])
        mask.append((ix.sum(1) > 0)[:,None]) # Bx1
    
    ytrue = th.concat(ytrue, dim=1) # BxKxT
    mask = th.concat(mask, dim=1) # BxK

    return ytrue.float(), mask.float()

class KCDiscoverer(nn.Module):

    def __init__(self, n_lstm_size):
        super().__init__()
        self.cell = nn.LSTM(1, 
            n_lstm_size, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True)
        self.ff = nn.Linear(n_lstm_size*2, 1)
        self.kc_ff = nn.Linear(n_lstm_size*2, 5)
    def forward(self, obs_seq):
        """
            Input: 
                obs_seq: BxT
            Output:
                membership_logits: BxT
                kc_param_logits: BxTx
        """
        
        cell_output, (hn, cn) = self.cell(obs_seq[:,:,None].float())
        cn = th.permute(cn, (1, 2, 0)) # BxHx2
        cn = th.reshape(cn, (cn.shape[0], cn.shape[1] * cn.shape[2])) # Bx2*H

        logits = self.ff(cell_output)[:,:,0] # BxT
        kc_param_logits = self.kc_ff(cn) # Bx5
        return logits , kc_param_logits

def min_filtering_auc_roc(pred_membership, obs_mask, ytrue, skill_mask):
    """
        pred_membership: B x T
        obs_mask: BxT
        ytrue: B x K x T
        skill_mask: BxK
    """
    result = np.zeros((ytrue.shape[0], ytrue.shape[1]))
    for k in range(ytrue.shape[1]):
        for b in range(ytrue.shape[0]):
            actual_seq = ytrue[b, k, :] # T 
            mem_seq = pred_membership[b, :] # T 
            mask = obs_mask[b,:].astype(bool) # T

            actual_seq = actual_seq[mask]
            mem_seq = mem_seq[mask]

            if np.unique(actual_seq).shape[0] > 1:
                result[b, k] = sklearn.metrics.balanced_accuracy_score(actual_seq, mem_seq > 0) * skill_mask[b, k]
            else:
                result[b, k] = 0.5
    return np.mean(np.max(result, axis=1))



if __name__ == "__main__":
    main()
