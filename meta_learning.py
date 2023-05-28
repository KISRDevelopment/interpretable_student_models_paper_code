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
def main():

    #test_loss()

    train()

def train():
    cfg = {
        "n_lstm_size" : 10,
        "lr" : 0.1,
        "epochs" : 200,
        "n_batch_size" : 2,
        "n_train_seqs" : 80,
        "n_valid_seqs" : 20,
        "max_n_skills" : 20,
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

            batch_obs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0).float() # BxT
            
            obs_mask = (pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) >= 0).float() # BxT
            batch_skill_seqs = pad_sequence([th.tensor(s['skill']) for s in batch_seqs], batch_first=True, padding_value=-1)
            batch_ytrue, skill_mask = make_ytrue(batch_skill_seqs, cfg['max_n_skills']) # BxKxT

            membership_logits = model(batch_obs) # BxT
            membership_logits = membership_logits * obs_mask - 10 * (1-obs_mask) # padding trials will not be members

            train_loss = min_filtering_loss(membership_logits, obs_mask, batch_ytrue, skill_mask)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            losses.append(train_loss.item())
        
        with th.no_grad():
            valid_obs = pad_sequence([th.tensor(s['obs']) for s in valid_seqs], batch_first=True, padding_value=0).float() # BxT
            obs_mask = (pad_sequence([th.tensor(s['obs']) for s in valid_seqs], batch_first=True, padding_value=-1) >= 0).float() # BxT
            batch_skill_seqs = pad_sequence([th.tensor(s['skill']) for s in valid_seqs], batch_first=True, padding_value=-1)
            valid_ytrue, skill_mask = make_ytrue(batch_skill_seqs, cfg['max_n_skills']) # BxKxT

            membership_logits = model(valid_obs)
            
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



# def test_loss():

#     membership_logits = th.tensor([[-10, 10, 10, 10, -10]]).float() 
#     ytrue = th.tensor([
#         [
#             [1, 0, 0, 0, 1],
#             [0, 1, 1, 1, 0]
#         ]
#     ]).float()

#     print(min_filtering_loss(membership_logits, ytrue))


class KCDiscoverer(nn.Module):

    def __init__(self, n_lstm_size):
        super().__init__()
        self.cell = nn.LSTM(1, 
            n_lstm_size, 
            num_layers=3, 
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
        
        cell_output,_ = self.cell(obs_seq[:,:,None])
        logits = self.ff(cell_output)[:,:,0]
        #kc_param_logits = self.kc_ff(cell_output)
        return logits #, kc_param_logits
        
def min_filtering_loss(membership_logits, obs_mask, ytrue, skill_mask):
    """
        Input:
            membership_logits: BxT
            obs_mask: BxT
            ytrue: BxKxT
            skill_mask: BxKxT
            where K is the number of valid answers.
            If a skill has no assignment then it is not considered in the loss.
        Output:
            loss
    """
    membership_log_probs = F.logsigmoid(membership_logits) # BxT
    nonmembership_log_probs = F.logsigmoid(-membership_logits) # BxT
    
    # Bx1xT * BxKxT = BxKxT
    ll = membership_log_probs[:,None,:] * ytrue  + nonmembership_log_probs[:,None,:] * (1-ytrue)

    ll = obs_mask[:,None,:] * ll
    
    # BxK / Bx1 = BxK
    ll = ll.sum(2) / obs_mask.sum(1)[:,None]
    ll = ll * skill_mask + (1-skill_mask) * -10

    # B
    mfl,_ = th.max(ll, dim=1)
    
    return -mfl.mean()

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

            result[b, k] = sklearn.metrics.balanced_accuracy_score(actual_seq, mem_seq > 0) * skill_mask[b, k]
            
    return np.mean(np.max(result, axis=1))

# class MultiHmmCell(jit.ScriptModule):
    
#     def __init__(self):
#         super(MultiHmmCell, self).__init__()

#     @jit.script_method
#     def forward(self, obs: Tensor, chain: Tensor, 
#         trans_logits: Tensor, 
#         obs_logits: Tensor, 
#         init_logits: Tensor) -> Tuple[Tensor, Tensor]:

#         n_batch = obs.shape[0]
#         log_alpha = F.log_softmax(init_logits, dim=1) # n_chains x n_states
#         log_alpha = th.tile(log_alpha, (n_batch, 1, 1)) # batch x chains x states
#         return self.forward_given_alpha(obs, chain, trans_logits, obs_logits, init_logits, log_alpha)
    
#     @jit.script_method
#     def forward_given_alpha(self, obs: Tensor, chain: Tensor, 
#         trans_logits: Tensor, 
#         obs_logits: Tensor, 
#         init_logits: Tensor,
#         log_alpha: Tensor) -> Tuple[Tensor, Tensor]:
#         """
#             Input:
#                 obs: [n_batch, t]
#                 chain: [n_batch, t, n_chains]
#                 trans_logits: [n_chains, n_states, n_states] (Target, Source)
#                 obs_logits: [n_chains, n_states, n_outputs]
#                 init_logits: [n_chains, n_states]
#                 log_alpha: [n_batch, n_chains, n_states]
#             output:
#                 logits: [n_batch, t, n_outputs]
#                 log_alpha: [n_batch, n_chains, n_states]
#         """

#         n_chains, n_states, n_outputs = obs_logits.shape 

#         outputs = th.jit.annotate(List[Tensor], [])
        
#         n_batch, _ = obs.shape
#         batch_idx = th.arange(n_batch)

        
#         log_obs = F.log_softmax(obs_logits, dim=2) # n_chains x n_states x n_obs
#         log_t = F.log_softmax(trans_logits, dim=1) # n_chains x n_states x n_states
        
#         # B X C X S
#         for i in range(0, obs.shape[1]):
#             curr_chain = chain[:,i,:] # B X C
            
#             # predict
#             a1 = (curr_chain[:,:,None, None] * log_obs[None,:,:,:]).sum(1) # B X S X O
#             a2 = (curr_chain[:,:,None] * log_alpha).sum(1) # BXCX1 * BXCXS = BXS

#             # B X S X O + B X S X 1
#             log_py = th.logsumexp(a1 + a2[:,:,None], dim=1)  # B X O
            
#             log_py = log_py - th.logsumexp(log_py, dim=1)[:,None]
#             outputs += [log_py]

#             # update
#             curr_y = obs[:,i]
#             a1 = th.permute(log_obs[:,:,curr_y], (2, 0, 1)) # B X C X S
#             log_py = (a1 * curr_chain[:,:,None]).sum(1) # B X S
            

#             a1 = (log_alpha * curr_chain[:,:,None]).sum(1) # BxCxS * BxCx1 = BxS
#             a2 = (log_t[None,:,:,:] * curr_chain[:,:,None,None]).sum(1) # 1xCxSxS * BxCx1x1 = BxSxS
#             a3 = th.logsumexp(log_py[:,None,:] + a1[:,None,:] + a2, dim=2)

#             # B x 1 X S + B x 1 x S + B x S x S = B x S
#             log_alpha = (1 - curr_chain[:,:,None]) * log_alpha + curr_chain[:,:,None] * a3[:,None,:]
        
        
#         outputs = th.stack(outputs)
#         outputs = th.transpose(outputs, 0, 1)
        
#         return outputs, log_alpha
def cluster(model, obs, problem_seq, n_problems):

    """
        obs: BxT
        problem_seq: BxT

        For each sequence, first determine which problems belong to the same skill
        then aggregate all these problems.

        Remove the selected problems then
        iterate
    """

    assigned = th.zeros_like(obs).bool() # BxT

    clusters = []
    with th.no_grad():
        while True:

            membership_logits = model(obs) * (1-assigned) + assigned * -10 # BxT
            hard_membership = (membership_logits > 0.) # BxT
            assigned = assigned | hard_membership

            assigned_problems = problem_seq[hard_membership] 
            cluster = th.zeros(n_problems)
            cluster[assigned_problems] = 1
            clusters.append(cluster)
    
    Q = th.concat(clusters, dim=0)

if __name__ == "__main__":
    main()
