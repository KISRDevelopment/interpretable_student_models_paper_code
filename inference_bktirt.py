import numpy as np
import metrics
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from typing import List, Tuple

from torch.nn.utils.rnn import pad_sequence
import utils 

class BktModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        #
        # BKT Parameters
        #
        self._dynamics_logits = nn.Embedding(cfg['n_kcs'], 3) # pL, pF, pI0
        
        self.obs_logits_problem_not_know = nn.Parameter(th.zeros(cfg['n_problems'])) 
        self.obs_logits_problem_boost = nn.Parameter(th.zeros(cfg['n_problems'])) 

        self.obs_logits_kc_not_know = nn.Parameter(th.randn(cfg['n_kcs']))
        self.obs_logits_kc_boost = nn.Parameter(th.randn(cfg['n_kcs']).exp())
        
        with th.no_grad():
            init_prototypes = th.randn(cfg['n_student_prototypes'], 4) # (Guessing, Not Slipping Boost, Learning, Not Forgetting)
            init_prototypes[:, 1] = init_prototypes[:, 1].exp()
        
        self.student_prototypes = nn.Parameter(init_prototypes) # Ax4 (Guessing, Not Slipping Boost, Learning, Not Forgetting)

        self.prototype_index = th.arange(cfg['n_student_prototypes']).long().to(cfg['device']) # A

        self._bkt_module = RnnBkt()

        self._device = cfg['device'] 

    def forward(self, seqs, ytrue):
        orig_batch_size = len(seqs)
        n_ability_levels = self.student_prototypes.shape[0]

        subseqs, max_len = utils.prepare_batch(seqs)
        
        #
        # pad all subsequences to identical lengths
        #

        # BxT
        padded_trial_id = self._bkt_module.pad([s['trial_id'] for s in subseqs], padding_value=-1).long().to(self._device)
        padded_problem = self._bkt_module.pad([s['problem'] for s in subseqs], padding_value=0).long().to(self._device)
        padded_correct = self._bkt_module.pad([s['correct'] for s in subseqs], padding_value=0).long().to(self._device)
        
        # B
        kc = th.tensor([s['kc'] for s in subseqs]).long().to(self._device)

        #
        # second stage: for each subsequence, handle all ability levels
        # the new batch size B' = B * number of ability levels
        #
        #   Sequence    1       2       3       ... 1       2       3   ...
        #   Ability     0       0       0       ... 1       1       1   ...
        #
        ability_index = th.repeat_interleave(self.prototype_index, kc.shape[0]) # B'
        ability_level = th.repeat_interleave(self.student_prototypes, kc.shape[0], dim=0) # B'x4
        
        padded_trial_id = th.tile(padded_trial_id, (n_ability_levels, 1)) # B'xT
        padded_problem = th.tile(padded_problem, (n_ability_levels, 1)) # B'xT
        padded_correct = th.tile(padded_correct, (n_ability_levels, 1)) # B'xT
        kc = th.tile(kc, (n_ability_levels,)) # B'

        #
        # run the model
        #
        logprob_pred, forward_logprobs, smoothed_logporbs = self.estimate_knowledge_state(padded_correct, kc, padded_problem, ability_level) # B'xTx2
        
        #
        # put everything back together
        #

        # B'*T
        adj_trial_id = padded_trial_id + ability_index[:, None] * orig_batch_size * max_len
        adj_trial_id[padded_trial_id == -1] = -1
        adj_trial_id = adj_trial_id.flatten() # B'*T
        mask_ix = adj_trial_id > -1
        valid_trial_id = adj_trial_id[mask_ix]
        

        # allocate storage for final result which will be in terms of the original student sequences
        forward_logprob0 = th.zeros(orig_batch_size*n_ability_levels*max_len).to(self._device)
        forward_logprob1 = th.zeros_like(forward_logprob0)
        forward_logprob0[valid_trial_id] = forward_logprobs[:,:,0].flatten()[mask_ix]
        forward_logprob1[valid_trial_id] = forward_logprobs[:,:,1].flatten()[mask_ix]
        forward_logprob0 = th.reshape(forward_logprob0, (n_ability_levels, orig_batch_size, max_len)) # AxBxM
        forward_logprob1 = th.reshape(forward_logprob1, (n_ability_levels, orig_batch_size, max_len)) # AxBxM
        forward_logprobs = th.concat((forward_logprob0[:,:,:,None], forward_logprob1[:,:,:,None]), dim=3) # AxBxMx2
        forward_logprobs = th.permute(forward_logprobs, (1, 0, 2, 3)) # BxAxTx2

        smoothed_logporbs0 = th.zeros(orig_batch_size*n_ability_levels*max_len).to(self._device)
        smoothed_logporbs1 = th.zeros_like(smoothed_logporbs0)
        smoothed_logporbs0[valid_trial_id] = smoothed_logporbs[:,:,0].flatten()[mask_ix]
        smoothed_logporbs1[valid_trial_id] = smoothed_logporbs[:,:,1].flatten()[mask_ix]
        smoothed_logporbs0 = th.reshape(smoothed_logporbs0, (n_ability_levels, orig_batch_size, max_len)) # AxBxM
        smoothed_logporbs1 = th.reshape(smoothed_logporbs1, (n_ability_levels, orig_batch_size, max_len)) # AxBxM
        smoothed_logporbs = th.concat((smoothed_logporbs0[:,:,:,None], smoothed_logporbs1[:,:,:,None]), dim=3) # AxBxMx2
        smoothed_logporbs = th.permute(smoothed_logporbs, (1, 0, 2, 3)) # BxAxTx2

        logprob_pred0 = th.zeros(orig_batch_size*n_ability_levels*max_len).to(self._device)
        logprob_pred1 = th.zeros_like(logprob_pred0)
        logprob_pred0[valid_trial_id] = logprob_pred[:,:,0].flatten()[mask_ix]
        logprob_pred1[valid_trial_id] = logprob_pred[:,:,1].flatten()[mask_ix]
        logprob_pred0 = th.reshape(logprob_pred0, (n_ability_levels, orig_batch_size, max_len)) # OxAxM
        logprob_pred1 = th.reshape(logprob_pred1, (n_ability_levels, orig_batch_size, max_len)) # OxAxM
        logprob_pred = th.concat((logprob_pred0[:,:,:,None], logprob_pred1[:,:,:,None]), dim=3) # OxAxMx2
        logprob_pred = th.permute(logprob_pred, (1, 0, 2, 3))

        # get posteriors over abilities
        logprob_ability_posterior = seq_bayesian(logprob_pred, ytrue) # BxAxT

        # normalize to get P(ability|y_1:t) BxAxT - Bx1xT = BxAxT
        logprob_ability_posterior = logprob_ability_posterior - th.logsumexp(logprob_ability_posterior, dim=1)[:,None,:]

        # marginalize over abilities
        forward_logprobs = th.logsumexp(forward_logprobs + logprob_ability_posterior[:,:,:, None], dim=1) # BxTx2
        smoothed_logprobs = th.logsumexp(smoothed_logporbs + logprob_ability_posterior[:, :, [-1], None], dim=1) # BxAxTx2 + BxAx1x1 = BxTx2

        """
            P(ability|y1:t)
            P(ht|y1:t)
            P(ht|y1:T)
        """
        return logprob_ability_posterior, forward_logprobs, smoothed_logprobs
    
    def estimate_knowledge_state(self, corr, kc, problem, ability_level):
        """
            Input:
                corr: trial correctness     BxT
                kc: kc membership (long)    B
                problem: problem ids (long) BxT
                ability_level: (float)      Bx4 (Guessing, Not Slipping, Learning, Not Forgetting)
            Returns:
                normed_forward_logprobs:    BxTx2 P(h_t|y1...yt)
                smoothed_logprobs:          BxTx2 P(y1:T|h_t)
        """
        dynamics_logits = self._dynamics_logits(kc) # Bx3

        obs_logits_problem_not_know = self.obs_logits_problem_not_know[problem] # BxT
        obs_logits_kc_not_know = self.obs_logits_kc_not_know[kc] # B
        obs_logits_problem_know = obs_logits_problem_not_know + F.relu(self.obs_logits_problem_boost[problem]) # BxT
        obs_logits_kc_know = obs_logits_kc_not_know + F.relu(self.obs_logits_kc_boost[kc]) # B 
        ability_level_know = ability_level[:, [0]] + F.relu(ability_level[:, [1]]) # Bx1

        # BxT
        obs_logits_guess = obs_logits_kc_not_know[:,None] + obs_logits_problem_not_know + ability_level[:, [0]]
        obs_logits_not_slip = obs_logits_problem_know + obs_logits_kc_know[:,None] + ability_level_know
        obs_logits = th.concat((obs_logits_guess[:,:,None], -obs_logits_not_slip[:,:,None]), dim=2) #BxTx2
        
        # adjust dynamics probabilities to account for student ability
        
        dynamics_logits[:, 0] = dynamics_logits[:, 0] + ability_level[:, 2] 
        dynamics_logits[:, 1] = dynamics_logits[:, 1] - ability_level[:, 3]

        logprob_pred, forward_logprobs = self._bkt_module(corr, dynamics_logits, obs_logits)
        normed_forward_logprobs = forward_logprobs - th.logsumexp(forward_logprobs, dim=2)[:,:, None] # BxTx2

        # compute smoothed probs
        smoothed_logprobs = self._bkt_module.smoother(corr, dynamics_logits, obs_logits)

        return logprob_pred, normed_forward_logprobs, smoothed_logprobs

def predict(cfg, model, seqs):

    max_seq_len = max([len(s['kc']) for s in seqs])
    n_seqs = len(seqs)

    all_ytrue = np.zeros((n_seqs, max_seq_len))
    all_problem = np.zeros((n_seqs, max_seq_len))
    all_kc = np.zeros((n_seqs, max_seq_len))
    
    all_logprob_ability_posterior = np.zeros((n_seqs, cfg['n_student_prototypes'], max_seq_len))
    all_forward_logprobs = np.zeros((n_seqs, max_seq_len, 2))
    all_smoothed_logprobs = np.zeros_like(all_forward_logprobs)

    cfg['n_test_batch_seqs'] = 10
    model.eval()
    with th.no_grad():

        for offset in range(0, len(seqs), cfg['n_test_batch_seqs']):
            end = offset + cfg['n_test_batch_seqs']
            batch_seqs = seqs[offset:end]
            print("%d - %d out of %d" % (offset, end, len(seqs)))

            # BxT
            ytrue = pad_sequence([th.tensor(s['correct']) for s in batch_seqs], batch_first=True, padding_value=0).float().to(cfg['device'])
            
            logprob_ability_posterior, forward_logprobs, smoothed_logprobs = model(batch_seqs, ytrue) # BxAxT, BxTx2, BxTx2
            T = forward_logprobs.shape[1]

            all_logprob_ability_posterior[offset:end, :, :T] = logprob_ability_posterior.cpu().numpy()
            all_forward_logprobs[offset:end, :T, :] = forward_logprobs.cpu().numpy()
            all_smoothed_logprobs[offset:end, :T, :] = smoothed_logprobs.cpu().numpy()

            all_ytrue[offset:end, :T] = ytrue.cpu().numpy()
            
            problem_seq = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0).numpy()
            kc_seq = pad_sequence([th.tensor(s['kc']) for s in batch_seqs], batch_first=True, padding_value=0).numpy()
            
            all_problem[offset:end, :forward_logprobs.shape[1]] = problem_seq
            all_kc[offset:end, :forward_logprobs.shape[1]] = kc_seq 

    return all_ytrue, all_problem, all_kc, all_logprob_ability_posterior, all_forward_logprobs, all_smoothed_logprobs


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


@jit.script
def seq_bayesian(logpred, ytrue):
    """
        Performs sequential Bayesian model averaging based on binary observations.
        Assumes there is B sequences of length T each and that
        there are A models predicting y_t given y_1..t-1.

        Input:
            logpred: BxAxTx2 representing p(y_t|y_1..t-1,alpha)
            ytrue: BxT
        Returns:
            logprob_posterior: BxAxT logP(alpha, y1...yt)
    """

    # logprobability of observations BxAxT
    logprob = ytrue[:,None,:] * logpred[:,:,:,1] + (1-ytrue[:,None,:]) * logpred[:,:,:,0]
        
    # calculate unnormed posterior over alphas
    # this represents p(alpha,y_1...yt-1)
    # p(alpha) ~ Uniform (i.e., for t=1)
    posteriors = logprob.cumsum(2)
    alpha_posterior = posteriors.roll(dims=2, shifts=1) # BxAxT
    alpha_posterior[:, :, 0] = 0.0 # uniform prior

    return posteriors+logprob # BxAxT
    

def main():

    """ quick test of implementation against reference """
    
    # obs = [0, 0, 1, 0, 0]
    
    # logit_pI0 = 0.0
    # logit_pG = np.log(0.1 / 0.9)
    # logit_pS = np.log(0.2 / 0.8)
    # logit_pL = np.log(0.3 / 0.7)
    # logit_pF = np.log(0.3 / 0.7)
    # corr = th.tensor([obs])
    
    # dynamics_logits = th.tensor([[logit_pL, logit_pF, logit_pI0]])
    
    # obs_logits = th.tensor([[logit_pG, logit_pS]]) # Bx2
    # obs_logits = th.tile(obs_logits, (1, corr.shape[1], 1)) # BxTx2
    # model = RnnBkt()
    # smoothed_state_logbprobs = model.smoother(corr, dynamics_logits, obs_logits)
    # print(smoothed_state_logbprobs.exp().numpy())

    #generate_smoothing_example()

    interpret()

def interpret():
    import json 
    import pandas as pd

    # cfg_path = "data/exp_model_comp_interpretable/fbkt-multdim-abilities_gervetetal_bridge_algebra06.json"
    # dataset_name = "gervetetal_bridge_algebra06"
    #state_dict_path = cfg_path.replace('.json', '.state_dicts')

    cfg_path = "data/exp_model_comp_interpretable/fbkt-multdim-abilities_gervetetal_bridge_algebra06.json"
    dataset_name = "gervetetal_algebra05"
    state_dict_path = "tmp/interpret_algebra05.state_dicts"
    
    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    cfg['n_kcs'] = np.max(df['skill']) + 1
    cfg['n_problems'] = np.max(df['problem']) + 1
    cfg['device'] = 'cuda:0'

    state_dicts = th.load(state_dict_path)

    split_id = 0

    model = BktModel(cfg)
    model.load_state_dict(state_dicts[split_id])
    model = model.to(cfg['device'])
    
    seqs = utils.to_seqs(df)

    # filter sequences
    seqs = [seqs[s] for s in seqs.keys() if len(seqs[s]['correct']) >= 100] 

    # calculate mean correctness
    seq_mu = np.array([np.mean(s['correct']) for s in seqs])

    # select examples
    seq_ids = select_examples(seq_mu, np.linspace(0.01, 0.99, 10))

    # sort by mean
    seq_ids = sorted(seq_ids, key=lambda sid: seq_mu[sid])
    
    seqs = [seqs[i] for i in seq_ids]
    seq_lens = [len(s['kc']) for s in seqs]
    seq_mu = seq_mu[seq_ids]
    print(seq_mu)
    
    all_ytrue, all_problem, all_kc, all_logprob_ability_posterior, all_forward_logprobs, all_smoothed_logprobs = predict(cfg, model, seqs)
    
    np.savez("tmp/exp_interpret_dynamics.npz", 
            seq_lens=[len(s['correct']) for s in seqs],
            ytrue=all_ytrue,
            seq_mu=seq_mu,
            logprob_ability_posterior=all_logprob_ability_posterior,
            forward_logprobs=all_forward_logprobs,
            smoothed_logprobs=all_smoothed_logprobs,
            student_prototypes=state_dicts[split_id]['student_prototypes'].cpu().numpy(),
            obs_logits_problem_not_know=model.obs_logits_problem_not_know.cpu().detach().numpy(),
            obs_logits_problem_boost=model.obs_logits_problem_boost.cpu().detach().numpy(),
            obs_logits_kc_not_know=model.obs_logits_kc_not_know.cpu().detach().numpy(),
            obs_logits_kc_boost=model.obs_logits_kc_boost.cpu().detach().numpy(),
            problem_seqs=all_problem,
            kc_seqs=all_kc,
            dynamics_logits=model._dynamics_logits.weight.cpu().detach().numpy())
    
def select_examples(seq_mu, thresholds):
    ix = np.argsort(seq_mu)
    n_seqs = seq_mu.shape[0]

    chosen_seq_ids = []
    for thres in thresholds:
        seq_id = ix[int(n_seqs * thres)]
        print("Threshold %0.2f , Sequence ID: %d, Mu: %0.2f" % (thres, seq_id, seq_mu[seq_id]))
        chosen_seq_ids.append(seq_id)
    
    return chosen_seq_ids

def generate_smoothing_example():
    import numpy.random as rng 
    import pandas as pd 

    # BKT parameters probabilities
    logit_pI0 = 0.
    logit_pG = np.log(0.2/0.8)
    logit_pS = np.log(0.2/0.8)
    logit_pL = np.log(0.1/0.9)
    logit_pF = np.log(0.05/0.95)

    # generate an example trial sequence
    pI0 = sigmoid(logit_pI0)
    pG = sigmoid(logit_pG)
    pS = sigmoid(logit_pS)
    pL = sigmoid(logit_pL)
    pF = sigmoid(logit_pF)
    seq = []
    states = []
    state = rng.binomial(1, pI0)
    for i in range(100):
        states.append(state)
        pC = (1-pS) if state == 1 else pG 

        ans = rng.binomial(1, pC)
        seq.append(ans)

        state = rng.binomial(1, (1-pF) if state == 1 else pL)

    # compute forward probabilities
    corr = th.tensor([seq]).long()
    model = RnnBkt()
    dynamics_logits = th.tensor([[logit_pL, logit_pF, logit_pI0]])
    obs_logits = th.tensor([[logit_pG, logit_pS]]) # Bx2
    obs_logits = th.tile(obs_logits, (1, corr.shape[1], 1)) # BxTx2
    _, forward_logprobs = model(corr, dynamics_logits, obs_logits)
    normed_forward_logprobs = forward_logprobs - th.logsumexp(forward_logprobs, dim=2)[:,:, None] # BxTx2

    # compute smoothed probs
    smoothed_logprobs = model.smoother(corr, dynamics_logits, obs_logits)

    with th.no_grad():
        output_df = pd.DataFrame({
            "state" : states, 
            "correct" : seq,
            "forward" : normed_forward_logprobs.exp()[0, :, 1].numpy(),
            "smoothed" : smoothed_logprobs.exp()[0, :, 1].numpy()
        })
    
    output_df.to_csv("tmp/smoothing_example.csv", index=False)
    print(output_df)
def sigmoid(x):
    return 1/(1+np.exp(-x))

if __name__ == "__main__":
    main()