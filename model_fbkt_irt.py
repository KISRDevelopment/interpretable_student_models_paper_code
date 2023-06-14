import numpy as np 
import numpy.random

import torch as th 
import torch.nn.functional as F 
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence
import torch.jit as jit
from torch import Tensor
from typing import List

import itertools
import pandas as pd 
import sys 
import json 
from collections import defaultdict
import metrics 
import copy 

import early_stopping_rules
import time 

import utils 
import sklearn.metrics

def main():
    cfg_path = sys.argv[1]
    dataset_name = sys.argv[2]
    output_path = sys.argv[3]

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    splits = np.load("data/splits/%s.npy" % dataset_name)
    
    cfg['n_kcs'] = np.max(df['skill']) + 1
    cfg['n_problems'] = np.max(df['problem']) + 1
    cfg['device'] = 'cuda:0'

    results_df, all_params = run(cfg, df, splits) 
    results_df.to_csv(output_path)

    param_output_path = output_path.replace(".csv", ".params.npy")
    np.savez(param_output_path, **all_params)

def run(cfg, df, splits):

    lens = df.groupby('student')['problem'].count()
    print("Min, median, max sequence length: ", (np.min(lens), np.median(lens), np.max(lens)))
    
    seqs = utils.to_seqs(df)
    

    results = []
    all_params = defaultdict(list)

    for s in range(splits.shape[0]):
        split = splits[s, :]

        train_ix = split == 2
        valid_ix = split == 1
        test_ix = split == 0

        train_df = df[train_ix]
        valid_df = df[valid_ix]
        test_df = df[test_ix]

        train_students = set(train_df['student'])
        valid_students = set(valid_df['student'])
        test_students = set(test_df['student'])

        train_seqs = [seqs[s] for s in train_students]
        valid_seqs = [seqs[s] for s in valid_students]
        test_seqs = [seqs[s] for s in test_students]
        
        # Train & Test
        tic = time.perf_counter()
        model = train(cfg, train_seqs, valid_seqs)
        ytrue, log_ypred_correct = predict(cfg, model, test_seqs)
        toc = time.perf_counter()

        ypred_correct = np.exp(log_ypred_correct)

        run_result = metrics.calculate_metrics(ytrue, ypred_correct)
        run_result['time_diff_sec'] = toc - tic 

        results.append(run_result)

        with th.no_grad():
            all_params['dynamics_logits'].append(model._dynamics_logits.weight.cpu().numpy())
            all_params['obs_logits_kc'].append(model.obs_logits_kc.cpu().numpy())
            all_params['obs_logits_problem'].append(model.obs_logits_problem.cpu().numpy())
            
            
        print(run_result)
        
    results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(splits.shape[0])])
    print(results_df)
    
    return results_df, all_params

def train(cfg, train_seqs, valid_seqs):
    
    # tic = time.perf_counter()
    model = BktModel(cfg['n_kcs'], cfg['n_problems'], cfg['fastbkt_n'], np.linspace(-3, 3, 11), cfg['device']).to(cfg['device'])
    # toc = time.perf_counter()
    #print("Model creation: %f" % (toc - tic))

    optimizer = th.optim.NAdam(model.parameters(), lr=cfg['learning_rate'])
    
    stopping_rule = early_stopping_rules.PatienceRule(cfg['es_patience'], cfg['es_thres'], minimize=False)


    n_seqs = len(train_seqs) if cfg['full_epochs'] else cfg['n_train_batch_seqs']

    best_state = None

    #tic_global = time.perf_counter()
    for e in range(cfg['epochs']):
        np.random.shuffle(train_seqs)
        losses = []

        # tic = time.perf_counter()
        for offset in range(0, n_seqs, cfg['n_train_batch_seqs']):
            end = offset + cfg['n_train_batch_seqs']
            batch_seqs = train_seqs[offset:end]
            
            # OxM
            ytrue = pad_sequence([th.tensor(s['correct']) for s in batch_seqs], batch_first=True, padding_value=0).float().to(cfg['device'])
            mask = pad_sequence([th.tensor(s['correct']) for s in batch_seqs],batch_first=True, padding_value=-1).to(cfg['device']) > -1
            
            output = model(batch_seqs, ytrue) # OxMx2

            train_loss = -(ytrue * output[:, :, 1] + (1-ytrue) * output[:, :, 0]).flatten()
            mask_ix = mask.flatten()

            train_loss = train_loss[mask_ix].mean()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            losses.append(train_loss.item())
        # toc = time.perf_counter()
        #print("Train time: %f" % (toc - tic))
        mean_train_loss = np.mean(losses)
        
        #
        # Validation
        #
        # tic = time.perf_counter()   
        ytrue, ypred = predict(cfg, model, valid_seqs)
        # toc = time.perf_counter()
        #print("Predict time: %f" % (toc - tic))

        # print("Evaluation:")
        # print(ytrue.shape, ytrue.dtype)
        # print(ypred.shape, ypred.dtype)
        # tic = time.perf_counter()
        auc_roc = sklearn.metrics.roc_auc_score(ytrue, ypred)
        # toc = time.perf_counter()
        #print("Evaluation time: %f" % (toc - tic))
        stop_training, new_best = stopping_rule.log(auc_roc)

        print("%4d Train loss: %8.4f, Valid AUC: %0.2f %s" % (e, mean_train_loss, auc_roc, '***' if new_best else ''))
        
        if new_best:
            best_state = copy.deepcopy(model.state_dict())
        
        if stop_training:
            break
    # toc_global = time.perf_counter()
    # print("Total train time: %f" % (toc_global - tic_global))

    model.load_state_dict(best_state)
    return model

        
def predict(cfg, model, seqs):

    seqs = sorted(seqs, reverse=True, key=lambda s: len(s['kc']))
    
    model.eval()
    with th.no_grad():
        all_ypred = []
        all_ytrue = []
        for offset in range(0, len(seqs), cfg['n_test_batch_seqs']):
            end = offset + cfg['n_test_batch_seqs']
            batch_seqs = seqs[offset:end]
            
            # OxM
            ytrue = pad_sequence([th.tensor(s['correct']) for s in batch_seqs], batch_first=True, padding_value=0).float().to(cfg['device'])
            mask = pad_sequence([th.tensor(s['correct']) for s in batch_seqs],batch_first=True, padding_value=-1).to(cfg['device']) > -1
            
            output = model(batch_seqs, ytrue) # OxMx2
            
            ypred = output[:, :, 1].flatten()
            ytrue = ytrue.flatten()
            mask_ix = mask.flatten()
                
            ypred = ypred[mask_ix].cpu().numpy()
            ytrue = ytrue[mask_ix].cpu().int().numpy()

            all_ypred.append(ypred)
            all_ytrue.append(ytrue)
            
        ypred = np.hstack(all_ypred)
        ytrue = np.hstack(all_ytrue)
    model.train()
    
    return ytrue, ypred

def to_student_sequences(df):
    seqs = defaultdict(lambda: defaultdict(list))
    for r in df.itertuples():
        seqs[r.student][r.skill].append((r.problem, r.correct))
    return seqs

def split_seqs_by_kc(seqs):
    obs_seqs = []
    problem_seqs = []

    for by_student in seqs:
        for kc, seq in by_student.items():
            obs_seqs.append((kc, [e[1] for e in seq]))
            problem_seqs.append([e[0] for e in seq])
    return obs_seqs, problem_seqs

def pad_to_multiple(seqs, multiple, padding_value):
    return th.tensor(utils.pad_to_multiple(seqs, multiple, padding_value))

class BktModel(nn.Module):

    def __init__(self, n_kcs, n_problems, fastbkt_n, ability_levels, device):
        super(BktModel, self).__init__()
        
        self._dynamics_logits = nn.Embedding(n_kcs, 3) # pL, pF, pI0

        # problem logits are initialized to zero to help with generalization
        # by relying only on kc logits pG, pS
        self.obs_logits_problem = nn.Parameter(th.zeros(n_problems, 2))
        self.obs_logits_kc = nn.Parameter(th.randn(n_kcs, 2))
        
        self._model = FastBkt(fastbkt_n, device)
        self._device = device 

        self.block_size = 4 * fastbkt_n

        self.fastbkt_n = fastbkt_n

        self.ability_levels = th.tensor(ability_levels).to(device) # A
        self.ability_index = th.arange(len(ability_levels)).long().to(device) # A 

    def forward(self, seqs, ytrue):
        orig_batch_size = len(seqs)
        n_ability_levels = self.ability_levels.shape[0]

        # prepare the batch
        subseqs, max_len = utils.prepare_batch(seqs)
        n_new_bach_size = len(subseqs)

        #
        # pad all subsequences to identical lengths
        #

        # BxT
        padded_trial_id = pad_to_multiple([s['trial_id'] for s in subseqs], multiple=self.fastbkt_n, padding_value=-1).long().to(self._device)
        padded_problem = pad_to_multiple([s['problem'] for s in subseqs], multiple=self.fastbkt_n, padding_value=0).long().to(self._device)
        padded_correct = pad_to_multiple([s['correct'] for s in subseqs], multiple=self.fastbkt_n, padding_value=0).float().to(self._device)
        
        # B
        kc = th.tensor([s['kc'] for s in subseqs]).long().to(self._device)

        #
        # second stage: for each subsequence, handle all ability levels
        # the new batch size B' = B * number of ability levels
        #
        #   Sequence    1       2       3       ... 1       2       3   ...
        #   Ability     0       0       0       ... 1       1       1   ...
        #
        ability_index = th.repeat_interleave(self.ability_index, kc.shape[0]) # B'
        ability_level = th.repeat_interleave(self.ability_levels, kc.shape[0]) # B'
        padded_trial_id = th.tile(padded_trial_id, (n_ability_levels, 1)) # B'xT
        padded_problem = th.tile(padded_problem, (n_ability_levels, 1)) # B'xT
        padded_correct = th.tile(padded_correct, (n_ability_levels, 1)) # B'xT
        kc = th.tile(kc, (n_ability_levels,)) # B'

        #
        # run the model
        #
        logprob_pred = self.forward_(padded_correct, kc, padded_problem, ability_level) # B'xTx2

        #
        # put everything back together
        #

        # allocate storage for final result which will be in terms of the original
        # student sequences
        logprob_pred0 = th.zeros(orig_batch_size*n_ability_levels*max_len).to(self._device)
        logprob_pred1 = th.zeros_like(logprob_pred0)

        # B'*T
        adj_trial_id = padded_trial_id + ability_index[:, None] * orig_batch_size * max_len
        adj_trial_id[padded_trial_id == -1] = -1
        adj_trial_id = adj_trial_id.flatten() # B'*T
        mask_ix = adj_trial_id > -1
        valid_trial_id = adj_trial_id[mask_ix]
        
        logprob_pred0[valid_trial_id] = logprob_pred[:,:,0].flatten()[mask_ix]
        logprob_pred1[valid_trial_id] = logprob_pred[:,:,1].flatten()[mask_ix]
        
        # rearrange final result into the right shape
        logprob_pred0 = th.reshape(logprob_pred0, (n_ability_levels, orig_batch_size, max_len)) # OxAxM
        logprob_pred1 = th.reshape(logprob_pred1, (n_ability_levels, orig_batch_size, max_len)) # OxAxM
        result = th.concat((logprob_pred0[:,:,:,None], logprob_pred1[:,:,:,None]), dim=3) # OxAxMx2
        result = th.permute(result, (1, 0, 2, 3))

        return collapse_over_abilities(result, ytrue)



    def forward_(self, corr, kc, problem, ability_level):
        """
            Input:
                corr: trial correctness     BxT
                kc: kc membership (long)    B
                problem: problem ids (long) BxT
                ability_level: (float)      B
            Returns:
                logprob_pred: log probability of correctness BxTx2
        """
        dynamics_logits = self._dynamics_logits(kc) # Bx3

        obs_logits_kc = self.obs_logits_kc[kc, :] # Bx2
        obs_logits_problem = self.obs_logits_problem[problem, :] # BxTx2
        obs_logits = obs_logits_kc[:,None,:] + obs_logits_problem # BxTx2

        # adjust observation probabilities to account for student ability
        obs_logits[:, :, 0] = ability_level[:, None] + obs_logits[:, :, 0] # greater ability -> greater prob of guessing
        obs_logits[:, :, 1] = obs_logits[:, :, 1] - ability_level[:, None] # greater ability -> smaller prob of slipping

        logprob_pred, logprob_h = self._model(corr, dynamics_logits, obs_logits)
        return logprob_pred
        # batching over sequence
        # all_outputs = []
        # for offset in range(0, corr.shape[1], self.block_size):
        #     end = offset + self.block_size
        #     corr_slice = corr[:, offset:end]
        #     obs_logits_slice = obs_logits[:, offset:end]

        #     end = offset + self.block_size
        #     if offset == 0:
        #         logprob_pred, logprob_h = self._model(corr_slice, dynamics_logits, obs_logits_slice)
        #     else:
                
        #         logprob_h = logprob_h.detach()
        #         logprob_pred, logprob_h = self._model.forward_(corr_slice, dynamics_logits, obs_logits_slice, logprob_h)

        #     all_outputs.append(logprob_pred)

        # final_output = th.concat(all_outputs, dim=1) # BxTx2
        
        # return final_output
    
def collapse_over_abilities(logpred, ytrue):
    """
        Input:
            logpred: BxAxTx2 representing p(y_t|y_1..t-1,alpha)
            ytrue: BxT
        Returns:
            logpreds: BxTx2
    """

    # logprobability of observations BxAxT
    logprob = ytrue[:,None,:] * logpred[:,:,:,1] + (1-ytrue[:,None,:]) * logpred[:,:,:,0]
        
    # calculate unnormed posterior over alphas
    # this represents p(alpha,y_1...yt-1)
    # p(alpha) ~ Uniform (i.e., for t=1)
    alpha_posterior = logprob.cumsum(2).roll(dims=2, shifts=1) # BxAxT
    alpha_posterior[:, :, 0] = 0.0 # uniform prior

    # compute unnormed predictions
    unnormed_preds = logpred + alpha_posterior[:,:,:,None] # BxAxTx2
    unnormed_preds = th.logsumexp(unnormed_preds, dim=1) # BxTx2

    # normalize BxTx2
    normed_preds = unnormed_preds - th.logsumexp(unnormed_preds, dim=2)[:,:,None]

    return normed_preds

class FastBkt(jit.ScriptModule):

    def __init__(self, n, device):
        super(FastBkt, self).__init__()
        
        self.n = n 
        
        # all possible trajectories of length n
        trajectories = make_trajectories(n)

        # transition indecies to compute the probaiblity of a trajectory
        # 0: -pL, 1: pL, 2: pF, 3: 1-pF, 4: 1-pI0, 5: pI0 
        trans_ind = make_transition_indices(trajectories)

        # transition indecies to compute the posterior predictive over
        # the hidden state
        pred_ind = make_predictive_indices(trajectories)

        # move to torch
        self._trajectories = th.tensor(trajectories).float().to(device)
        self._trans_ind = th.tensor(trans_ind).long().to(device)
        self._pred_ind = th.tensor(pred_ind).long().to(device)
        self._device = device 

    def forward(self, corr: Tensor, dynamics_logits: Tensor, obs_logits: Tensor) -> Tensor:
        
        # initial and state transition dynamics (B)
        logit_pL, logit_pF, logit_pI0 = dynamics_logits[:, 0], dynamics_logits[:, 1], dynamics_logits[:, 2]

        # transition probabilities that will be indexed into
        trans_logits = th.vstack((-logit_pL, logit_pL, logit_pF, -logit_pF, -logit_pI0, logit_pI0)).T # Bx5
        trans_logprobs = F.logsigmoid(trans_logits) # Bx5

        #
        # probability of each trajectory
        #

        # Bx2**Nx1 * Bx1x1
        initial_logprobs = self._trajectories[None,:, [0]] * trans_logprobs[:,None,[5]] + \
            (1-self._trajectories[None, :, [0]]) * trans_logprobs[:,None,[4]] # Bx2**Nx1
        
        # Bx2**Nx1
        logprob_h = trans_logprobs[:,self._trans_ind].sum(2, keepdims=True) + initial_logprobs
        
        return self.forward_(corr, dynamics_logits, obs_logits, logprob_h)

    def forward_(self, corr: Tensor, dynamics_logits: Tensor, obs_logits: Tensor, logprob_h: Tensor) -> Tensor:
        """
            Input: 
                corr: trial sequence BxT
                dynamics_logits: BKT parameter logits (pL, pF, pI0) Bx3
                obs_logits: per-timestep logits (pG, pS) BxTx2
            Output:
                output_logprobs BxTx2

            Let 2**N = V
        """

        # initial and state transition dynamics (B)
        logit_pL, logit_pF, logit_pI0 = dynamics_logits[:, 0], dynamics_logits[:, 1], dynamics_logits[:, 2]

        # observation probabilities (BxT)
        logit_pG, logit_pS = obs_logits[:, :, 0], obs_logits[:, :, 1]

        # transition probabilities that will be indexed into
        trans_logits = th.vstack((-logit_pL, logit_pL, logit_pF, -logit_pF, -logit_pI0, logit_pI0)).T # Bx5
        trans_logprobs = F.logsigmoid(trans_logits) # Bx5

        # iterate over time 
        pred_corrects = th.jit.annotate(List[Tensor], [])
        pred_incorrects = th.jit.annotate(List[Tensor], [])
        for i in range(0, corr.shape[1], self.n):
            from_index = i
            to_index = from_index + self.n 

            # grab slice BxN
            corr_slice = corr[:, from_index:to_index]

            # grab slice of guess and slip probabilities BxN
            logit_pG_slice = logit_pG[:, from_index:to_index]
            logit_pS_slice = logit_pS[:, from_index:to_index]

            # probability of answering correctly Bx2**NxN
            # 1x2**NxN * Bx1xN
            obs_logits_correct = self._trajectories[None,:,:] * (-logit_pS_slice[:,None,:]) + (1-self._trajectories[None,:,:]) * logit_pG_slice[:,None,:]
            obs_logprob_correct = F.logsigmoid(obs_logits_correct) # Bx2**NxN

            # probability of answering incorrectly Bx2**NxN
            # print(logit_pS.shape)
            # print(self._trajectories.shape)
            obs_logits_incorrect = self._trajectories[None,:,:] * (logit_pS_slice[:,None,:]) + (1-self._trajectories[None,:,:]) * (-logit_pG_slice[:,None,:])
            obs_logprob_incorrect = F.logsigmoid(obs_logits_incorrect) # Bx2**NxN


            # likelikelihood of observing each observation under each trajectory Bx2**NxN
            # corr_slice[:,None,:]          Bx1xN
            # obs_logprob_correct           Bx2**NxN
            obs_loglik_given_h = corr_slice[:,None,:] * obs_logprob_correct + \
                (1-corr_slice[:,None,:]) * obs_logprob_incorrect

            # running likelihood of prior observations Bx2**NxN
            past_loglik_given_h = obs_loglik_given_h.cumsum(2).roll(dims=2, shifts=1)
            past_loglik_given_h[:,:,0] = 0.0

            # probability of correct/incorrect for each trajectory (weighted by trajectory weight)
            # Bx2**NxN + Bx2**NxN + Bx2**Nx1 = Bx2**NxN
            pred_correct_given_h = past_loglik_given_h + obs_logprob_correct + logprob_h
            pred_incorrect_given_h = past_loglik_given_h + obs_logprob_incorrect + logprob_h

            # unnormed probabilities of correctness and incorrectness BxN
            pred_correct = th.logsumexp(pred_correct_given_h, dim=1)
            pred_incorrect = th.logsumexp(pred_incorrect_given_h, dim=1)
            pred_corrects.append(pred_correct)
            pred_incorrects.append(pred_incorrect)

            #
            # new state prior for next iteration
            #
            seq_loglik_given_h = obs_loglik_given_h.sum(2) # Bx2**N
            seq_loglik = seq_loglik_given_h + logprob_h[:,:,0] # Bx2**N
            next_h_one_logprob = F.logsigmoid(trans_logits[:, self._pred_ind]) + seq_loglik # Bx2**N
            next_h_zero_logprob = F.logsigmoid(-trans_logits[:, self._pred_ind]) + seq_loglik  # Bx2**N
            next_h_one_logprob = th.logsumexp(next_h_one_logprob, dim=1, keepdims=True) # Bx1
            next_h_zero_logprob = th.logsumexp(next_h_zero_logprob, dim=1, keepdims=True) # Bx1
            
            # 1x2**N * Bx1 = Bx2**N
            initial_logprobs = self._trajectories[None,:,0] * next_h_one_logprob + (1-self._trajectories[None,:,0]) * next_h_zero_logprob
            
            # Bx2**Nx1 + Bx2**Nx1 = Bx2**Nx1
            logprob_h = trans_logprobs[:,self._trans_ind].sum(2, keepdims=True) + initial_logprobs[:,:,None]

                
        # BxT
        pred_corrects = th.concat(pred_corrects, dim=1)
        pred_incorrects = th.concat(pred_incorrects, dim=1)

        # BxTx2
        preds = th.concat((pred_incorrects[:,:,None], pred_corrects[:,:,None]), dim=2)
        
        # BxTx2 - BxTx1
        logprob_next = preds - th.logsumexp(preds, dim=2, keepdims=True)

        # BxTx2
        return logprob_next, logprob_h

def make_trajectories(n):
    """
        constructs all possible trajectories of binary state of
        a sequence of length n.
        Returns a matrix of size 2^n x n
    """
    trajectories = np.array(list(itertools.product(*[[0, 1]]*n)))
    
    return trajectories

def make_transition_indices(trajectories):
    """
        computes the transition indices
        Returns a matrix of size 2^n x n-1
        because it excludes the first trial.
    """

    convolved = np.zeros_like(trajectories)
    for i in range(trajectories.shape[0]):
        indices = np.convolve(trajectories[i,:], [1, 2], mode='same')
        convolved[i, :] = indices
    convolved = convolved[:,1:]

    return convolved

def make_predictive_indices(trajectories):
    """
        computes the indices to predict the transition from
        the last state.
    """
    target_state = np.ones(trajectories.shape[0])
    indices = trajectories[:,-1] * 2 + target_state
    return indices

if __name__ == "__main__":
    main()
