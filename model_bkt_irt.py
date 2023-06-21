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

import layer_fastbkt 
import layer_bkt 
import layer_seq_bayesian

def main():
    cfg_path = sys.argv[1]
    dataset_name = sys.argv[2]
    output_path = sys.argv[3]

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    splits = np.load("data/splits/%s.npy" % dataset_name)

    if not cfg['problem_effects']:
        df['problem'] = 0
    if cfg.get('single_kc', False):
        df['skill'] = 0
    if cfg.get('problems_as_skills', False):
        df['skill'] = df['problem']
    
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
    model = BktModel(cfg).to(cfg['device'])
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
            corr_seqs = [th.tensor(s['correct']) for s in batch_seqs]
            ytrue = pad_sequence(corr_seqs, batch_first=True, padding_value=0).float().to(cfg['device'])
            mask = pad_sequence(corr_seqs, batch_first=True, padding_value=-1).to(cfg['device']) > -1
            
            #tic = time.perf_counter()
            output = model(batch_seqs, ytrue) # OxMx2
            #toc = time.perf_counter()
            # print("Model call: %f secs" % (toc - tic))
            
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

class BktModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        #
        # BKT Parameters
        #
        self._dynamics_logits = nn.Embedding(cfg['n_kcs'], 3) # pL, pF, pI0
        self.obs_logits_problem = nn.Parameter(th.zeros(cfg['n_problems'], 2))
        self.obs_logits_kc = nn.Parameter(th.randn(cfg['n_kcs'], 2))
        
        if cfg['bkt_module'] == 'fbkt':
            bkt_module = layer_fastbkt.FastBkt(cfg['fastbkt_n'], cfg['device'])
        elif cfg['bkt_module'] == 'bkt':
            bkt_module = layer_bkt.RnnBkt()
        else:
            raise "Unknown BKT module"
        self._bkt_module = bkt_module

        self._device = cfg['device'] 

        self.ability_levels = th.tensor(cfg['ability_levels']).to(cfg['device']) # A
        self.ability_index = th.arange(self.ability_levels.shape[0]).long().to(cfg['device']) # A 

    def forward(self, seqs, ytrue):
        orig_batch_size = len(seqs)
        n_ability_levels = self.ability_levels.shape[0]

        # prepare the batch
        #tic = time.perf_counter()
        subseqs, max_len = utils.prepare_batch(seqs)
        n_new_bach_size = len(subseqs)
        #toc = time.perf_counter()
        #print("Batch prepared: %f secs" % (toc - tic))
        
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

        logpred, _ = layer_seq_bayesian.seq_bayesian(result, ytrue)

        return logpred


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

        logprob_pred = self._bkt_module(corr, dynamics_logits, obs_logits)
        return logprob_pred



if __name__ == "__main__":
    main()
