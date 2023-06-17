import numpy as np
import metrics
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from typing import List, Tuple
import pandas as pd 
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import copy 
import json
import time 
import sklearn.metrics
from scipy.stats import qmc

import layer_multihmmcell
import layer_kc_discovery
import early_stopping_rules
def main():
    import sys

    cfg_path = sys.argv[1]
    dataset_name = sys.argv[2]
    output_path = sys.argv[3]
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    
    splits = np.load("data/splits/%s.npy" % dataset_name)
    
    cfg['n_problems'] = np.max(df['problem']) + 1
    cfg['device'] = 'cuda:0'

    results_df = run(cfg, df, splits)

    results_df.to_csv(output_path)

def run(cfg, df, splits):
    
    # problems_to_skills = dict(zip(df['problem'], df['skill']))
    # n_problems = np.max(df['problem']) + 1
    # A = np.array([problems_to_skills[p] for p in range(n_problems)])
    # cfg['ref_labels'] = A
    
    print("# of problems: %d" % cfg['n_problems'])
    gdf = df.groupby('problem')['student'].count()
    lower = np.percentile(gdf, q=2.5)
    upper = np.percentile(gdf, q=97.5)
    print("95%% occurance range: %d-%d" % (lower,upper))
    print("# of problems occuring at least 10 times: %d" % np.sum(gdf >= 10))
    
    seqs = to_student_sequences(df)
    
    all_ytrue = []
    all_ypred = []

    results = []
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

        model, best_aux = train(cfg, train_seqs, valid_seqs)
        ytrue_test, log_ypred_test = predict(cfg, model, test_seqs)
        
        ypred_test = np.exp(log_ypred_test)

        run_result = metrics.calculate_metrics(ytrue_test, ypred_test)
        
        results.append(run_result)
        
    results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(splits.shape[0])])
    
    return results_df


def to_student_sequences(df):
    seqs = defaultdict(lambda: {
        "obs" : [],
        "problem" : []
    })
    for r in df.itertuples():
        seqs[r.student]["obs"].append(r.correct)
        seqs[r.student]["problem"].append(r.problem)
    return seqs

def train(cfg, train_seqs, valid_seqs):

    model = BktModel(cfg).to(cfg['device'])
    
    optimizer = th.optim.NAdam(model.parameters(), lr=cfg['learning_rate'])
    
    stopping_rule = early_stopping_rules.PatienceRule(cfg['es_patience'], cfg['es_thres'], minimize=False)

    n_seqs = len(train_seqs) if cfg['full_epochs'] else cfg['n_train_batch_seqs']

    best_state = None 

    for e in range(cfg['epochs']):
        np.random.shuffle(train_seqs)
        losses = []

        for offset in range(0, n_seqs, cfg['n_train_batch_seqs']):
            end = offset + cfg['n_train_batch_seqs']
            batch_seqs = train_seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_problem_seqs = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1
            
            logpred, corr = model.forward(batch_obs_seqs, batch_problem_seqs) # BxSxTx2
            
            # BxSxT
            train_loss = -(corr * logpred[:, :, :, 1] + (1-corr) * logpred[:, :, :, 0])
            mask_ix = th.tile(batch_mask_seqs[:, None, :], (1, cfg['n_train_samples'], 1)).to(cfg['device'])
            
            # flatten
            train_loss = train_loss.flatten()
            mask_ix = mask_ix.flatten()
            train_loss = train_loss[mask_ix].mean() 

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            
            losses.append(train_loss.item())
        
        mean_train_loss = np.mean(losses)

        #
        # Validation
        #
        ytrue, ypred = predict(cfg, model, valid_seqs)
        
        auc_roc = metrics.calculate_metrics(ytrue, ypred)['auc_roc']
        
        stop_training, new_best = stopping_rule.log(auc_roc)

        print("%4d Train loss: %8.4f, Valid AUC: %0.2f %s" % (e, mean_train_loss, auc_roc, '***' if new_best else ''))
        
        if new_best:
            best_state = copy.deepcopy(model.state_dict())
        
        if stop_training:
            break

    model.load_state_dict(best_state)
    exit()
    return model, best_aux
    

def predict(cfg, model, seqs):

    model.eval()
    seqs = sorted(seqs, key=lambda s: len(s), reverse=True)

    with th.no_grad():

        # draw sample assignments
        As = [model.kc_discovery.sample_A(1e-6, True) for i in range(cfg['n_test_samples'])]

        # go over batches
        batch_logpred_correct = []
        batch_corr = []
        for offset in range(0, len(seqs), cfg['n_test_batch_seqs']):
            end = offset + cfg['n_test_batch_seqs']
            batch_seqs = seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_problem_seqs = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1).to(cfg['device']) > -1

            mask_ix = batch_mask_seqs.flatten()

            # call the model for each A 
            all_logpred_correct = []
            for A in As:
                logpred = model.forward_test(batch_obs_seqs, batch_problem_seqs, A) # BxTx2
                logpred_correct = logpred[:,:,1].flatten() # B*T
                logpred_correct = logpred_correct[mask_ix]
                all_logpred_correct.append(logpred_correct.cpu())
            
            # average preds
            final_logpred_correct = th.vstack(all_logpred_correct) # SxNb
            final_logpred_correct = th.logsumexp(final_logpred_correct, dim=0) - np.log(final_logpred_correct.shape[0]) # Nb
            batch_logpred_correct.append(final_logpred_correct.numpy())

            # get actual observations
            corr = batch_obs_seqs.flatten() 
            corr = corr[mask_ix] # Nb
            batch_corr.append(corr.cpu().numpy())
        
        # concatenate everything
        logpred = np.hstack(batch_logpred_correct)
        ytrue = np.hstack(batch_corr)
    
    
    return ytrue, logpred


class BktModel(nn.Module):
    def __init__(self, cfg):
        super(BktModel, self).__init__()
        
        #
        # BKT Parameters
        #
        self._dynamics_logits = nn.Parameter(th.randn(cfg['n_kcs'], 3)) # pL, pF, pI0
        self._obs_logits = nn.Parameter(th.randn(cfg['n_kcs'], 2)) # pG, pS

        # KC Discovery
        self.kc_discovery = layer_kc_discovery.SimpleKCDiscovery(cfg['n_problems'], cfg['n_kcs'])

        #
        # BKT Module
        #
        self.hmm = layer_multihmmcell.MultiHmmCell()

        self.cfg = cfg
    
    def forward(self, corr, problem):
        """
            Input:
                corr: BxT
                problem: BxT (long)
            Output:
                logpred: BxSxTx2
        """
        cfg = self.cfg 

        orig_batch_size = corr.shape[0]

        # 
        # put BKT parameters into the right format
        #
        trans_logits, obs_logits, init_logits = get_logits(self._dynamics_logits, self._obs_logits)

        #
        # sample assignments
        #
        final_kc = []
        for i in range(cfg['n_train_samples']):
            # PxK
            A = self.kc_discovery.sample_A(cfg['tau'], cfg['hard_train_samples'])
            # compute effective KC based on assignemts
            kc = A[problem,:] # BxTxK
            final_kc.append(kc)
        
        kc = th.concat(final_kc, dim=0) # B*S xTxK
        corr = th.tile(corr, (cfg['n_train_samples'], 1)) # B*S xT

        # call BKT
        logpred = self.hmm(corr, kc, trans_logits, obs_logits, init_logits) # B'xTx2

        # reshape to make samples available
        logpred = th.reshape(logpred, (orig_batch_size, -1, logpred.shape[1], logpred.shape[2])) # BxSxTx2
        corr = th.reshape(corr, (orig_batch_size, -1, corr.shape[1]))
        return logpred, corr 

    
    def forward_test(self, corr, problem, A):
        # put BKT parameters into the right format
        trans_logits, obs_logits, init_logits = get_logits(self._dynamics_logits, self._obs_logits)

        # compute effective KC based on assignemts
        kc = A[problem,:] # BxTxK
        
        # call BKT
        logpred = self.hmm(corr, kc, trans_logits, obs_logits, init_logits) # BxTx2

        return logpred 
        
@jit.script
def get_logits(dynamics_logits, obs_logits):
    
    trans_logits = th.hstack((  dynamics_logits[:, [0]]*0, # 1-pL
                                dynamics_logits[:, [1]],  # pF
                                dynamics_logits[:, [0]],  # pL
                                dynamics_logits[:, [1]]*0)).reshape((-1, 2, 2)) # 1-pF (Latent KCs x 2 x 2)
    obs_logits = th.concat((obs_logits[:, [0]]*0, # 1-pG
                            obs_logits[:, [0]],  # pG
                            obs_logits[:, [1]],  # pS
                            obs_logits[:, [1]]*0), dim=1).reshape((-1, 2, 2)) # 1-pS (Latent KCs x 2 x 2)
    init_logits = th.hstack((dynamics_logits[:, [2]]*0, 
                             dynamics_logits[:, [2]])) # (Latent KCs x 2)

    return trans_logits, obs_logits, init_logits

if __name__ == "__main__":
    main()
