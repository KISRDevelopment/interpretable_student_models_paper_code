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
import loss_sequence 
import utils 

def main():
    import sys

    cfg_path = sys.argv[1]
    dataset_name = sys.argv[2]
    output_path = sys.argv[3]
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    if len(sys.argv) > 4:
        problem_feature_mat_path = sys.argv[4]
        cfg['problem_feature_mat_path'] = problem_feature_mat_path

    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    
    splits = np.load("data/splits/%s.npy" % dataset_name)
    

    # problems occuring less than x times should be assigned
    if cfg.get('min_problem_freq', 0) > 0:
        utils.trim_problems(df, cfg['min_problem_freq'])
    
    cfg['n_problems'] = np.max(df['problem']) + 1
    cfg['device'] = 'cuda:0'

    results_df = run(cfg, df, splits)
    
    results_df.to_csv(output_path)

def run(cfg, df, splits):
    
    problems_to_skills = dict(zip(df['problem'], df['skill']))
    n_skills = np.max(df['skill']) + 1
    ps = [problems_to_skills[p] for p in range(cfg['n_problems'])]
    A = np.zeros((cfg['n_problems'], n_skills))
    A[np.arange(A.shape[0]), ps] = 1
    
    cfg['ref_labels'] = A
    cfg['n_kcs'] = n_skills
    
    if 'problem_feature_mat_path' in cfg:
        problem_feature_mat = np.load(cfg['problem_feature_mat_path'])
        mu = np.mean(problem_feature_mat, axis=0, keepdims=True)
        std = np.std(problem_feature_mat, axis=0, ddof=1, keepdims=True)
        problem_feature_mat = (problem_feature_mat - mu) / (std+1e-9)
        cfg['problem_feature_mat'] = th.tensor(problem_feature_mat).float().to(cfg['device'])

    seqs = to_student_sequences(df)
    ref_assignment = get_problem_skill_assignment(df)

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

        train_problems = set(train_df['problem'])
        test_problems = set(test_df['problem'])
        print("# novel test problems: %d" % len(test_problems - train_problems))
        
        model = train(cfg, train_seqs, valid_seqs)
        ytrue_test, log_ypred_test = predict(cfg, model, test_seqs)
        
        ypred_test = np.exp(log_ypred_test)

        run_result = metrics.calculate_metrics(ytrue_test, ypred_test)
        rand_index = compare_kc_assignment(model, ref_assignment)
        run_result['rand_index'] = rand_index

        results.append(run_result)
        
        results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(len(results))])
        print(results_df)
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
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1).to(cfg['device']) > -1
            
            logpred, corr = model.forward(batch_obs_seqs, batch_problem_seqs) # BxSxTx2
            
            # BxSxT
            train_loss = -(corr * logpred[:, :, :, 1] + (1-corr) * logpred[:, :, :, 0])
            mask_ix = th.tile(batch_mask_seqs[:, None, :], (1, cfg['n_train_samples'], 1)).to(cfg['device'])
            
            # flatten
            train_loss = train_loss.flatten()
            mask_ix = mask_ix.flatten()
            train_loss = train_loss[mask_ix].mean() 

            aux_loss = 0
            if cfg['aux_loss_coeff'] > 0:
                mlogprobs = F.log_softmax(model.kc_discovery.get_logits(), dim=1)
                aux_loss = loss_sequence.nback_loss(batch_problem_seqs, batch_mask_seqs, mlogprobs, np.arange(cfg['min_lag'], cfg['max_lag']+1))
            
            train_loss = train_loss + cfg['aux_loss_coeff'] * aux_loss
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
    
    return model
    

def predict(cfg, model, seqs):

    model.eval()
    seqs = sorted(seqs, key=lambda s: len(s), reverse=True)

    with th.no_grad():

        # draw sample assignments
        As = [model.kc_discovery.sample_A(1e-6, True)[None,:,:] for i in range(cfg['n_test_samples'])]
        As = th.vstack(As) # SxPxK
        
        #print("Sampled")

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

            logpred = model.forward_test_multisample(batch_obs_seqs, batch_problem_seqs, As) # SxBxTx2
            logpred_correct = logpred[:, :, :, 1] # SxBxT

            #final_logpred_correct = th.logsumexp(logpred_correct, dim=0) - np.log(logpred_correct.shape[0]) # BxT
            final_logpred_correct = logpred_correct.mean(0)
            
            final_logpred_correct = final_logpred_correct.flatten()
            final_logpred_correct = final_logpred_correct[mask_ix]

            batch_logpred_correct.append(final_logpred_correct.cpu().numpy())

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
        if 'problem_feature_mat' in cfg:
            self.kc_discovery = layer_kc_discovery.FeaturizedKCDiscovery(cfg['problem_feature_mat'], cfg['n_kcs'])
        else:
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
        
        kc = th.concat(final_kc, dim=0) # S*BxTxK
        corr = th.tile(corr, (cfg['n_train_samples'], 1)) # S*B xT

        # call BKT
        logpred = self.hmm(corr, kc, trans_logits, obs_logits, init_logits) # B'xTx2

        # reshape to make samples available
        logpred = th.reshape(logpred, (-1, orig_batch_size, logpred.shape[1], logpred.shape[2])) # SxBxTx2
        corr = th.reshape(corr, (-1, orig_batch_size, corr.shape[1]))

        logpred = th.permute(logpred, (1, 0, 2, 3))
        corr = th.permute(corr, (1, 0, 2))

        return logpred, corr 
    
    def forward_test_multisample(self, corr, problem, As):
        """
            As: SxPxK
        """

        # put BKT parameters into the right format
        trans_logits, obs_logits, init_logits = get_logits(self._dynamics_logits, self._obs_logits)

        kc = th.reshape(As[:, problem, :], (As.shape[0] * corr.shape[0], corr.shape[1], As.shape[2])) # S*BxTxK
        tiled_corr = th.tile(corr, (As.shape[0], 1)) # B*S x T

        logpred = self.hmm(tiled_corr, kc, trans_logits, obs_logits, init_logits) # B*S x Tx2

        logpred = th.reshape(logpred, (As.shape[0], -1, corr.shape[1], 2)) # SxBxTx2

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

def compare_kc_assignment(model, ref_assignment):
    with th.no_grad():
        membership_logits = model.kc_discovery.get_logits().cpu().numpy()
    pred_assignment = np.argmax(membership_logits, axis=1)

    rand_index = sklearn.metrics.adjusted_rand_score(ref_assignment, pred_assignment)

    return rand_index

def get_problem_skill_assignment(df):

    problems_to_skills = dict(zip(df['problem'], df['skill']))
    n_problems = np.max(df['problem']) + 1
    return np.array([problems_to_skills[p] for p in range(n_problems)])
    

if __name__ == "__main__":
    main()
