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
import csbkt
import sklearn.cluster
import loss_sequence 
import joint_pmf
import early_stopping_rules

def main():
    import sys

    cfg_path = sys.argv[1]
    dataset_name = sys.argv[2]
    output_path = sys.argv[3]
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    cfg['device'] = 'cuda:0'
    
    if len(sys.argv) > 4:
        problem_feature_mat_path = sys.argv[4]
        cfg['problem_feature_mat_path'] = problem_feature_mat_path

    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    
    splits = np.load("data/splits/%s.npy" % dataset_name)
    
    results_df = run(cfg, df, splits)
    print(results_df)

    results_df.to_csv(output_path)

    # param_output_path = output_path.replace(".csv", ".params.npy")
    # np.savez(param_output_path, **all_params)

def run(cfg, df, splits):
    n_problems = np.max(df['problem']) + 1
    cfg['n_problems'] = n_problems

    print("# of problems: %d, Students: %d" % (n_problems, np.max(df['student'])+1))
    gdf = df.groupby('problem')['student'].count()
    lower = np.percentile(gdf, q=2.5)
    upper = np.percentile(gdf, q=97.5)
    print("95%% occurance range: %d-%d" % (lower,upper))
    print("# of problems occuring at least 10 times: %d" % np.sum(gdf >= 10))
    
    if cfg['pred_layer'] == 'featurized_nido':
        if 'problem_feature_mat_path' in cfg:
            problem_feature_mat = np.load(cfg['problem_feature_mat_path'])
        else:
            problem_feature_mat = position_encode_problems.encode_problem_pos_distribs(df, n_problems)
            #problem_feature_mat = th.tensor(problem_feature_mat).float().to(cfg['device'])

        mu = np.mean(problem_feature_mat, axis=0, keepdims=True)
        std = np.std(problem_feature_mat, axis=0, ddof=1, keepdims=True)
        problem_feature_mat = (problem_feature_mat - mu) / std
        cfg['problem_feature_mat'] = th.tensor(problem_feature_mat).float().to(cfg['device'])
    
    seqs = to_student_sequences(df)
    
    results = []
    
    ref_assignment = get_problem_skill_assignment(df)

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

        model = train(train_seqs, valid_seqs, cfg)
        
        ytrue_test, log_ypred_test = predict(model, test_seqs, cfg)
        
        ypred_test = np.exp(log_ypred_test)

        run_result = metrics.calculate_metrics(ytrue_test, ypred_test)
        
        rand_index = compare_kc_assignment(model, ref_assignment)
        run_result['rand_index'] = rand_index

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

def train(train_seqs, 
          valid_seqs, 
          cfg):

    model = csbkt.CsbktModel(cfg).to(cfg['device'])
    pmf = joint_pmf.JointPMF(cfg['n_skills'])
    optimizer = th.optim.NAdam(model.parameters(), lr=cfg['lr'])
    
    stopping_rule = early_stopping_rules.PatienceRule(cfg['es_patience'], cfg['es_thres'], minimize=False)

    best_state = None
    for e in range(cfg['epochs']):
        np.random.shuffle(train_seqs)

        offset = 0
        end = offset + cfg['n_batch_seqs']
        batch_seqs = train_seqs[offset:end]

        batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
        batch_problem_seqs = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
        batch_mask_seqs = (pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1).to(cfg['device'])
            
        output, log_alpha = model(batch_obs_seqs, batch_problem_seqs)

        #logprob_same_kc = same_kc_loss(batch_problem_seqs, model.pred_layer.get_membership_logits()).flatten() # B*T

        train_loss = -(batch_obs_seqs * output[:, :, 1] + (1-batch_obs_seqs) * output[:, :, 0]).flatten() 
        
        mlogprobs = pmf(model.pred_layer.get_membership_logits())
        aux_loss = loss_sequence.nback_loss(batch_problem_seqs, batch_mask_seqs, mlogprobs, np.arange(cfg['min_lag'], cfg['max_lag']+1))

        mask_ix = batch_mask_seqs.flatten()
            
        train_loss = train_loss[mask_ix].mean() 
        #aux_loss = -logprob_same_kc[mask_ix].mean()
        train_loss = train_loss + cfg['aux_loss_coeff'] * aux_loss
            
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        yvalid_true, yvalid_logprob_correct = predict(model, valid_seqs, cfg)
        auc_roc = metrics.calculate_metrics(yvalid_true, yvalid_logprob_correct)['auc_roc']

        stop_training, new_best = stopping_rule.log(auc_roc)

        print("%4d Train loss: %8.4f, Valid AUC: %0.2f %s" % (e, train_loss.item(), auc_roc, '***' if new_best else ''))
        
        if new_best:
            best_state = copy.deepcopy(model.state_dict())
        
        if stop_training:
            break

    model.load_state_dict(best_state)

    return model
    

def predict(model, seqs, cfg):
    model.eval()
    seqs = sorted(seqs, key=lambda s: len(s), reverse=True)
    
    with th.no_grad():
        all_ypred = []
        all_ytrue = []

        for offset in range(0, len(seqs), cfg['n_test_batch_seqs']):
            end = offset + cfg['n_test_batch_seqs']
            batch_seqs = seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_problem_seqs = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_mask_seqs = (pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1).to(cfg['device'])

            log_prob, _ = model(batch_obs_seqs, batch_problem_seqs, test=True)
                
            ypred = log_prob[:, :, 1].flatten()
            ytrue = batch_obs_seqs.flatten()
            
            mask_ix = batch_mask_seqs.flatten()
                    
            ypred = ypred[mask_ix].cpu().numpy()
            ytrue = ytrue[mask_ix].cpu().numpy()

            all_ypred.append(ypred)
            all_ytrue.append(ytrue)
            
        ypred = np.hstack(all_ypred)
        ytrue = np.hstack(all_ytrue)
    model.train()

    return ytrue, ypred

def compare_kc_assignment(model, ref_assignment):

    membership_logits = model.get_membership_logits()
    pred_assignment = csbkt.get_effective_assignment(membership_logits)

    rand_index = sklearn.metrics.adjusted_rand_score(ref_assignment, pred_assignment)

    return rand_index

def get_problem_skill_assignment(df):

    problems_to_skills = dict(zip(df['problem'], df['skill']))
    n_problems = np.max(df['problem']) + 1
    return np.array([problems_to_skills[p] for p in range(n_problems)])
    

if __name__ == "__main__":
    main()