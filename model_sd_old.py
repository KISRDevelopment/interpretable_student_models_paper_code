import numpy as np
import metrics
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from typing import List
import pandas as pd 
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import copy 
import json
import time 
import sklearn.metrics
import early_stopping_rules 
import layer_multihmmcell
import layer_kc_discovery
import loss_sequence
import utils
class BktModel(nn.Module):
    def __init__(self, cfg):
        super(BktModel, self).__init__()
        
        self.cfg = cfg 

        # KC Discovery
        if 'problem_feature_mat' in cfg:
            self.kc_discovery = layer_kc_discovery.FeaturizedKCDiscovery(cfg['problem_feature_mat'], cfg['n_latent_kcs'])
        else:
            self.kc_discovery = layer_kc_discovery.SimpleKCDiscovery(cfg['n_kcs'], cfg['n_latent_kcs'])

        self.hmm = layer_multihmmcell.MultiHmmCell()

    def get_kc_params(self, tau, hard_samples):
        return self.kc_discovery.get_params(tau, hard_samples)
    
    def forward(self, corr, actual_kc, trans_logits, obs_logits, init_logits):
        return self.hmm(corr, actual_kc, trans_logits, obs_logits, init_logits)

    def get_params(self):
        params = self.kc_discovery.get_params(1e-6, True)
        kc_membership_probs = F.softmax(self.kc_discovery.get_logits(), dim=1) # n_problems * n_latent_kcs

        return params[1], params[2], params[3], kc_membership_probs

def to_student_sequences(df):
    seqs = defaultdict(lambda: {
        "obs" : [],
        "kc" : []
    })
    for r in df.itertuples():
        seqs[r.student]["obs"].append(r.correct)
        seqs[r.student]["kc"].append(r.skill)
    return seqs

def train(train_seqs, valid_seqs, cfg):

    model = BktModel(cfg)
    model = model.to(cfg['device'])
    
    optimizer = th.optim.NAdam(model.parameters(), lr=cfg['learning_rate'])
    
    stopping_rule = early_stopping_rules.PatienceRule(cfg['es_patience'], cfg['es_thres'], minimize=False)


    best_state = None 
    best_rand_index = 0
    for e in range(cfg['epochs']):
        np.random.shuffle(train_seqs)
        losses = []

        for offset in range(0, len(train_seqs), cfg['n_train_batch_seqs']):
            end = offset + cfg['n_train_batch_seqs']
            batch_seqs = train_seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_kc_seqs = pad_sequence([th.tensor(s['kc']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1).to(cfg['device']) > -1
            
            A, trans_logits, obs_logits, init_logits = model.get_kc_params(cfg['tau'], cfg['hard_train_samples'])
            actual_kc = A[batch_kc_seqs] # B X T X LC

            output = model(batch_obs_seqs, actual_kc, trans_logits, obs_logits, init_logits)
            
            train_loss = -(batch_obs_seqs * output[:, :, 1] + (1-batch_obs_seqs) * output[:, :, 0]).flatten() 
            
            mask_ix = batch_mask_seqs.flatten()
            train_loss = train_loss[mask_ix].mean()

            aux_loss = 0
            if cfg['aux_loss_coeff'] > 0:
                mlogprobs = F.log_softmax(model.kc_discovery.get_logits(), dim=1)
                aux_loss = loss_sequence.nback_loss(batch_kc_seqs, batch_mask_seqs, mlogprobs, np.arange(cfg['min_lag'], cfg['max_lag']+1))
            train_loss = train_loss + cfg['aux_loss_coeff'] * aux_loss

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            losses.append(train_loss.item())
            print("%d out of %d" % (len(losses), np.ceil(len(train_seqs) / cfg['n_train_batch_seqs'] )))
        
        mean_train_loss = np.mean(losses)

        #
        # Validation
        #
        ytrue, ypred = predict(model, valid_seqs, cfg)

        auc_roc = metrics.calculate_metrics(ytrue, ypred)['auc_roc']
        
        stop_training, new_best = stopping_rule.log(auc_roc)

        print("%4d Train loss: %8.4f, Valid AUC: %0.2f %s" % (e, mean_train_loss, auc_roc, '***' if new_best else ''))
        
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
        for sample in range(cfg['n_test_samples']):
            sample_ypred = []
            all_ytrue = []

            A, trans_logits, obs_logits, init_logits = model.get_kc_params(1e-6, True)
            for offset in range(0, len(seqs), cfg['n_test_batch_seqs']):
                end = offset + cfg['n_test_batch_seqs']
                batch_seqs = seqs[offset:end]

                batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0)
                batch_kc_seqs = pad_sequence([th.tensor(s['kc']) for s in batch_seqs], batch_first=True, padding_value=0)
                batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1

                actual_kc = A[batch_kc_seqs.to(cfg['device'])]
                output = model(batch_obs_seqs.to(cfg['device']), actual_kc, trans_logits, obs_logits, init_logits).cpu()
                    
                ypred = output[:, :, 1].flatten()
                ytrue = batch_obs_seqs.flatten()
                mask_ix = batch_mask_seqs.flatten()
                    
                ypred = ypred[mask_ix].numpy()
                ytrue = ytrue[mask_ix].numpy()

                sample_ypred.append(ypred)
                all_ytrue.append(ytrue)
                
            sample_ypred = np.hstack(sample_ypred)
            ytrue = np.hstack(all_ytrue)
            
            all_ypred.append(sample_ypred)
        ypred = np.mean(np.vstack(all_ypred), axis=0)

    model.train()

    return ytrue, ypred

def main(cfg, df, splits):
    
    ref_assignment = get_problem_skill_assignment(df)
    if 'problem_rep_mat_path' in cfg:
        problem_feature_mat = np.load(cfg['problem_rep_mat_path'])
        mu = np.mean(problem_feature_mat, axis=0, keepdims=True)
        std = np.std(problem_feature_mat, axis=0, ddof=1, keepdims=True)
        problem_feature_mat = (problem_feature_mat - mu) / (std+1e-9)
        cfg['problem_feature_mat'] = th.tensor(problem_feature_mat).float().to(cfg['device'])

    df['skill'] = df['problem']
    cfg['n_kcs'] = np.max(df['problem']) + 1
    
    
    seqs = to_student_sequences(df)
    
    all_ytrue = []
    all_ypred = []

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

        n_kcs = int(np.max(df['skill']) + 1)

        tic = time.perf_counter()

        model = train(train_seqs, valid_seqs, cfg)

        ytrue_test, log_ypred_test = predict(model, test_seqs, cfg)
        toc = time.perf_counter()

        ypred_test = np.exp(log_ypred_test)

        with th.no_grad():
            param_alpha, param_obs, param_t, Aprior = model.get_params()
            all_params['alpha'].append(param_alpha.cpu().numpy())
            all_params['obs'].append(param_obs.cpu().numpy())
            all_params['t'].append(param_t.cpu().numpy())
            all_params['Aprior'].append(Aprior.cpu().numpy())
        
        run_result = metrics.calculate_metrics(ytrue_test, ypred_test)
        run_result['time_diff_sec'] = toc - tic 
        run_result['rand_index'] = compare_kc_assignment(model, ref_assignment)
        results.append(run_result)
        all_ytrue.extend(ytrue_test)
        all_ypred.extend(ypred_test)

        results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(len(results))])
        print(results_df)
    
    
    all_ytrue = np.array(all_ytrue)
    all_ypred = np.array(all_ypred)

    
    results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(splits.shape[0])])
    
    return results_df, dict(all_params)


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

    cfg['device'] = 'cuda:0'
    results_df, all_params = main(cfg, df, splits)

    results_df.to_csv(output_path)

    param_output_path = output_path.replace(".csv", ".params.npy")
    np.savez(param_output_path, **all_params)

