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

@jit.script
def get_logits(dynamics_logits, obs_logits):
    """
        dynamics_logits: chains x 3 (pL, pF, pI0)
        obs_logits: chains x 2 (pG, pS)
        Returns:
            trans_logits, obs_logits, init_logits
    """
    trans_logits = th.hstack((dynamics_logits[:, [0]]*0, # 1-pL
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

class BktModel(nn.Module):
    def __init__(self, cfg, n_kcs):
        super(BktModel, self).__init__()
        
        self.cfg = cfg 

        # KC Discovery
        if 'problem_feature_mat' in cfg:
            print("Using problem features ...")
            self.kc_discovery = layer_kc_discovery.FeaturizedKCDiscovery(cfg['problem_feature_mat'], n_kcs)
        else:
            self.kc_discovery = layer_kc_discovery.SimpleKCDiscovery(cfg['n_kcs'], n_kcs, cfg['initial_kcs'])
        
        self._dynamics_logits = nn.Parameter(th.randn(n_kcs, 3)) # pL, pF, pI0
        self._obs_logits = nn.Parameter(th.randn(n_kcs, 2)) # pG, pS

        self.hmm = layer_multihmmcell.MultiHmmCell()
    
    def sample(self, tau, hard):
        return self.kc_discovery.sample_A(tau, hard)

    def forward(self, corr, actual_kc):
        #actual_kc = A[problem, :]
        trans_logits, obs_logits, init_logits = get_logits(self._dynamics_logits, self._obs_logits)

        log_obs = F.log_softmax(obs_logits, dim=2)
        log_t = F.log_softmax(trans_logits, dim=1)

        return self.hmm(corr, actual_kc, log_t, log_obs, init_logits)

    def get_membership_logits(self):
        return self.kc_discovery.get_logits()

    def get_params(self):
        kc_membership_probs = F.softmax(self.kc_discovery.get_logits(), dim=1) # n_problems * n_latent_kcs
        trans_logits, obs_logits, init_logits = get_logits(self._dynamics_logits, self._obs_logits)
        return trans_logits, obs_logits, init_logits, kc_membership_probs

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

    model = BktModel(cfg, cfg['n_latent_kcs'])

    model = model.to(cfg['device'])
    
    optimizer = th.optim.NAdam(model.parameters(), lr=cfg['learning_rate'])
    
    stopping_rule = early_stopping_rules.PatienceRule(cfg['es_patience'], cfg['es_thres'], minimize=False)


    best_state = None 
    best_rand_index = 0
    for e in range(cfg['epochs']):
        np.random.shuffle(train_seqs)
        losses = []

        done = False
        for offset in range(0, len(train_seqs), cfg['n_train_batch_seqs']):
            end = offset + cfg['n_train_batch_seqs']
            batch_seqs = train_seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_kc_seqs = pad_sequence([th.tensor(s['kc']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1).to(cfg['device']) > -1
            
            mask_ix = batch_mask_seqs.flatten()
            
            A = model.sample(cfg['tau'], cfg['hard_train_samples'])
            output = model(batch_obs_seqs, A[batch_kc_seqs, :])
            
            train_loss = -(batch_obs_seqs * output[:, :, 1] + (1-batch_obs_seqs) * output[:, :, 0]).flatten() 
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
            #print("%d out of %d" % (len(losses), np.ceil(len(train_seqs) / cfg['n_train_batch_seqs'] )))

            if not cfg['full_epochs'] or (end >= len(train_seqs)):
                
                mean_train_loss = np.mean(losses)
                losses = []
                #
                # Validation
                #
                ytrue, ypred = predict(model, valid_seqs, cfg)

                auc_roc = metrics.calculate_metrics(ytrue, ypred)['auc_roc']
                
                stop_training, new_best = stopping_rule.log(auc_roc)

                print("%4d Train loss: %8.4f, Valid AUC: %0.2f %s" % (e, mean_train_loss, auc_roc, '***' if new_best else ''))
                
                if new_best:
                    best_state = copy.deepcopy(model.state_dict())
                    best_auc_roc = auc_roc

                if stop_training:
                    done = True
                    break
        if done:
            break 
    
    model.load_state_dict(best_state)

    return model
    

def predict(model, seqs, cfg):
    
    model.eval()
    seqs = sorted(seqs, key=lambda s: len(s), reverse=True)
    with th.no_grad():

        # draw sample assignments
        As = [model.sample(1e-6, True)[None,:,:] for i in range(cfg['n_test_samples'])]
        As = th.vstack(As) # SxPxK
        
        #print("Sampled")

        # go over batches
        batch_logpred_correct = []
        batch_corr = []
        for offset in range(0, len(seqs), cfg['n_test_batch_seqs']):
            end = offset + cfg['n_test_batch_seqs']
            batch_seqs = seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_problem_seqs = pad_sequence([th.tensor(s['kc']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1).to(cfg['device']) > -1

            mask_ix = batch_mask_seqs.flatten()

            # call model
            tiled_batch_obs_seqs = th.tile(batch_obs_seqs, (cfg['n_test_samples'], 1)) # S*BxT
            B, T, K = batch_obs_seqs.shape[0], batch_obs_seqs.shape[1], As.shape[2]
            actual_kc_seqs = As[:, batch_problem_seqs, :].view(-1, T, K) # S*BxTxK
            logpred = model(tiled_batch_obs_seqs, actual_kc_seqs) # S*BxTx2
            logpred_correct = logpred[:, :, 1] # S*BxT
            logpred_correct = logpred_correct.view(-1, B, T) # SxBxT
            logpred_correct = logpred_correct.mean(0) # BxT

            #final_logpred_correct = th.logsumexp(logpred_correct, dim=0) - np.log(logpred_correct.shape[0]) # BxT
            #final_logpred_correct = logpred_correct.mean(0)
            
            logpred_correct = logpred_correct.flatten()
            logpred_correct = logpred_correct[mask_ix]

            batch_logpred_correct.append(logpred_correct.cpu().numpy())

            # get actual observations
            corr = batch_obs_seqs.flatten() 
            corr = corr[mask_ix] # Nb
            batch_corr.append(corr.cpu().numpy())
        
        # concatenate everything
        logpred = np.hstack(batch_logpred_correct)
        ytrue = np.hstack(batch_corr)
    model.train()
    
    return ytrue, logpred


def main(cfg, df, splits):
    
    ref_assignment = get_problem_skill_assignment(df)
    if 'problem_feature_mat_path' in cfg:
        problem_feature_mat = np.load(cfg['problem_feature_mat_path'])
        #mu = np.mean(problem_feature_mat, axis=0, keepdims=True)
        #std = np.std(problem_feature_mat, axis=0, ddof=1, keepdims=True)
        #problem_feature_mat = (problem_feature_mat - mu) / (std+1e-9)
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

        tic = time.perf_counter()

        model = train(train_seqs, valid_seqs, cfg)

        ytrue_test, log_ypred_test = predict(model, test_seqs, cfg)
        toc = time.perf_counter()

        ypred_test = np.exp(log_ypred_test)

        model.eval()
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
        membership_logits = model.get_membership_logits().cpu().numpy()
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
        print("Trimming problems")
        utils.trim_problems(df, cfg['min_problem_freq'])

    cfg['device'] = 'cuda:0'
    results_df, all_params = main(cfg, df, splits)

    results_df.to_csv(output_path)

    param_output_path = output_path.replace(".csv", ".params.npy")
    np.savez(param_output_path, **all_params)

