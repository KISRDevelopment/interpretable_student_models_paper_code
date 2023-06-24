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

class BktModel(nn.Module):
    def __init__(self, cfg):
        super(BktModel, self).__init__()
        
        self.cfg = cfg 

        weight_matrix = th.rand((cfg['n_kcs'], cfg['n_latent_kcs']))
        weight_matrix[:, cfg['n_initial_kcs']:] = -10

        self.kc_membership_logits = nn.Embedding.from_pretrained(weight_matrix, freeze=False)

        # [n_hidden,n_hidden] (Target,Source)
        self.trans_logits = nn.Parameter(th.randn(cfg['n_latent_kcs'], 2, 2))
        self.obs_logits = nn.Parameter(th.randn(cfg['n_latent_kcs'], 2, 2))
        self.init_logits = nn.Parameter(th.randn(cfg['n_latent_kcs'], 2))

        self.hmm = layer_multihmmcell.MultiHmmCell()

        self._A = None
        
    def sample_A(self, tau, hard_samples):
        
        self._A = nn.functional.gumbel_softmax(self.kc_membership_logits.weight, hard=hard_samples, tau=tau, dim=1)
    
    def forward(self, corr, actual_kc):
        
        return self.hmm(corr, actual_kc, self.trans_logits, self.obs_logits, self.init_logits)

    def get_params(self):
        alpha = F.softmax(self.init_logits, dim=1) # n_chains x n_states
        obs = F.softmax(self.obs_logits, dim=2) # n_chains x n_states x n_obs
        t = F.softmax(self.trans_logits, dim=1) # n_chains x n_states x n_states
        kc_membership_probs = F.softmax(self.kc_membership_logits.weight, dim=1) # n_problems * n_latent_kcs

        return alpha, obs, t, kc_membership_probs

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

        prev_n_utilized_kcs = cfg['n_initial_kcs']
        for offset in range(0, len(train_seqs), cfg['n_train_batch_seqs']):
            end = offset + cfg['n_train_batch_seqs']
            batch_seqs = train_seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_kc_seqs = pad_sequence([th.tensor(s['kc']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1
            
            rep_obs_seqs = []
            rep_kc_seqs = []
            rep_mask_seqs = []
            rep_utilized_kcs = []
            for r in range(cfg['n_train_samples']):
                model.sample_A(cfg['tau'], cfg['hard_train_samples'])
                actual_kc = model._A[batch_kc_seqs] # B X T X LC
                rep_obs_seqs.append(batch_obs_seqs)
                rep_kc_seqs.append(actual_kc)
                rep_mask_seqs.append(batch_mask_seqs)
            
            final_obs_seq = th.vstack(rep_obs_seqs).to(cfg['device'])
            final_kc_seq = th.vstack(rep_kc_seqs).to(cfg['device'])
            final_mask_seq = th.vstack(rep_mask_seqs).to(cfg['device'])
            mask_ix = final_mask_seq.flatten()

            output = model(final_obs_seq, final_kc_seq)
            
            train_loss = -(final_obs_seq * output[:, :, 1] + (1-final_obs_seq) * output[:, :, 0]).flatten() 
             
            train_loss = train_loss[mask_ix].mean()

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
        
        # rand_index = 0
        # n_utilized_kcs = 0
        # with th.no_grad():
        #     ref_labels = cfg['ref_labels']
        #     indecies = []
        #     n_utilized_kcs = []
        #     for s in range(100):
        #         model.sample_A(1e-6, True)
        #         n_utilized_kcs.append((model._A.sum(0) > 0).sum().cpu().numpy())
        #         if ref_labels is not None:
        #             pred_labels = th.argmax(model._A, dim=1).cpu().numpy()
        #             rand_index = sklearn.metrics.adjusted_rand_score(ref_labels, pred_labels)
        #             indecies.append(rand_index)
        #     if ref_labels is not None:
        #         rand_index = np.mean(indecies)
        #     n_utilized_kcs = np.mean(n_utilized_kcs)

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

            model.sample_A(1e-6, True)
            for offset in range(0, len(seqs), cfg['n_test_batch_seqs']):
                end = offset + cfg['n_test_batch_seqs']
                batch_seqs = seqs[offset:end]

                batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0)
                batch_kc_seqs = pad_sequence([th.tensor(s['kc']) for s in batch_seqs], batch_first=True, padding_value=0)
                batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1

                actual_kc = model._A[batch_kc_seqs.to(cfg['device'])]
                output = model(batch_obs_seqs.to(cfg['device']), actual_kc).cpu()
                    
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
    
    if cfg['use_problems']:
        df['skill'] = df['problem']
        cfg['n_kcs'] = np.max(df['problem']) + 1
        cfg['device'] = 'cuda:0'

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
        
        results.append(run_result)
        all_ytrue.extend(ytrue_test)
        all_ypred.extend(ypred_test)

        results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(len(results))])
        print(results_df)
    
    
    all_ytrue = np.array(all_ytrue)
    all_ypred = np.array(all_ypred)

    
    results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(splits.shape[0])])
    
    return results_df, dict(all_params)

if __name__ == "__main__":
    import sys

    dataset_name = sys.argv[1]
    
    cfg = {
        "learning_rate" : 0.1, 
        "epochs" : 20, 
        "es_patience" : 10,
        "es_thres" : 0.01,
        "tau" : 1.5,
        "n_latent_kcs" : 20,
        "lambda" : 0.00,
        "n_train_batch_seqs" : 200,
        "n_test_batch_seqs" : 500,
        "hard_train_samples" : False,
        "ref_labels" : None,
        "use_problems" : True,
        "n_initial_kcs" : 5,
        "n_test_samples" : 50,
        "n_train_samples" : 1
    }

    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    splits = np.load("data/splits/%s.npy" % dataset_name)
    results_df, all_params = main(cfg, df, splits)
