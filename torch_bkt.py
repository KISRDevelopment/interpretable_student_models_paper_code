#
#   Neural BKT RNN Implementation
#
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
import early_stopping_rules 

import sklearn.metrics 

class MultiHmmCell(jit.ScriptModule):
    
    def __init__(self, n_states, n_outputs, n_chains):
        super(MultiHmmCell, self).__init__()
        
        self.n_states = n_states
        self.n_outputs = n_outputs
        
        # [n_hidden,n_hidden] (Target,Source)
        self.trans_logits = nn.Parameter(th.randn(n_chains, n_states, n_states))
        self.obs_logits = nn.Parameter(th.randn(n_chains, n_states, n_outputs))
        self.init_logits = nn.Parameter(th.randn(n_chains, n_states))
        
    @jit.script_method
    def forward(self, obs: Tensor, chain: Tensor) -> Tensor:
        """
            obs: [n_batch, t]
            chain: [n_batch, t]
            output:
            [n_batch, t, n_outputs]
        """
        outputs = th.jit.annotate(List[Tensor], [])
        
        n_batch, _ = obs.shape
        batch_idx = th.arange(n_batch)

        log_alpha = F.log_softmax(self.init_logits, dim=1) # n_chains x n_states
        log_obs = F.log_softmax(self.obs_logits, dim=2) # n_chains x n_states x n_obs
        log_t = F.log_softmax(self.trans_logits, dim=1) # n_chains x n_states x n_states
        
        # [n_batch, n_chains, n_states]
        log_alpha = th.tile(log_alpha, (n_batch, 1, 1))
        for i in range(0, obs.shape[1]):
            curr_chain = chain[:,i] # [n_batch]
            
            # predict
            # B X S X O + B X S X 1
            log_py = th.logsumexp(log_obs[curr_chain,:,:] + log_alpha[batch_idx, curr_chain, :, None], dim=1)  #[n_batch, n_obs]
            log_py = log_py - th.logsumexp(log_py, dim=1)[:,None]
            outputs += [log_py]

            # update
            curr_y = obs[:,i]
            log_py = log_obs[curr_chain, :, curr_y] # [n_batch, n_states]
            
            # B x 1 X S + B x 1 x S + B x S x S
            log_alpha[batch_idx, curr_chain, :] = th.logsumexp(log_py[:,None,:] + log_alpha[batch_idx,curr_chain,None,:] + log_t[curr_chain,:,:], dim=2)
        
        outputs = th.stack(outputs)
        outputs = th.transpose(outputs, 0, 1)
        
        return outputs

class BktModel(nn.Module):
    def __init__(self, n_kcs):
        super(BktModel, self).__init__()
        self.hmm = MultiHmmCell(2, 2, n_kcs)

    def forward(self, corr, kc):
        return self.hmm(corr, kc)

    def get_params(self):
        alpha = F.softmax(self.hmm.init_logits, dim=1) # n_chains x n_states
        obs = F.softmax(self.hmm.obs_logits, dim=2) # n_chains x n_states x n_obs
        t = F.softmax(self.hmm.trans_logits, dim=1) # n_chains x n_states x n_states
        
        return alpha, obs, t

def to_student_sequences(df):
    seqs = defaultdict(lambda: {
        "obs" : [],
        "kc" : []
    })
    for r in df.itertuples():
        seqs[r.student]["obs"].append(r.correct)
        seqs[r.student]["kc"].append(r.skill)
    return seqs

def train(cfg, train_seqs, valid_seqs):

    model = BktModel(cfg['n_kcs'])
    model = model.to(cfg['device'])
    
    optimizer = th.optim.NAdam(model.parameters(), lr=cfg['learning_rate'])
    
    best_state = None 
    
    n_batch_seqs = cfg['n_train_batch_seqs']
    n_valid_seqs = cfg['n_test_batch_seqs']

    stopping_rule = early_stopping_rules.PatienceRule(cfg['es_patience'], cfg['es_thres'], minimize=False)

    for e in range(cfg['epochs']):
        np.random.shuffle(train_seqs)
        losses = []

        n_seqs = len(train_seqs) if cfg['full_epochs'] else n_batch_seqs

        #tic = time.perf_counter()

        for offset in range(0, n_seqs, n_batch_seqs):
            end = offset + n_batch_seqs
            batch_seqs = train_seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_kc_seqs = pad_sequence([th.tensor(s['kc']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1
            batch_mask_seqs = batch_mask_seqs.to(cfg['device'])

            output = model(batch_obs_seqs, batch_kc_seqs)
            
            train_loss = -(batch_obs_seqs * output[:, :, 1] + (1-batch_obs_seqs) * output[:, :, 0]).flatten()
            mask_ix = batch_mask_seqs.flatten()

            train_loss = train_loss[mask_ix].mean()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            losses.append(train_loss.item())
        # toc = time.perf_counter()
        # print("Train time: %f" % (toc - tic))

        mean_train_loss = np.mean(losses)

        #
        # Validation
        #
        #tic = time.perf_counter()
        ytrue, ypred = predict(cfg, model, valid_seqs)
        #toc = time.perf_counter()
        #print("Predict time: %f" % (toc - tic))

        #print("Evaluation:")
        #print(ytrue.shape, ytrue.dtype)
        #print(ypred.shape, ypred.dtype)
        #tic = time.perf_counter()
        auc_roc = sklearn.metrics.roc_auc_score(ytrue, ypred)
        #toc = time.perf_counter()
        #print("Evaluation time: %f" % (toc - tic))

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
        all_ypred = []
        all_ytrue = []
        for offset in range(0, len(seqs), cfg['n_test_batch_seqs']):
            end = offset + cfg['n_test_batch_seqs']
            batch_seqs = seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_kc_seqs = pad_sequence([th.tensor(s['kc']) for s in batch_seqs], batch_first=True, padding_value=0).to(cfg['device'])
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1
            batch_mask_seqs = batch_mask_seqs.to(cfg['device'])

            output = model(batch_obs_seqs.to(cfg['device']), batch_kc_seqs.to(cfg['device']))
                
            ypred = output[:, :, 1].flatten()
            ytrue = batch_obs_seqs.flatten()
            mask_ix = batch_mask_seqs.flatten()
            ypred = ypred[mask_ix]
            ytrue = ytrue[mask_ix]

            all_ypred.append(ypred.cpu().numpy())
            all_ytrue.append(ytrue.cpu().int().numpy())
            
        ypred = np.hstack(all_ypred)
        ytrue = np.hstack(all_ytrue)
    model.train()
    
    return ytrue, ypred


def main(cfg, df, splits):
    
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
        model = train(cfg, train_seqs, valid_seqs)
        ytrue_test, log_ypred_test = predict(cfg, model, test_seqs)
        toc = time.perf_counter()

        ypred_test = np.exp(log_ypred_test)

        with th.no_grad():
            param_alpha, param_obs, param_t = model.get_params()
            all_params['alpha'].append(param_alpha.cpu().numpy())
            all_params['obs'].append(param_obs.cpu().numpy())
            all_params['t'].append(param_t.cpu().numpy())

        run_result = metrics.calculate_metrics(ytrue_test, ypred_test)
        run_result['time_diff_sec'] = toc - tic 

        results.append(run_result)
        all_ytrue.extend(ytrue_test)
        all_ypred.extend(ypred_test)

    all_ytrue = np.array(all_ytrue)
    all_ypred = np.array(all_ypred)

    results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(splits.shape[0])])
    print(results_df)

    return results_df, all_params

if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1]
    dataset_name = sys.argv[2]
    output_path = sys.argv[3]

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    n_kcs = int(np.max(df['skill']) + 1)

    cfg['n_kcs'] = n_kcs 
    cfg['device'] = 'cuda:0'

    splits = np.load("data/splits/%s.npy" % dataset_name)
    results_df, all_params = main(cfg, df, splits)

    results_df.to_csv(output_path)

    param_output_path = output_path.replace(".csv", ".params.npy")
    np.savez(param_output_path, **all_params)
