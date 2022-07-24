#
#   Bayesian Knowledge Tracing PyTorch Implementation
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
class MultiHmmCell(jit.ScriptModule):
    
    def __init__(self, n_states, n_outputs, n_chains, n_problems):
        super(MultiHmmCell, self).__init__()
        
        self.n_states = n_states
        self.n_outputs = n_outputs
        
        # [n_hidden,n_hidden] (Target,Source)
        self.trans_logits = nn.Parameter(th.randn(n_chains, n_states, n_states))

        # we initialize to zeros so that the model puts more emphasis on kc logits
        # and so unseen problems have zero logit
        self.obs_logits_problem = nn.Parameter(th.zeros(n_problems, n_states, n_outputs))
        self.obs_logits_kc = nn.Parameter(th.randn(n_chains, n_states, n_outputs))

        self.init_logits = nn.Parameter(th.randn(n_chains, n_states))
        
    @jit.script_method
    def forward(self, obs: Tensor, chain: Tensor, problem: Tensor) -> Tensor:
        """
            obs: [n_batch, t]
            chain: [n_batch, t]
            problem: [n_batch, t]
            output:
            [n_batch, t, n_outputs]
        """
        
        outputs = th.jit.annotate(List[Tensor], [])
        
        n_batch, _ = obs.shape
        batch_idx = th.arange(n_batch)

        log_alpha = F.log_softmax(self.init_logits, dim=1) # n_chains x n_states
        log_t = F.log_softmax(self.trans_logits, dim=1) # n_chains x n_states x n_states
        
        # [n_batch, n_chains, n_states]
        log_alpha = th.tile(log_alpha, (n_batch, 1, 1))
        for i in range(0, obs.shape[1]):
            curr_chain = chain[:,i] # [n_batch]
            curr_problem = problem[:,i] #n_batch

            # B X S X O
            logit = self.obs_logits_problem[curr_problem,:,:] + self.obs_logits_kc[curr_chain,:,:]
            log_obs = F.log_softmax(logit, dim=2)

            # predict
            # B X S X O + B X S X 1
            log_py = th.logsumexp(log_obs + log_alpha[batch_idx, curr_chain, :, None], dim=1)  #[n_batch, n_obs]
            log_py = log_py - th.logsumexp(log_py, dim=1)[:,None]
            outputs += [log_py]

            # update
            curr_y = obs[:,i]
            log_py = log_obs[batch_idx, :, curr_y] # [n_batch, n_states]
            
            # B x 1 X S + B x 1 x S + B x S x S
            log_alpha[batch_idx, curr_chain, :] = th.logsumexp(log_py[:,None,:] + log_alpha[batch_idx,curr_chain,None,:] + log_t[curr_chain,:,:], dim=2)
        
        outputs = th.stack(outputs)
        outputs = th.transpose(outputs, 0, 1)
        
        return outputs


class BktModel(nn.Module):
    def __init__(self, n_kcs, n_problems):
        super(BktModel, self).__init__()
        self.hmm = MultiHmmCell(2, 2, n_kcs, n_problems)

    def forward(self, corr, kc, problem):
        
        return self.hmm(corr, kc, problem)

def to_student_sequences(df):
    seqs = defaultdict(lambda: {
        "obs" : [],
        "kc" : [],
        "problem" : []
    })
    for r in df.itertuples():
        seqs[r.student]["obs"].append(r.correct)
        seqs[r.student]["kc"].append(r.skill)
        seqs[r.student]["problem"].append(r.problem)
    
    return seqs

def train(train_seqs, valid_seqs, n_kcs, n_problems, device, learning_rate, epochs, patience, n_batch_seqs):

    model = BktModel(n_kcs, n_problems)
    model.to(device)
    
    optimizer = th.optim.NAdam(model.parameters(), lr=learning_rate)
    best_val_auc_roc =  0.5
    best_state = None 
    epochs_since_last_best = 0

    for e in range(epochs):
        np.random.shuffle(train_seqs)
        losses = []

        for offset in range(0, len(train_seqs), n_batch_seqs):
            end = offset + n_batch_seqs
            batch_seqs = train_seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_kc_seqs = pad_sequence([th.tensor(s['kc']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_problem_seqs = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1

            output = model(batch_obs_seqs.to(device), batch_kc_seqs.to(device), batch_problem_seqs.to(device)).cpu()
            
            train_loss = -(batch_obs_seqs * output[:, :, 1] + (1-batch_obs_seqs) * output[:, :, 0]).flatten()
            mask_ix = batch_mask_seqs.flatten()

            train_loss = train_loss[mask_ix].mean()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            losses.append(train_loss.item())
        
        mean_train_loss = np.mean(losses)

        #
        # Validation
        #
        ytrue, ypred = predict(model, valid_seqs, n_batch_seqs, device)

        auc_roc = metrics.calculate_metrics(ytrue, ypred)['auc_roc']
        print("Train loss: %8.4f, Valid AUC: %0.2f" % (mean_train_loss, auc_roc))
        if auc_roc > best_val_auc_roc:
            best_val_auc_roc = auc_roc
            epochs_since_last_best = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_since_last_best += 1
            
        if epochs_since_last_best >= patience:
            break

    model.load_state_dict(best_state)

    return model
    

def predict(model, seqs, n_batch_seqs, device):
    model.eval()
    with th.no_grad():
        all_ypred = []
        all_ytrue = []
        for offset in range(0, len(seqs), n_batch_seqs):
            end = offset + n_batch_seqs
            batch_seqs = seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_kc_seqs = pad_sequence([th.tensor(s['kc']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1
            batch_problem_seqs = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0)
            
            output = model(batch_obs_seqs.to(device), batch_kc_seqs.to(device), batch_problem_seqs.to(device)).cpu()
                
            ypred = output[:, :, 1].flatten()
            ytrue = batch_obs_seqs.flatten()
            mask_ix = batch_mask_seqs.flatten()
                
            ypred = ypred[mask_ix].numpy()
            ytrue = ytrue[mask_ix].numpy()

            all_ypred.append(ypred)
            all_ytrue.append(ytrue)
            
        ypred = np.hstack(all_ypred)
        ytrue = np.hstack(all_ytrue)
    model.train()

    return ytrue, ypred

        

def main():
    df = pd.read_csv("data/datasets/gervetetal_statics.csv")
    n_problems = df['problem'].max() + 1
    print("Problems: %d" % n_problems)
    gdf = df.groupby('problem')['student'].count()
    ix = gdf >= 100
    print("Problems by at least 100: %d" % np.sum(ix))
    
    seqs = to_student_sequences(df)
    
    splits = np.load("data/splits/gervetetal_statics.npy")
    split = splits[0, :]

    train_ix = split == 2
    valid_ix = split == 1
    test_ix = split == 0

    train_students = set(df[train_ix]['student'])
    valid_students = set(df[valid_ix]['student'])
    test_students = set(df[test_ix]['student'])

    train_problems = set(df[train_ix]['problem'])
    test_problems = set(df[test_ix]['problem'])
    print("Problems in test but not in train: %d" % len(test_problems - train_problems))
    
    train_seqs = [seqs[s] for s in train_students]
    valid_seqs = [seqs[s] for s in valid_students]
    test_seqs = [seqs[s] for s in test_students]

    n_kcs = int(np.max(df['skill']) + 1)
    model = train(train_seqs, 
                  valid_seqs,
                  n_kcs,
                  n_problems,
                  device='cuda:0', 
                  learning_rate=0.5, 
                  epochs=20, 
                  patience=5,
                  n_batch_seqs=500)

    ytrue_test, ypred_test = predict(model, test_seqs, 500, 'cuda:0')
    auc_roc = metrics.calculate_metrics(ytrue_test, ypred_test)['auc_roc']

    print("Test auc: %0.2f" % (auc_roc))

def main(cfg_path, dataset_name, output_path):
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    n_problems = df['problem'].max() + 1

    splits = np.load("data/splits/%s.npy" % dataset_name)
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

        n_kcs = int(np.max(df['skill']) + 1)
        model = train(train_seqs, valid_seqs, 
            n_kcs=n_kcs, 
            n_problems=n_problems,
            device='cuda:0',
            **cfg)

        ytrue_test, log_ypred_test = predict(model, test_seqs, cfg['n_batch_seqs'], 'cuda:0')
        
        ypred_test = np.exp(log_ypred_test)

        results.append(metrics.calculate_metrics(ytrue_test, ypred_test))
        all_ytrue.extend(ytrue_test)
        all_ypred.extend(ypred_test)

    all_ytrue = np.array(all_ytrue)
    all_ypred = np.array(all_ypred)

    overall_metrics = metrics.calculate_metrics(all_ytrue, all_ypred)
    results.append(overall_metrics)

    results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(splits.shape[0])] + ['Overall'])
    print(results_df)

    results_df.to_csv(output_path)

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
