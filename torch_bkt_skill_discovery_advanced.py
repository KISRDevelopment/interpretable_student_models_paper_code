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
import sklearn.metrics

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
            chain: [n_batch, t, n_chains]
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
            curr_chain = chain[:,i,:] # B X C
            
            curr_problem = problem[:,i] #n_batch

            # B X S X O
            logit = self.obs_logits_problem[curr_problem,:,:]
            log_obs = F.log_softmax(logit, dim=2)
            
            # predict
            a1 = (curr_chain[:,:,None, None] * log_obs[:,None,:,:]).sum(1) # B X S X O
            a2 = (curr_chain[:,:,None] * log_alpha).sum(1) # BXCX1 * BXCXS = BXS

            # B X S X O + B X S X 1
            log_py = th.logsumexp(a1 + a2[:,:,None], dim=1)  # B X O
            
            log_py = log_py - th.logsumexp(log_py, dim=1)[:,None]
            outputs += [log_py]

            # update
            curr_y = obs[:,i]
            a1 = log_obs[batch_idx,:,curr_y] # B X S
            log_py = (a1[:,None,:] * curr_chain[:,:,None]).sum(1) # B X S
            

            a1 = (log_alpha * curr_chain[:,:,None]).sum(1) # BxCxS * BxCx1 = BxS
            a2 = (log_t[None,:,:,:] * curr_chain[:,:,None,None]).sum(1) # 1xCxSxS * BxCx1x1 = BxSxS
            a3 = th.logsumexp(log_py[:,None,:] + a1[:,None,:] + a2, dim=2)

            # B x 1 X S + B x 1 x S + B x S x S = B x S
            log_alpha = (1 - curr_chain[:,:,None]) * log_alpha + curr_chain[:,:,None] * a3[:,None,:]

        outputs = th.stack(outputs)
        outputs = th.transpose(outputs, 0, 1)
        
        return outputs


class BktModel(nn.Module):
    def __init__(self, n_problems, n_kcs):
        super(BktModel, self).__init__()
        self.n_kcs = n_kcs

        n_initial_kcs = 2
        weight_matrix = th.rand((n_problems, n_kcs))
        weight_matrix[:, n_initial_kcs:] = -5

        self.kc_membership_logits = nn.Embedding.from_pretrained(weight_matrix, freeze=False)


        self.hmm = MultiHmmCell(2, 2, n_kcs, n_problems)
        
    def sample_A(self, tau, hard_samples):
        self._A = nn.functional.gumbel_softmax(self.kc_membership_logits.weight, hard=hard_samples, tau=tau, dim=1)
        
    def forward(self, corr, problem):
        kc = self._A[problem]
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

def train(train_seqs, valid_seqs, n_problems, device, learning_rate, epochs, patience, n_batch_seqs,**kwargs):

    model = BktModel(n_problems, kwargs['n_latent_kcs'])
    model.to(device)
    
    optimizer = th.optim.NAdam(model.parameters(), lr=learning_rate)
    best_val_auc_roc =  0.
    best_state = None 
    epochs_since_last_best = 0
    best_aux = {}
    for e in range(epochs):
        np.random.shuffle(train_seqs)
        losses = []

        for offset in range(0, len(train_seqs), n_batch_seqs):
            
            end = offset + n_batch_seqs
            batch_seqs = train_seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_problem_seqs = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1
            model.sample_A(kwargs['tau'], kwargs['hard_samples'])
        
            output = model(batch_obs_seqs.to(device), batch_problem_seqs.to(device)).cpu()
            
            train_loss = -(batch_obs_seqs * output[:, :, 1] + (1-batch_obs_seqs) * output[:, :, 0]).flatten()
            mask_ix = batch_mask_seqs.flatten()

            # K X P X 1 X 1
            # 1 x P X S X O
            # K x S X O
            hard_idx = th.argmax(model._A, dim=1)
            hard_A = th.zeros_like(model._A)
            hard_A[th.arange(hard_A.shape[0]), hard_idx] = 1
            hard_A = hard_A - model._A.detach() + model._A 
            n_utilized_kcs = hard_A.sum(0)
            n_utilized_kcs /= (n_utilized_kcs + 1e-3)
            
            kc_to_problem = model._A.T # K X P
            

            mu = (kc_to_problem[:,:,None,None] * model.hmm.obs_logits_problem[None,:,:,:]).sum(1) / (1e-6+kc_to_problem.sum(1, keepdim=True)[:,:,None])
            var = (model.hmm.obs_logits_problem[:,None,:,:] - mu[None,:,:,:]).square().sum(0) / (1e-6+kc_to_problem.sum(1, keepdim=True)[:,:,None]) # K X S X O
            #print(var.mean().item())
            #train_loss = train_loss[mask_ix].mean() + kwargs['lambda']*var.mean() / n_utilized_kcs.sum()
            train_loss = train_loss[mask_ix].mean() + kwargs['lambda']*var.mean()*th.log(n_utilized_kcs.sum())
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            losses.append(train_loss.item())
        
        mean_train_loss = np.mean(losses)

        #
        # Validation
        #
        ytrue, ypred = predict(model, valid_seqs, kwargs['n_test_batch_seqs'], device, kwargs['n_valid_samples'])

        with th.no_grad():
            ref_labels = kwargs['ref_labels']
            indecies = []
            n_utilized_kcs = []
            for s in range(100):
                model.sample_A(1e-6, True)
                n_utilized_kcs.append((model._A.sum(0) > 0).sum().cpu().numpy())
                pred_labels = th.argmax(model._A, dim=1).cpu().numpy()

                if ref_labels is not None:
                    rand_index = sklearn.metrics.adjusted_rand_score(ref_labels, pred_labels)
                    indecies.append(rand_index)
            rand_index = np.mean(indecies) if len(indecies) > 0 else 0
            n_utilized_kcs = np.mean(n_utilized_kcs)

        auc_roc = metrics.calculate_metrics(ytrue, ypred)['auc_roc']
        new_best = auc_roc > best_val_auc_roc

        print("%4d Train loss: %8.4f, Valid AUC: %0.2f (Rand: %0.2f, Utilized KCS: %d) %s" % (e, mean_train_loss, auc_roc, 
            rand_index,
            n_utilized_kcs,
            '***' if new_best else ''))
            
        if auc_roc > best_val_auc_roc:
            best_val_auc_roc = auc_roc
            epochs_since_last_best = 0
            best_state = copy.deepcopy(model.state_dict())
            best_aux['rand_index'] = rand_index
            best_aux['n_utilized_kcs'] = n_utilized_kcs
        else:
            epochs_since_last_best += 1
            
        if epochs_since_last_best >= patience:
            break

    model.load_state_dict(best_state)

    return model, best_aux
    

def predict(model, seqs, n_batch_seqs, device, n_samples):
    model.eval()
    with th.no_grad():
        all_ypred = []
        for sample in range(n_samples):
            sample_ypred = []
            all_ytrue = []
            model.sample_A(1e-6, True)
            for offset in range(0, len(seqs), n_batch_seqs):
                end = offset + n_batch_seqs
                batch_seqs = seqs[offset:end]

                batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0)
                batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1
                batch_problem_seqs = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0)
                
                output = model(batch_obs_seqs.to(device), batch_problem_seqs.to(device)).cpu()
                    
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
    n_problems = df['problem'].max() + 1

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

        model, best_aux = train(train_seqs, valid_seqs, 
            n_problems=n_problems,
            device='cuda:0',
            **cfg)

        ytrue_test, log_ypred_test = predict(model, test_seqs, cfg['n_test_batch_seqs'], 'cuda:0', cfg['n_test_samples'])
        
        ypred_test = np.exp(log_ypred_test)

        r = metrics.calculate_metrics(ytrue_test, ypred_test)
        r = { **r, **best_aux }
        results.append(r)
        all_ytrue.extend(ytrue_test)
        all_ypred.extend(ypred_test)

    all_ytrue = np.array(all_ytrue)
    all_ypred = np.array(all_ypred)

    overall_metrics = metrics.calculate_metrics(all_ytrue, all_ypred)
    results.append(overall_metrics)

    results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(splits.shape[0])] + ['Overall'])
    
    return result_df

if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1]
    dataset_name = sys.argv[2]
    output_path = sys.argv[3]

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    splits = np.load("data/splits/%s.npy" % dataset_name)
    results_df = main(cfg, df, splits)

    results_df.to_csv(output_path)

