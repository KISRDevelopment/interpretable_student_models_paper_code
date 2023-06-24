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
class MultiHmmCell(jit.ScriptModule):
    
    def __init__(self, n_states, n_outputs, n_chains):
        super(MultiHmmCell, self).__init__()
        
        self.n_states = n_states
        self.n_outputs = n_outputs
        self.n_chains = n_chains 

        # [n_hidden,n_hidden] (Target,Source)
        self.trans_logits = nn.Parameter(th.randn(n_chains, n_states, n_states))
        self.obs_logits = nn.Parameter(th.randn(n_chains, n_states, n_outputs))
        self.init_logits = nn.Parameter(th.randn(n_chains, n_states))
        
    @jit.script_method
    def forward(self, obs: Tensor, chain: Tensor) -> Tensor:
        """
            obs: [n_batch, t]
            chain: [n_batch, t, n_chains]
            output:
            [n_batch, t, n_outputs]
        """
        outputs = th.jit.annotate(List[Tensor], [])
        
        n_batch, _ = obs.shape
        batch_idx = th.arange(n_batch)

        log_alpha = F.log_softmax(self.init_logits, dim=1) # n_chains x n_states
        log_obs = F.log_softmax(self.obs_logits, dim=2) # n_chains x n_states x n_obs
        log_t = F.log_softmax(self.trans_logits, dim=1) # n_chains x n_states x n_states
        
        # B X C X S
        log_alpha = th.tile(log_alpha, (n_batch, 1, 1))
        for i in range(0, obs.shape[1]):
            curr_chain = chain[:,i,:] # B X C
            
            # predict
            a1 = (curr_chain[:,:,None, None] * log_obs[None,:,:,:]).sum(1) # B X S X O
            a2 = (curr_chain[:,:,None] * log_alpha).sum(1) # BXCX1 * BXCXS = BXS

            # B X S X O + B X S X 1
            log_py = th.logsumexp(a1 + a2[:,:,None], dim=1)  # B X O
            
            log_py = log_py - th.logsumexp(log_py, dim=1)[:,None]
            outputs += [log_py]

            # update
            curr_y = obs[:,i]
            a1 = th.permute(log_obs[:,:,curr_y], (2, 0, 1)) # B X C X S
            log_py = (a1 * curr_chain[:,:,None]).sum(1) # B X S
            

            a1 = (log_alpha * curr_chain[:,:,None]).sum(1) # BxCxS * BxCx1 = BxS
            a2 = (log_t[None,:,:,:] * curr_chain[:,:,None,None]).sum(1) # 1xCxSxS * BxCx1x1 = BxSxS
            a3 = th.logsumexp(log_py[:,None,:] + a1[:,None,:] + a2, dim=2)

            # B x 1 X S + B x 1 x S + B x S x S = B x S
            log_alpha = (1 - curr_chain[:,:,None]) * log_alpha + curr_chain[:,:,None] * a3[:,None,:]
        
        
        outputs = th.stack(outputs)
        outputs = th.transpose(outputs, 0, 1)
        
        return outputs

class BktModel(nn.Module):
    def __init__(self, n_kcs, n_latent_kcs, n_initial_kcs):
        super(BktModel, self).__init__()
        
        weight_matrix = th.rand((n_kcs, n_latent_kcs))
        weight_matrix[:, n_initial_kcs:] = -10

        self.kc_membership_logits = nn.Embedding.from_pretrained(weight_matrix, freeze=False)

        self.hmm = MultiHmmCell(2, 2, n_latent_kcs)
        self.n_kcs = n_kcs
        self.n_latent_kcs = n_latent_kcs

        self._A = None
        
    def sample_A(self, tau, hard_samples):
        
        self._A = nn.functional.gumbel_softmax(self.kc_membership_logits.weight, hard=hard_samples, tau=tau, dim=1)
        
    def forward(self, corr, kc):
        actual_kc = self._A[kc]
        return self.hmm(corr, actual_kc)

    def get_params(self):
        alpha = F.softmax(self.hmm.init_logits, dim=1) # n_chains x n_states
        obs = F.softmax(self.hmm.obs_logits, dim=2) # n_chains x n_states x n_obs
        t = F.softmax(self.hmm.trans_logits, dim=1) # n_chains x n_states x n_states
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

def train(train_seqs, valid_seqs, n_kcs, device, learning_rate, epochs, n_batch_seqs, stopping_rule, tau, **kwargs):

    model = BktModel(n_kcs, kwargs['n_latent_kcs'], kwargs['n_initial_kcs'])
    model = model.to(device)
    
    optimizer = th.optim.NAdam(model.parameters(), lr=learning_rate)
    
    best_state = None 
    best_rand_index = 0
    for e in range(epochs):
        np.random.shuffle(train_seqs)
        losses = []

        prev_n_utilized_kcs = kwargs['n_initial_kcs']
        for offset in range(0, len(train_seqs), n_batch_seqs):
            end = offset + n_batch_seqs
            batch_seqs = train_seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_kc_seqs = pad_sequence([th.tensor(s['kc']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1
            
            rep_obs_seqs = []
            rep_kc_seqs = []
            rep_mask_seqs = []
            rep_utilized_kcs = []
            for r in range(kwargs['n_train_samples']):
                model.sample_A(tau, kwargs['hard_samples'])
                
                actual_kc = model._A[batch_kc_seqs] #th.matmul(kc, self._A) # B X T X LC
                rep_obs_seqs.append(batch_obs_seqs)
                rep_kc_seqs.append(actual_kc)
                rep_mask_seqs.append(batch_mask_seqs)
            
            final_obs_seq = th.vstack(rep_obs_seqs).to(device)
            final_kc_seq = th.vstack(rep_kc_seqs).to(device)
            final_mask_seq = th.vstack(rep_mask_seqs).to(device)
            mask_ix = final_mask_seq.flatten()

            output = model.hmm(final_obs_seq, final_kc_seq)
            
            train_loss = -(final_obs_seq * output[:, :, 1] + (1-final_obs_seq) * output[:, :, 0]).flatten() 
             
            train_loss = train_loss[mask_ix].mean()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            losses.append(train_loss.item())
            print("%d out of %d" % (len(losses), np.ceil(len(train_seqs) / n_batch_seqs )))
        # tau = np.maximum(0.5, tau * 0.95)
        # print("new tau: %0.2f" % tau)
        mean_train_loss = np.mean(losses)

        #
        # Validation
        #
        ytrue, ypred = predict(model, valid_seqs, kwargs['n_test_batch_seqs'], device, kwargs['n_valid_samples'])

        auc_roc = metrics.calculate_metrics(ytrue, ypred)['auc_roc']
        
        rand_index = 0
        n_utilized_kcs = 0
        with th.no_grad():
            ref_labels = kwargs['ref_labels']
            indecies = []
            n_utilized_kcs = []
            for s in range(100):
                model.sample_A(1e-6, True)
                n_utilized_kcs.append((model._A.sum(0) > 0).sum().cpu().numpy())
                if ref_labels is not None:
                    pred_labels = th.argmax(model._A, dim=1).cpu().numpy()
                    rand_index = sklearn.metrics.adjusted_rand_score(ref_labels, pred_labels)
                    indecies.append(rand_index)
            if ref_labels is not None:
                rand_index = np.mean(indecies)
            n_utilized_kcs = np.mean(n_utilized_kcs)

        r = stopping_rule(auc_roc)

        print("%4d Train loss: %8.4f, Valid AUC: %0.2f (Rand: %0.2f, Utilized KCS: %d) %s" % (e, mean_train_loss, auc_roc, 
            rand_index,
            n_utilized_kcs,
            '***' if r['new_best'] else ''))
        
        if r['new_best']:
            best_state = copy.deepcopy(model.state_dict())
            best_aux = {
                "rand_index" : rand_index,
                "n_utilized_kcs" : n_utilized_kcs
            }
        if r['stop']:
            break

    model.load_state_dict(best_state)

    return model, best_aux
    

def predict(model, seqs, n_batch_seqs, device, n_samples):
    model.eval()
    seqs = sorted(seqs, key=lambda s: len(s), reverse=True)
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
                batch_kc_seqs = pad_sequence([th.tensor(s['kc']) for s in batch_seqs], batch_first=True, padding_value=0)
                batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1

                output = model(batch_obs_seqs.to(device), batch_kc_seqs.to(device)).cpu()
                    
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

def create_early_stopping_rule(patience, min_perc_improvement):

    best_value = -np.inf 
    waited = 0
    def stop(value):
        nonlocal best_value
        nonlocal waited 

        if np.isinf(best_value):
            best_value = value
            return { "stop" : False, "new_best" : True } 
        
        perc_improvement = (value - best_value)*100/best_value
        new_best = False
        if value > best_value:
            best_value = value 
            new_best = True 
        
        # only reset counter if more than min percent improvement
        # otherwise continue increasing
        if perc_improvement > min_perc_improvement:
            waited = 0
        else:
            waited += 1
        
        if waited >= patience:
            return { "stop" : True, "new_best" : new_best }
        
        return { "stop" : False, "new_best" : new_best }

    return stop

def main(cfg, df, splits):
    
    if cfg['use_problems']:
        df['skill'] = df['problem']

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

        stopping_rule = create_early_stopping_rule(cfg['patience'], cfg.get('min_perc_improvement', 0))

        model, best_aux = train(train_seqs, valid_seqs, 
            n_kcs=n_kcs, 
            device='cuda:0',
            stopping_rule=stopping_rule,
            **cfg)

        ytrue_test, log_ypred_test = predict(model, test_seqs, cfg['n_test_batch_seqs'], 'cuda:0', cfg['n_test_samples'])
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
        run_result = { **run_result , **best_aux }
        
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
        "patience" : 5,
        "tau" : 1.5,
        "n_latent_kcs" : 20,
        "lambda" : 0.00,
        "n_batch_seqs" : 20,
        "n_test_batch_seqs" : 500,
        "hard_samples" : False,
        "ref_labels" : None,
        "use_problems" : True,
        "n_initial_kcs" : 5,
        "n_valid_samples" : 50,
        "n_test_samples" : 50,
        "n_train_samples" : 1
    }

    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    splits = np.load("data/splits/%s.npy" % dataset_name)
    results_df, all_params = main(cfg, df, splits)
