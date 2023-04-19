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
from scipy.stats import qmc

def main():
    import sys

    cfg_path = sys.argv[1]
    dataset_name = sys.argv[2]
    output_path = sys.argv[3]
    problem_rep_path = None
    if len(sys.argv) > 4:
        problem_rep_path = sys.argv[4]
    
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    
    splits = np.load("data/splits/%s.npy" % dataset_name)
    
    if problem_rep_path is None:
        problem_rep_mat = None
    else:
        problem_rep_mat = np.load(problem_rep_path)
    
    results_df, all_params = run(cfg, df, splits, problem_rep_mat)

    results_df.to_csv(output_path)

    param_output_path = output_path.replace(".csv", ".params.npy")
    np.savez(param_output_path, **all_params)

def run(cfg, df, splits, problem_rep_mat):
    
    problems_to_skills = dict(zip(df['problem'], df['skill']))
    n_problems = np.max(df['problem']) + 1
    A = np.array([problems_to_skills[p] for p in range(n_problems)])
    cfg['ref_labels'] = A
        
    seqs = to_student_sequences(df)
    
    all_ytrue = []
    all_ypred = []

    results = []
    all_params = defaultdict(list)

    if problem_rep_mat is None:
        problem_rep_mat = np.eye(n_problems)
    
    problem_rep_mat = th.tensor(problem_rep_mat).float().to('cuda:0')

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

        stopping_rule = create_early_stopping_rule(cfg['patience'], cfg.get('min_perc_improvement', 0))
        
        split_rep_mat = problem_rep_mat
        if len(split_rep_mat.shape) > 2:
            split_rep_mat = problem_rep_mat[i,:,:]
        
        model, best_aux = train(
            train_seqs, 
            valid_seqs, 
            problem_rep_mat=split_rep_mat, 
            device='cuda:0',
            stopping_rule=stopping_rule,
            **cfg)

        ytrue_test, log_ypred_test = predict(model, test_seqs, split_rep_mat, cfg['n_test_batch_seqs'], 'cuda:0', cfg['n_test_samples'])
        toc = time.perf_counter()

        ypred_test = np.exp(log_ypred_test)

        with th.no_grad():
            param_alpha, param_obs, param_t, Aprior = model.get_params(problem_rep_mat)
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

    all_ytrue = np.array(all_ytrue)
    all_ypred = np.array(all_ypred)

    
    results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(splits.shape[0])])
    
    return results_df, dict(all_params)


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
          problem_rep_mat, 
          n_latent_kcs,
          latent_dim,
          device, 
          learning_rate, 
          epochs, 
          n_batch_seqs, 
          stopping_rule, 
          tau, **kwargs):

    
    problem_dim = problem_rep_mat.shape[1]

    model = BktModel(n_latent_kcs, latent_dim, problem_dim)
    model = model.to(device)
    
    optimizer = th.optim.NAdam(model.parameters(), lr=learning_rate)
    
    best_state = None 
    best_rand_index = 0
    for e in range(epochs):
        np.random.shuffle(train_seqs)
        losses = []

        for offset in range(0, len(train_seqs), n_batch_seqs):
            end = offset + n_batch_seqs
            batch_seqs = train_seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_problem_seqs = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1
            
            rep_obs_seqs = []
            rep_actual_kc_seqs = []
            rep_mask_seqs = []
            rep_utilized_kcs = []
            for r in range(kwargs['n_train_samples']):

                #
                # sample assignment
                #
                A = model.sample_A(problem_rep_mat, tau, kwargs['hard_samples'])
                
                # 
                # translate problem to KC
                #
                actual_kc = A[batch_problem_seqs] # B X T X Latent KCs

                #
                # add to list
                #
                rep_obs_seqs.append(batch_obs_seqs)
                rep_actual_kc_seqs.append(actual_kc)
                rep_mask_seqs.append(batch_mask_seqs)
            
            #
            # construct final inputs to model
            #
            final_obs_seq = th.vstack(rep_obs_seqs).to(device)
            final_kc_seq = th.vstack(rep_actual_kc_seqs).to(device)
            final_mask_seq = th.vstack(rep_mask_seqs).to(device)
            mask_ix = final_mask_seq.flatten()

            output = model.forward(final_obs_seq, final_kc_seq)
            
            train_loss = -(final_obs_seq * output[:, :, 1] + (1-final_obs_seq) * output[:, :, 0]).flatten() 
            
            train_loss = train_loss[mask_ix].mean() + kwargs['reg'] * model.loglam.exp()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            losses.append(train_loss.item())
        
        mean_train_loss = np.mean(losses)

        #
        # Validation
        #
        ytrue, ypred = predict(model, valid_seqs, problem_rep_mat, kwargs['n_test_batch_seqs'], device, kwargs['n_valid_samples'])

        auc_roc = metrics.calculate_metrics(ytrue, ypred)['auc_roc']
        
        rand_index = 0
        n_utilized_kcs = 0
        with th.no_grad():
            ref_labels = kwargs['ref_labels']
            indecies = []
            n_utilized_kcs = []
            for s in range(100):
                A = model.sample_A(problem_rep_mat, 1e-6, True)
                n_utilized_kcs.append((A.sum(0) > 0).sum().cpu().numpy())
                if ref_labels is not None:
                    pred_labels = th.argmax(A, dim=1).cpu().numpy()
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
    

def predict(model, seqs, problem_rep_mat, n_batch_seqs, device, n_samples):
    model.eval()
    seqs = sorted(seqs, key=lambda s: len(s), reverse=True)
    with th.no_grad():

        all_ypred = []
        for sample in range(n_samples):
            sample_ypred = []
            all_ytrue = []

            #
            # draw sample assignment
            #
            A = model.sample_A(problem_rep_mat, 1e-6, True)

            for offset in range(0, len(seqs), n_batch_seqs):
                end = offset + n_batch_seqs
                batch_seqs = seqs[offset:end]

                batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0)
                batch_problem_seqs = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0)
                batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1

                actual_kc = A[batch_problem_seqs]

                output = model(batch_obs_seqs.to(device), actual_kc).cpu()
                    
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

class MultiHmmCell(jit.ScriptModule):
    
    def __init__(self):
        super(MultiHmmCell, self).__init__()
        
    @jit.script_method
    def forward(self, obs: Tensor, chain: Tensor, 
        trans_logits: Tensor, 
        obs_logits: Tensor, 
        init_logits: Tensor) -> Tensor:
        """
            Input:
                obs: [n_batch, t]
                chain: [n_batch, t, n_chains]
                trans_logits: [n_chains, n_states, n_states] (Target, Source)
                obs_logits: [n_chains, n_states, n_outputs]
                init_logits: [n_chains, n_states]
            output:
                [n_batch, t, n_outputs]
        """

        n_chains, n_states, n_outputs = obs_logits.shape 

        outputs = th.jit.annotate(List[Tensor], [])
        
        n_batch, _ = obs.shape
        batch_idx = th.arange(n_batch)

        log_alpha = F.log_softmax(init_logits, dim=1) # n_chains x n_states
        log_obs = F.log_softmax(obs_logits, dim=2) # n_chains x n_states x n_obs
        log_t = F.log_softmax(trans_logits, dim=1) # n_chains x n_states x n_states
        
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
    def __init__(self, n_latent_kcs, latent_dim, problem_dim, **kwargs):
        super(BktModel, self).__init__()
        
        #
        # Problem representations
        # If problems have features, then we use linear transform
        # otherwise we use the more efficient embedding
        #
        self.problem_embd = nn.Linear(problem_dim, latent_dim)
        
        #
        # KC representations
        #
        self.kc_embd = nn.Embedding(n_latent_kcs, latent_dim)
        
        #
        # Projects KC representation into BKT probabilities
        # pL, pF, pG, pS, pI0
        #
        self.kc_logits_f = nn.Linear(latent_dim, 5)

        #
        # controls the number of available KCs
        #
        self.loglam = nn.Parameter(th.randn(1).cuda())

        #
        # BKT Module
        #
        self.hmm = MultiHmmCell()
        
        self.problem_dim = problem_dim
        self.n_latent_kcs = n_latent_kcs
        self.kcs_range = th.arange(self.n_latent_kcs).cuda() / self.n_latent_kcs
        
    def sample_A(self, problem_reps, tau, hard_samples):
        """
            problem_reps: P x problem_dim where P is # of problems
                (None if problem_features is False)
            tau: Gumbel-softmax temperature
            hard_samples: return hard samples or not
            Output:
                Assignment matrix P x Latent KCs
        """
        membership_logits = self._compute_membership_logits(problem_reps)

        # sample
        A = nn.functional.gumbel_softmax(membership_logits, hard=hard_samples, tau=tau, dim=1)
        return A 
    
    def _compute_membership_logits(self, problem_reps):

        problem_embeddings = self.problem_embd(problem_reps)
        
        # compute logits
        membership_logits = problem_embeddings @ self.kc_embd.weight.T  # Problems x Latent KCs
        
        # reduce number of available KCs according to lambda
        lam = self.loglam.exp()
        decay_mat = th.exp(-self.kcs_range / lam)[None,:].cuda() # 1 x Latent KCs
        membership_logits = membership_logits * decay_mat + (1-decay_mat) * -10

        return membership_logits

    def forward(self, corr, actual_kc):
        trans_logits, obs_logits, init_logits = self._get_kc_params()
        return self.hmm(corr, actual_kc, trans_logits, obs_logits, init_logits)

    def _get_kc_params(self):
        kc_logits = self.kc_logits_f(self.kc_embd.weight) # Latent KCs x 5
        trans_logits = th.hstack((-kc_logits[:, [0]], # 1-pL
                                  kc_logits[:, [1]],  # pF
                                  kc_logits[:, [0]],  # pL
                                  -kc_logits[:, [1]])).reshape((-1, 2, 2)) # 1-pF (Latent KCs x 2 x 2)
        obs_logits = th.hstack((-kc_logits[:, [2]], # 1-pG
                                  kc_logits[:, [2]],  # pG
                                  kc_logits[:, [3]],  # pS
                                  -kc_logits[:, [3]])).reshape((-1, 2, 2)) # 1-pS (Latent KCs x 2 x 2)
        init_logits = th.hstack((-kc_logits[:, [4]], kc_logits[:, [4]])) # (Latent KCs x 2)

        return trans_logits, obs_logits, init_logits

    def get_params(self, problem_rep_mat):
        with th.no_grad():
            trans_logits, obs_logits, init_logits = self._get_kc_params()

            alpha = F.softmax(init_logits, dim=1) # n_chains x n_states
            obs = F.softmax(obs_logits, dim=2) # n_chains x n_states x n_obs
            t = F.softmax(trans_logits, dim=1) # n_chains x n_states x n_states

            membership_logits = self._compute_membership_logits(problem_rep_mat)
            kc_membership_probs = F.softmax(membership_logits, dim=1) # n_problems * n_latent_kcs

            return alpha, obs, t, kc_membership_probs

if __name__ == "__main__":
    main()
