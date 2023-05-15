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
import position_encode_problems 
import nnkmeans 
import sklearn.cluster

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
    
    print("# of problems: %d" % n_problems)
    gdf = df.groupby('problem')['student'].count()
    lower = np.percentile(gdf, q=2.5)
    upper = np.percentile(gdf, q=97.5)
    print("95%% occurance range: %d-%d" % (lower,upper))
    print("# of problems occuring at least 10 times: %d" % np.sum(gdf >= 10))
    #exit()
    seqs = to_student_sequences(df)
    
    all_ytrue = []
    all_ypred = []

    results = []
    all_params = defaultdict(list)

    if problem_rep_mat is not None:
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
        if problem_rep_mat is None:
            split_rep_mat = position_encode_problems.encode_problem_pos_distribs(train_df, n_problems)
            #split_rep_mat = position_encode_problems.encode_problem_ids(train_df, n_problems)
            split_rep_mat = th.tensor(split_rep_mat).float().to('cuda:0')
        else:
            if len(split_rep_mat.shape) > 2:
                split_rep_mat = problem_rep_mat[s,:,:]
        
        model, best_aux = train(
            train_seqs, 
            valid_seqs, 
            problem_rep_mat=split_rep_mat, 
            device='cuda:0',
            stopping_rule=stopping_rule,
            **cfg)
        
        if problem_rep_mat is None:
            split_rep_mat = position_encode_problems.encode_problem_pos_distribs(df[train_ix | valid_ix], n_problems)
            split_rep_mat = th.tensor(split_rep_mat).float().to('cuda:0')

        ytrue_test, log_ypred_test = predict(model, test_seqs, split_rep_mat, cfg['n_test_batch_seqs'], 'cuda:0', cfg['n_test_samples'])
        toc = time.perf_counter()

        ypred_test = np.exp(log_ypred_test)

        with th.no_grad():
            pass
            # param_alpha, param_obs, param_t, Aprior = model.get_params(split_rep_mat)
            # all_params['alpha'].append(param_alpha.cpu().numpy())
            # all_params['obs'].append(param_obs.cpu().numpy())
            # all_params['t'].append(param_t.cpu().numpy())
            # all_params['Aprior'].append(Aprior.cpu().numpy())
        
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
          n_batch_trials, 
          stopping_rule, 
          tau, **kwargs):

    
    problem_dim = problem_rep_mat.shape[1]

    model = BktModel(problem_dim, latent_dim, n_latent_kcs, kwargs['kmeans_epsilon'], 
        kwargs['kmeans_max_iter'], kwargs['kmeans_tau'], device)
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

            # determine # of problems in a batch because we don't want to handle
            # all problems 
            problems_in_batch = sorted(set.union(*[set(s['problem']) for s in batch_seqs]))
            
            # reindex
            problem_batch_idx = dict(zip(problems_in_batch, range(len(problems_in_batch))))

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_problem_seqs = pad_sequence([th.tensor([ problem_batch_idx[p] for p in s['problem']]) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1
            
            #print("# of problems in batch (%d / %d): %d" % (offset, len(train_seqs), len(problems_in_batch)))

            rep_obs_seqs = []
            rep_actual_kc_seqs = []
            rep_mask_seqs = []
            rep_utilized_kcs = []

            membership_logits = model.compute_membership_logits(problem_rep_mat[problems_in_batch, :])

            for r in range(kwargs['n_train_samples']):
                
                #
                # sample assignment
                # Problems in batch x Latent KCs
                #
                A = model.sample_A(membership_logits, tau, kwargs['hard_samples'])
                
                # 
                # translate problem to KC
                #
                actual_kc = A[batch_problem_seqs] # B X T X Latent KCs

                kc_reps = A.T @ problem_rep_mat[problems_in_batch, :] # Latent KC x d
                
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

            output = model.forward(final_obs_seq, final_kc_seq, kc_reps, n_batch_trials)
            
            train_loss = -(final_obs_seq * output[:, :, 1] + (1-final_obs_seq) * output[:, :, 0]).flatten() 
            
            train_loss = train_loss[mask_ix].mean() 

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
            membership_logits = model.compute_membership_logits(problem_rep_mat)
            #pred_labels_argmax = th.argmax(membership_logits, dim=1)
            # rand_index = sklearn.metrics.adjusted_rand_score(ref_labels, pred_labels)
            # n_utilized_kcs = 0

            for s in range(100):
                A = model.sample_A(membership_logits, 1, True)
                n_utilized_kcs.append((A.sum(0) > 0).sum().cpu().numpy())
                if ref_labels is not None:
                    pred_labels = th.argmax(A, dim=1).cpu().numpy()
                    
                    rand_index = sklearn.metrics.adjusted_rand_score(ref_labels, pred_labels)
                    indecies.append(rand_index)
            if ref_labels is not None:
                rand_index = np.mean(indecies)
            n_utilized_kcs = np.mean(n_utilized_kcs)


            print(model.kc_discovery.softmax_log_tau.exp())
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
    #print("Running prediction")
    with th.no_grad():

        all_ypred = []
        for sample in range(n_samples):
            sample_ypred = []
            all_ytrue = []
            
            #
            # draw sample assignment
            #
            membership_logits = model.compute_membership_logits(problem_rep_mat)
            A = model.sample_A(membership_logits, 1e-6, True)
            kc_reps = A.T @ problem_rep_mat # Latent KC x d
            
            #print("Sample %d" % sample)
            for offset in range(0, len(seqs), n_batch_seqs):
                end = offset + n_batch_seqs
                batch_seqs = seqs[offset:end]

                batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0).to(device)
                batch_problem_seqs = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0).to(device)
                batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1

                actual_kc = A[batch_problem_seqs]

                output = model(batch_obs_seqs, actual_kc, kc_reps, n_batch_trials=500).cpu()
                
                ypred = output[:, :, 1].flatten()
                ytrue = batch_obs_seqs.flatten()
                mask_ix = batch_mask_seqs.flatten()
                    
                ypred = ypred[mask_ix].numpy()
                ytrue = ytrue[mask_ix].cpu().numpy()

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

class KCDiscovery(nn.Module):

    def __init__(self, problem_dim, n_latent_kcs, kmeans_epsilon, kmeans_max_iter, kmeans_tau, device):
        super().__init__()
        
        self.n_latent_kcs = n_latent_kcs
        self.device = device
        self.kmeans = nnkmeans.NNKmeans(kmeans_epsilon, kmeans_max_iter)
        self.kmeans_log_tau = nn.Parameter(th.randn(1))
        self.softmax_log_tau = th.tensor(np.log(0.243)).float()
        self._centroids = None 

    def compute_membership_logits(self, problem_reps):
        if self._centroids is None:
            with th.no_grad():
                problem_reps_np = problem_reps.cpu().numpy()
                centroids,_ = sklearn.cluster.kmeans_plusplus(problem_reps_np, n_clusters=self.n_latent_kcs)
                self._centroids = th.tensor(centroids).float().to(self.device)
        
        logits, self._centroids = self.kmeans(problem_reps, self._centroids.detach(), self.kmeans_log_tau.exp()) # P x K
        
        return logits 
    
    def sample_A(self, membership_logits, tau, hard):
        return nn.functional.gumbel_softmax(membership_logits / self.softmax_log_tau.exp(), hard=hard, tau=tau, dim=1)
    
class BktModel(nn.Module):
    def __init__(self, problem_dim, latent_dim, n_latent_kcs, kmeans_epsilon, kmeans_max_iter, kmeans_tau, device):
        super(BktModel, self).__init__()
        
        #
        # Problem representations
        #
        # self.problem_embd = nn.Sequential(
        #     nn.Linear(problem_dim, latent_dim),
        #     nn.Tanh(),
        #     nn.Linear(latent_dim, latent_dim)
        # )
        
        #
        # KC Discovery
        self.kc_discovery = KCDiscovery(problem_dim, n_latent_kcs, kmeans_epsilon, kmeans_max_iter, kmeans_tau, device)
        
        #
        # Projects KC representation into BKT probabilities
        # pL, pF, pG, pS, pI0
        #
        self.kc_logits_f = nn.Sequential(
            nn.Linear(problem_dim, 5),
            nn.Tanh(),
            nn.Linear(5, 5))

        #
        # BKT Module
        #
        self.hmm = MultiHmmCell()
        
        self.problem_dim = problem_dim
        self.n_latent_kcs = n_latent_kcs
        
    def compute_membership_logits(self, problem_reps):
        return self.kc_discovery.compute_membership_logits(problem_reps)
    
    def sample_A(self, membership_logits, tau, hard_samples):
        """
            membership_logits: P x F where P is # of problems
            tau: Gumbel-softmax temperature
            hard_samples: return hard samples or not
            Output:
                Assignment matrix P x Latent KCs
        """
        return self.kc_discovery.sample_A(membership_logits, tau, hard_samples)
    
    def forward(self, corr, actual_kc, kc_reps, n_batch_trials):
        """
            kc_reps: BxN_latent_KCxd
        """
        trans_logits, obs_logits, init_logits = self._get_kc_params(kc_reps)
        logits, log_alpha = self.hmm(corr, actual_kc, trans_logits, obs_logits, init_logits)
        return logits 
        
        # n_trials = corr.shape[1]

        # batch_logits = []
        # for i in range(0, n_trials, n_batch_trials):
        #     to_trial = i + n_batch_trials
        #     corr_block = corr[:, i:to_trial]
        #     actual_kc_block = actual_kc[:, i:to_trial, :]
            
        #     if i == 0:
        #         logits, log_alpha = self.hmm(corr_block, actual_kc_block, trans_logits, obs_logits, init_logits)
        #     else:
        #         log_alpha = log_alpha.detach()
        #         logits, log_alpha = self.hmm.forward_given_alpha(corr_block, actual_kc_block, 
        #             trans_logits, obs_logits, init_logits, log_alpha)
            
        #     batch_logits.append(logits)
        
        # return th.concat(batch_logits, dim=1)

    def _get_kc_params(self, kc_reps):
        kc_logits = self.kc_logits_f(kc_reps) # Latent KCs x 5
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
        # with th.no_grad():
        #     trans_logits, obs_logits, init_logits = self._get_kc_params()

        #     alpha = F.softmax(init_logits, dim=1) # n_chains x n_states
        #     obs = F.softmax(obs_logits, dim=2) # n_chains x n_states x n_obs
        #     t = F.softmax(trans_logits, dim=1) # n_chains x n_states x n_states

        #     membership_logits = self._compute_membership_logits(problem_rep_mat)
        #     kc_membership_probs = F.softmax(membership_logits, dim=1) # n_problems * n_latent_kcs

        #     return alpha, obs, t, kc_membership_probs
        pass

class MultiHmmCell(jit.ScriptModule):
    
    def __init__(self):
        super(MultiHmmCell, self).__init__()

    @jit.script_method
    def forward(self, obs: Tensor, chain: Tensor, 
        trans_logits: Tensor, 
        obs_logits: Tensor, 
        init_logits: Tensor) -> Tuple[Tensor, Tensor]:

        n_batch = obs.shape[0]
        log_alpha = F.log_softmax(init_logits, dim=1) # n_chains x n_states
        log_alpha = th.tile(log_alpha, (n_batch, 1, 1)) # batch x chains x states
        return self.forward_given_alpha(obs, chain, trans_logits, obs_logits, init_logits, log_alpha)
    
    @jit.script_method
    def forward_given_alpha(self, obs: Tensor, chain: Tensor, 
        trans_logits: Tensor, 
        obs_logits: Tensor, 
        init_logits: Tensor,
        log_alpha: Tensor) -> Tuple[Tensor, Tensor]:
        """
            Input:
                obs: [n_batch, t]
                chain: [n_batch, t, n_chains]
                trans_logits: [n_chains, n_states, n_states] (Target, Source)
                obs_logits: [n_chains, n_states, n_outputs]
                init_logits: [n_chains, n_states]
                log_alpha: [n_batch, n_chains, n_states]
            output:
                logits: [n_batch, t, n_outputs]
                log_alpha: [n_batch, n_chains, n_states]
        """

        n_chains, n_states, n_outputs = obs_logits.shape 

        outputs = th.jit.annotate(List[Tensor], [])
        
        n_batch, _ = obs.shape
        batch_idx = th.arange(n_batch)

        
        log_obs = F.log_softmax(obs_logits, dim=2) # n_chains x n_states x n_obs
        log_t = F.log_softmax(trans_logits, dim=1) # n_chains x n_states x n_states
        
        # B X C X S
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
        
        return outputs, log_alpha


if __name__ == "__main__":
    main()
