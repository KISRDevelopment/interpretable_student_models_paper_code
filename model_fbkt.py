import numpy as np 
import numpy.random

import torch as th 
import torch.nn.functional as F 
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence
import torch.jit as jit
from torch import Tensor
from typing import List

import itertools
import pandas as pd 
import sys 
import json 
from collections import defaultdict
import metrics 
import copy 

import early_stopping_rules
import time 

import utils 
import sklearn.metrics

def main():
    cfg_path = sys.argv[1]
    dataset_name = sys.argv[2]
    output_path = sys.argv[3]

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    splits = np.load("data/splits/%s.npy" % dataset_name)

    cfg['n_kcs'] = np.max(df['skill']) + 1
    cfg['device'] = 'cuda:0'

    results_df, all_params = run(cfg, df, splits) 
    results_df.to_csv(output_path)

    param_output_path = output_path.replace(".csv", ".params.npy")
    np.savez(param_output_path, **all_params)

def run(cfg, df, splits):

    seqs = to_student_sequences(df)
    
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
        
        # Train & Test
        tic = time.perf_counter()
        model = train(cfg, train_seqs, valid_seqs)
        ytrue, log_ypred_correct = predict(cfg, model, test_seqs)
        toc = time.perf_counter()

        ypred_correct = np.exp(log_ypred_correct)

        run_result = metrics.calculate_metrics(ytrue, ypred_correct)
        run_result['time_diff_sec'] = toc - tic 

        results.append(run_result)

        with th.no_grad():
            param_alpha, param_obs, param_t = model.get_params()
            all_params['alpha'].append(param_alpha.cpu().numpy())
            all_params['obs'].append(param_obs.cpu().numpy())
            all_params['t'].append(param_t.cpu().numpy())
        
        print(run_result)
        
    results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(splits.shape[0])])
    print(results_df)
    
    return results_df, all_params

def train(cfg, train_seqs, valid_seqs):
    
    #train_seqs = split_seqs_by_kc(train_seqs)

    #valid_seqs = split_seqs_by_kc(valid_seqs)
    #valid_seqs = sorted(valid_seqs, reverse=True, key=lambda s: len(s[1]))

    tic = time.perf_counter()
    model = BktModel(cfg['n_kcs'], cfg['fastbkt_n'], cfg['device'])
    toc = time.perf_counter()
    print("Model creation: %f" % (toc - tic))

    optimizer = th.optim.NAdam(model.parameters(), lr=cfg['learning_rate'])
    
    stopping_rule = early_stopping_rules.PatienceRule(cfg['es_patience'], cfg['es_thres'], minimize=False)


    n_seqs = len(train_seqs) if cfg['full_epochs'] else cfg['n_train_batch_seqs']

    best_state = None

    tic_global = time.perf_counter()
    for e in range(cfg['epochs']):
        np.random.shuffle(train_seqs)
        losses = []

        tic = time.perf_counter()
        for offset in range(0, n_seqs, cfg['n_train_batch_seqs']):
            end = offset + cfg['n_train_batch_seqs']
            batch_seqs = split_seqs_by_kc(train_seqs[offset:end])
            
            # prepare model input
            batch_obs_seqs = pad_to_multiple([seq for kc, seq in batch_seqs], multiple=cfg['fastbkt_n'], padding_value=0).float().to(cfg['device'])
            batch_kc = th.tensor([kc for kc, _ in batch_seqs]).long().to(cfg['device'])
            mask = pad_to_multiple([seq for kc, seq in batch_seqs], multiple=cfg['fastbkt_n'], padding_value=-1).to(cfg['device']) > -1

            output = model(batch_obs_seqs, batch_kc)

            train_loss = -(batch_obs_seqs * output[:, :, 1] + (1-batch_obs_seqs) * output[:, :, 0]).flatten()
            mask_ix = mask.flatten()

            train_loss = train_loss[mask_ix].mean()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            losses.append(train_loss.item())
        toc = time.perf_counter()
        print("Train time: %f" % (toc - tic))
        mean_train_loss = np.mean(losses)
       
        #
        # Validation
        #
        tic = time.perf_counter()   
        ytrue, ypred = predict(cfg, model, valid_seqs)
        toc = time.perf_counter()
        print("Predict time: %f" % (toc - tic))

        print("Evaluation:")
        print(ytrue.shape, ytrue.dtype)
        print(ypred.shape, ypred.dtype)
        tic = time.perf_counter()
        auc_roc = sklearn.metrics.roc_auc_score(ytrue, ypred)
        toc = time.perf_counter()
        print("Evaluation time: %f" % (toc - tic))
        stop_training, new_best = stopping_rule.log(auc_roc)

        print("%4d Train loss: %8.4f, Valid AUC: %0.2f %s" % (e, mean_train_loss, auc_roc, '***' if new_best else ''))
        
        if new_best:
            best_state = copy.deepcopy(model.state_dict())
        
        if stop_training:
            break
    toc_global = time.perf_counter()
    print("Total train time: %f" % (toc_global - tic_global))

    model.load_state_dict(best_state)
    return model

        
def predict(cfg, model, seqs):

    seqs = sorted(seqs, reverse=True, key=lambda s: len(s))
    
    model.eval()
    with th.no_grad():
        all_ypred = []
        all_ytrue = []
        for offset in range(0, len(seqs), cfg['n_test_batch_seqs']):
            end = offset + cfg['n_test_batch_seqs']
            batch_seqs = split_seqs_by_kc(seqs[offset:end])

            batch_obs_seqs = pad_to_multiple([seq for kc, seq in batch_seqs], multiple=cfg['fastbkt_n'], padding_value=0).float().to(cfg['device'])
            batch_kc = th.tensor([kc for kc, _ in batch_seqs]).long().to(cfg['device'])
            mask = pad_to_multiple([seq for kc, seq in batch_seqs], multiple=cfg['fastbkt_n'], padding_value=-1).to(cfg['device']) > -1

            output = model(batch_obs_seqs, batch_kc)
                
            ypred = output[:, :, 1].flatten()
            ytrue = batch_obs_seqs.flatten()
            mask_ix = mask.flatten()
                
            ypred = ypred[mask_ix].cpu().numpy()
            ytrue = ytrue[mask_ix].cpu().int().numpy()

            all_ypred.append(ypred)
            all_ytrue.append(ytrue)
            
        ypred = np.hstack(all_ypred)
        ytrue = np.hstack(all_ytrue)
    model.train()
    
    return ytrue, ypred
def to_student_sequences(df):
    seqs = defaultdict(lambda: defaultdict(list))
    for r in df.itertuples():
        seqs[r.student][r.skill].append(r.correct)
    return seqs

def split_seqs_by_kc(seqs):
    splitted = []
    for by_kc in seqs:
        for kc, seq in by_kc.items():
            splitted.append((kc, seq))
    return splitted

def pad_to_multiple(seqs, multiple, padding_value):
    return th.tensor(utils.pad_to_multiple(seqs, multiple, padding_value))

class BktModel(nn.Module):

    def __init__(self, n_kcs, fastbkt_n, device):
        super(BktModel, self).__init__()
        self._logits = nn.Embedding(n_kcs, 5).to(device) # pL, pF, pG, pS, pI0
        self._model = FastBkt(fastbkt_n, device)

    def forward(self, corr, kc):
        """
            Input:
                corr: trial correctness     BxT
                kc: kc membership (long)    B
            Returns:
                logprob_pred: log probability of correctness BxTx2
        """
        logits = self._logits(kc) # Bx5
        return self._model(corr, logits)

    def get_params(self):
        
        kc_logits = self._logits.weight
        trans_logits = th.hstack((-kc_logits[:, [0]], # 1-pL
                                kc_logits[:, [1]],  # pF
                                kc_logits[:, [0]],  # pL
                                -kc_logits[:, [1]])).reshape((-1, 2, 2)) # 1-pF (Latent KCs x 2 x 2)
        obs_logits = th.hstack((-kc_logits[:, [2]], # 1-pG
                                kc_logits[:, [2]],  # pG
                                kc_logits[:, [3]],  # pS
                                -kc_logits[:, [3]])).reshape((-1, 2, 2)) # 1-pS (Latent KCs x 2 x 2)
        init_logits = th.hstack((-kc_logits[:, [4]], kc_logits[:, [4]])) # (Latent KCs x 2)


        alpha = F.softmax(init_logits, dim=1) # n_chains x n_states
        obs = F.softmax(obs_logits, dim=2) # n_chains x n_states x n_obs
        t = F.softmax(trans_logits, dim=1) # n_chains x n_states x n_states
        
        return alpha, obs, t
    
class FastBkt(jit.ScriptModule):

    def __init__(self, n, device):
        super(FastBkt, self).__init__()
        
        self.n = n 

        # all possible trajectories of length n
        trajectories = make_trajectories(n)

        # transition indecies to compute the probaiblity of a trajectory
        # 0: -pL, 1: pL, 2: pF, 3: 1-pF, 4: 1-pI0, 5: pI0 
        trans_ind = make_transition_indices(trajectories)

        # transition indecies to compute the posterior predictive over
        # the hidden state
        pred_ind = make_predictive_indices(trajectories)

        # move to torch
        self._trajectories = th.tensor(trajectories).float().to(device)
        self._trans_ind = th.tensor(trans_ind).long().to(device)
        self._pred_ind = th.tensor(pred_ind).long().to(device)
        self._device = device 

    
    def forward(self, corr: Tensor, logits: Tensor) -> Tensor:
        """
            Input: 
                corr: trial sequence BxT
                logit: BKT parameter logits (pL, pF, pG, pS, pI0) Bx5
            Output:
                output_logprobs BxTx2
        """

        # give nice names to params (B)
        logit_pL, logit_pF, logit_pG, logit_pS, logit_pI0 = logits[:, 0], logits[:, 1], logits[:, 2], logits[:, 3], logits[:, 4]

        # transition probabilities that will be indexed into
        trans_logits = th.vstack((-logit_pL, logit_pL, logit_pF, -logit_pF, -logit_pI0, logit_pI0)).T # Bx5
        trans_logprobs = F.logsigmoid(trans_logits) # Bx5

        # probability of answering correctly Bx2**NxN
        # 1x2**NxN * Bx1x1 
        obs_logits_correct = self._trajectories[None,:,:] * (-logit_pS[:,None,None]) + (1-self._trajectories[None,:,:]) * logit_pG[:,None,None]
        obs_logprob_correct = F.logsigmoid(obs_logits_correct) # Bx2**NxN

        # probability of answering incorrectly Bx2**NxN
        # print(logit_pS.shape)
        # print(self._trajectories.shape)
        obs_logits_incorrect = self._trajectories[None,:,:] * (logit_pS[:,None,None]) + (1-self._trajectories[None,:,:]) * (-logit_pG[:,None,None])
        obs_logprob_incorrect = F.logsigmoid(obs_logits_incorrect) # Bx2**NxN

        #
        # probability of each trajectory
        #

        # Bx2**Nx1 * Bx1x1
        initial_logprobs = self._trajectories[None,:, [0]] * trans_logprobs[:,None,[5]] + \
            (1-self._trajectories[None, :, [0]]) * trans_logprobs[:,None,[4]] # Bx2**Nx1
        
        # Bx2**Nx1
        logprob_h = trans_logprobs[:,self._trans_ind].sum(2, keepdims=True) + initial_logprobs
        
        # iterate over time 
        pred_corrects = th.jit.annotate(List[Tensor], [])
        pred_incorrects = th.jit.annotate(List[Tensor], [])
        for i in range(0, corr.shape[1], self.n):
            from_index = i
            to_index = from_index + self.n 

            # grab slice BxN
            corr_slice = corr[:, from_index:to_index]

            # likelikelihood of observing each observation under each trajectory Bx2**NxN
            # corr_slice[:,None,:]          Bx1xN
            # obs_logprob_correct           Bx2**NxN
            obs_loglik_given_h = corr_slice[:,None,:] * obs_logprob_correct + \
                (1-corr_slice[:,None,:]) * obs_logprob_incorrect

            # running likelihood of prior observations Bx2**NxN
            past_loglik_given_h = obs_loglik_given_h.cumsum(2).roll(dims=2, shifts=1)
            past_loglik_given_h[:,:,0] = 0.0

            # probability of correct/incorrect for each trajectory (weighted by trajectory weight)
            # Bx2**NxN + Bx2**NxN + Bx2**Nx1 = Bx2**NxN
            pred_correct_given_h = past_loglik_given_h + obs_logprob_correct + logprob_h
            pred_incorrect_given_h = past_loglik_given_h + obs_logprob_incorrect + logprob_h

            # unnormed probabilities of correctness and incorrectness BxN
            pred_correct = th.logsumexp(pred_correct_given_h, dim=1)
            pred_incorrect = th.logsumexp(pred_incorrect_given_h, dim=1)
            pred_corrects.append(pred_correct)
            pred_incorrects.append(pred_incorrect)

            #
            # new state prior for next iteration
            #
            seq_loglik_given_h = obs_loglik_given_h.sum(2) # Bx2**N
            seq_loglik = seq_loglik_given_h + logprob_h[:,:,0] # Bx2**N
            next_h_one_logprob = F.logsigmoid(trans_logits[:, self._pred_ind]) + seq_loglik # Bx2**N
            next_h_zero_logprob = F.logsigmoid(-trans_logits[:, self._pred_ind]) + seq_loglik  # Bx2**N
            next_h_one_logprob = th.logsumexp(next_h_one_logprob, dim=1, keepdims=True) # Bx1
            next_h_zero_logprob = th.logsumexp(next_h_zero_logprob, dim=1, keepdims=True) # Bx1
            
            # 1x2**N * Bx1 = Bx2**N
            initial_logprobs = self._trajectories[None,:,0] * next_h_one_logprob + (1-self._trajectories[None,:,0]) * next_h_zero_logprob
            
            # Bx2**Nx1 + Bx2**Nx1 = Bx2**Nx1
            logprob_h = trans_logprobs[:,self._trans_ind].sum(2, keepdims=True) + initial_logprobs[:,:,None]

        # BxT
        pred_corrects = th.concat(pred_corrects, dim=1)
        pred_incorrects = th.concat(pred_incorrects, dim=1)

        # BxTx2
        preds = th.concat((pred_incorrects[:,:,None], pred_corrects[:,:,None]), dim=2)
        
        # BxTx2 - BxTx1
        logprob_next = preds - th.logsumexp(preds, dim=2, keepdims=True)

        # BxTx2
        return logprob_next

def make_trajectories(n):
    """
        constructs all possible trajectories of binary state of
        a sequence of length n.
        Returns a matrix of size 2^n x n
    """
    trajectories = np.array(list(itertools.product(*[[0, 1]]*n)))
    
    return trajectories

def make_transition_indices(trajectories):
    """
        computes the transition indices
        Returns a matrix of size 2^n x n-1
        because it excludes the first trial.
    """

    convolved = np.zeros_like(trajectories)
    for i in range(trajectories.shape[0]):
        indices = np.convolve(trajectories[i,:], [1, 2], mode='same')
        convolved[i, :] = indices
    convolved = convolved[:,1:]

    return convolved

def make_predictive_indices(trajectories):
    """
        computes the indices to predict the transition from
        the last state.
    """
    target_state = np.ones(trajectories.shape[0])
    indices = trajectories[:,-1] * 2 + target_state
    return indices

if __name__ == "__main__":
    main()
