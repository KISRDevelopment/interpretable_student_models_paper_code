import numpy as np 
import torch as th 
import torch.nn as nn
import sys 
import json 
import pandas as pd 
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import metrics 
import copy 

def main():
    cfg_path = sys.argv[1]
    dataset_name = sys.argv[2]
    output_path = sys.argv[3]

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    splits = np.load("data/splits/%s.npy" % dataset_name)

    evaluate(cfg, df, splits)

    # results_df, all_params = main(cfg, df, splits)

    # results_df.to_csv(output_path)

    # param_output_path = output_path.replace(".csv", ".params.npy")
    # np.savez(param_output_path, **all_params)

def evaluate(cfg, df, splits, device='cuda:0'):
    n_problems = np.max(df['problem']) + 1
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

        n_train_batch_seqs = cfg['n_train_batch_seqs']
        n_valid_batch_seqs = cfg['n_test_batch_seqs']
        print("Train batch size: %d, valid: %d" % (n_train_batch_seqs, n_valid_batch_seqs))

        
        model = train(train_seqs, 
            valid_seqs, 
            n_problems,
            cfg,
            device)

        n_test_batch_seqs = cfg['n_test_batch_seqs']
        print("Test batch size: %d" % n_test_batch_seqs)

        ytrue_test, ypred_test = predict(model, test_seqs, n_test_batch_seqs, device)
        
        # with th.no_grad():
        #     param_alpha, param_obs, param_t = model.get_params()
        #     all_params['alpha'].append(param_alpha.cpu().numpy())
        #     all_params['obs'].append(param_obs.cpu().numpy())
        #     all_params['t'].append(param_t.cpu().numpy())

        run_result = metrics.calculate_metrics(ytrue_test, ypred_test)
        
        results.append(run_result)
        all_ytrue.extend(ytrue_test)
        all_ypred.extend(ypred_test)

    all_ytrue = np.array(all_ytrue)
    all_ypred = np.array(all_ypred)

    results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(splits.shape[0])])
    print(results_df)

    return results_df, all_params

def train(train_seqs, valid_seqs, n_problems, cfg, device):

    model = ExpMovAvgModel(n_problems, cfg['n_hidden'])
    model = model.to(device)
    
    optimizer = th.optim.NAdam(model.parameters(), lr=cfg['learning_rate'])
    
    best_auc_roc = 0.
    best_state = None 
    waited = 0

    n_batch_seqs = cfg['n_train_batch_seqs']
    n_valid_seqs = cfg['n_valid_batch_seqs']
    for e in range(cfg['epochs']):
        np.random.shuffle(train_seqs)
        losses = []

        for offset in range(0, len(train_seqs), n_batch_seqs):
            end = offset + n_batch_seqs
            batch_seqs = train_seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0).to(device)
            batch_problem_seqs = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0).to(device)
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1
            batch_mask_seqs = batch_mask_seqs.to(device)

            pC = model(batch_obs_seqs, batch_problem_seqs)
            logpC = th.log(pC)
            logpnC = th.log(1 - pC)

            train_loss = -(batch_obs_seqs * logpC + (1-batch_obs_seqs) * logpnC).flatten()
            mask_ix = batch_mask_seqs.flatten()

            sim_mat = model.problem_sim_mat()

            train_loss = train_loss[mask_ix].mean() + cfg['lambda'] * sim_mat.square().sum()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            losses.append(train_loss.item())
        
        mean_train_loss = np.mean(losses)
        
        #
        # Validation
        #
        ytrue, ypred = predict(model, valid_seqs, n_valid_seqs, device)

        auc_roc = metrics.calculate_metrics(ytrue, ypred)['auc_roc']

        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_state = copy.deepcopy(model.state_dict())
            waited = 0
            new_best = True 
        else:
            new_best = False 
            waited += 1

        print("%4d Train loss: %8.4f, Valid AUC: %0.2f %s" % (e, mean_train_loss, auc_roc, '***' if new_best else ''))
        
        if waited == cfg['patience']:
            break

    model.load_state_dict(best_state)

    return model

def to_student_sequences(df):
    seqs = defaultdict(lambda: {
        "obs" : [],
        "problem" : []
    })
    for r in df.itertuples():
        seqs[r.student]["obs"].append(r.correct)
        seqs[r.student]["problem"].append(r.problem)
    return seqs

def predict(model, seqs, n_batch_seqs, device):
    model.eval()
    seqs = sorted(seqs, key=lambda s: len(s), reverse=True)
    with th.no_grad():
        all_ypred = []
        all_ytrue = []
        for offset in range(0, len(seqs), n_batch_seqs):
            end = offset + n_batch_seqs
            batch_seqs = seqs[offset:end]

            batch_obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_problem_seqs = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0)
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1

            pC = model(batch_obs_seqs.to(device), batch_problem_seqs.to(device)).cpu()
                
            ypred = pC.flatten()
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

class ExpMovAvgModel(nn.Module):
    def __init__(self, n_problems, n_hidden):
        super(ExpMovAvgModel, self).__init__()
        
        self.n_problems = n_problems 
        self.problem_embd = nn.Embedding(n_problems, n_hidden)
        self.problem_lambda = nn.Linear(n_hidden, 1)

    def problem_sim_mat(self):

        mag = th.linalg.vector_norm(self.problem_embd.weight, dim=1) # P
        denom = mag[:,None] @ mag[None,:] # PxP
        x = self.problem_embd.weight # PxD
        sim_mat = x @ x.T / denom 
        return sim_mat

    def forward(self, y, problem_seq):
        """
            problem_seq: [n_batch, t]
            y: [n_batch, t]
        """

        # compute problem cosine similarity
        mag = th.linalg.vector_norm(self.problem_embd.weight, dim=1) # P
        mag = mag[problem_seq] # BxT
        denom = th.bmm(mag[:,:,None], mag[:,None,:]) # BxTx1 * Bx1xT = BxTxT
        
        x = self.problem_embd(problem_seq) # BxTxD
        sim_mat = th.bmm(x, th.permute(x, (0, 2, 1))) / denom # BxTxT
        
        # don't allow negatives
        sim_mat = (sim_mat + 1) / 2
        
        # instantiate mask and apply to similarity matrix
        mask = th.ones_like(sim_mat).tril(diagonal=-1)
        sim_mat = sim_mat * mask 
        
        # compute effective delta_t based on similarities
        rev_idx = th.arange(sim_mat.shape[1]-1, end=-1, step=-1)
        delta_t = sim_mat[:, :, rev_idx].cumsum(2)[:, :, rev_idx] # BxTxT
        
        # compute the attention weights BxTxT
        lam = th.exp(self.problem_lambda(x)) # BxTx1
        weight = sim_mat * th.exp(-lam * delta_t)
        
        # B x T x T
        weight_normed = weight / (weight.sum(dim=2, keepdims=True) + 1e-6)
        
        # Bx1xT * BxTxT = BxT
        yhat = (y[:,None,:] * weight_normed).sum(2)
        
        return th.clamp(yhat, 0.01, 0.99) 



if __name__ == "__main__":
    main()
