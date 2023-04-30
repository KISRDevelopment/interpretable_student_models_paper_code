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
import sklearn.metrics
def main(cfg_path, dataset_name, output_path):
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    splits = np.load("data/splits/%s.npy" % dataset_name)

    problems_to_skills = dict(zip(df['problem'], df['skill']))
    n_problems = np.max(df['problem']) + 1
    A = np.array([problems_to_skills[p] for p in range(n_problems)])
    cfg['ref_labels'] = A

    results_df, all_embdeddings = evaluate(cfg, df, splits)
    
    results_df.to_csv(output_path)

    embd_output_path = output_path.replace(".csv", ".embeddings.npy")
    np.save(embd_output_path, all_embdeddings)

def evaluate(cfg, df, splits, device='cuda:0'):
    n_problems = np.max(df['problem']) + 1
    seqs = to_student_sequences(df)
    
    all_ytrue = []
    all_ypred = []

    results = []
    all_params = defaultdict(list)

    all_embdeddings = []
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

        ytrue_test, ypred_test = predict(model, test_seqs, n_test_batch_seqs, cfg['n_batch_trials'], cfg['n_samples'], device)
        
        with th.no_grad():
            state_dict = model.state_dict()
            problem_embddings = state_dict['problem_embd.weight'].cpu()
            all_embdeddings.append(problem_embddings.numpy())
            
        run_result = metrics.calculate_metrics(ytrue_test, ypred_test)
        
        results.append(run_result)
        all_ytrue.extend(ytrue_test)
        all_ypred.extend(ypred_test)

    all_ytrue = np.array(all_ytrue)
    all_ypred = np.array(all_ypred)

    results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(splits.shape[0])])
    
    all_embdeddings = np.array(all_embdeddings)
    return results_df, all_embdeddings

def train(train_seqs, valid_seqs, n_problems, cfg, device):

    model = ExpMovAvgModel(n_problems, cfg['n_latent_kcs'], cfg['n_hidden'])
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

            obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0).to(device)
            problem_seqs = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0).to(device)
            mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1
            mask_seqs = mask_seqs.to(device)

            pC = th.zeros((len(batch_seqs), obs_seqs.shape[1])).to(device)

            A = model.sample_A(cfg['tau'], False)
            
            for i in range(0, obs_seqs.shape[1], cfg['n_batch_trials']):
                to_trial = i + cfg['n_batch_trials']
                batch_obs_seqs = obs_seqs[:, i:to_trial]
                batch_problem_seqs = problem_seqs[:, i:to_trial]
                pC[:, i:to_trial] = model(batch_obs_seqs, batch_problem_seqs, A)
            
            logpC = th.log(pC)
            logpnC = th.log(1 - pC)

            train_loss = -(obs_seqs * logpC + (1-obs_seqs) * logpnC).flatten()
            mask_ix = mask_seqs.flatten()

            train_loss = train_loss[mask_ix].mean()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            losses.append(train_loss.item())
        
        mean_train_loss = np.mean(losses)
        
        #
        # Validation
        #
        ytrue, ypred = predict(model, valid_seqs, n_valid_seqs, cfg['n_batch_trials'], cfg['n_samples'], device)

        auc_roc = metrics.calculate_metrics(ytrue, ypred)['auc_roc']

        #
        # compare to reference
        #
        rand_index = 0
        n_utilized_kcs = 0
        with th.no_grad():
            ref_labels = cfg['ref_labels']
            indecies = []
            n_utilized_kcs = []
            for s in range(cfg['n_samples']):
                A = model.sample_A(1e-6, True)
                n_utilized_kcs.append((A.sum(0) > 0).sum().cpu().numpy())
                if ref_labels is not None:
                    pred_labels = th.argmax(A, dim=1).cpu().numpy()
                    rand_index = sklearn.metrics.adjusted_rand_score(ref_labels, pred_labels)
                    indecies.append(rand_index)
            if ref_labels is not None:
                rand_index = np.mean(indecies)
            n_utilized_kcs = np.mean(n_utilized_kcs)


        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_state = copy.deepcopy(model.state_dict())
            waited = 0
            new_best = True 
        else:
            new_best = False 
            waited += 1

        print("%4d Train loss: %8.4f, Valid AUC: %0.2f (Rand: %0.2f, Utilized KCS: %d) %s" % (e, mean_train_loss, auc_roc, 
            rand_index,
            n_utilized_kcs,
            '***' if new_best else ''))
        
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

def predict(model, seqs, n_batch_seqs, n_batch_trials, n_samples, device):
    model.eval()
    seqs = sorted(seqs, key=lambda s: len(s), reverse=True)
    with th.no_grad():
        all_ypred = []
        for s in range(n_samples):
            sample_ypred = []
            all_ytrue = []

            A = model.sample_A(1e-6, True)
            
            for offset in range(0, len(seqs), n_batch_seqs):
                end = offset + n_batch_seqs
                batch_seqs = seqs[offset:end]

                obs_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=0).to(device)
                problem_seqs = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0).to(device)
                mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1
                mask_seqs = mask_seqs.to(device)

                pC = th.zeros((len(batch_seqs), obs_seqs.shape[1])).to(device)

                for i in range(0, obs_seqs.shape[1], n_batch_trials):
                    to_trial = i + n_batch_trials
                    batch_obs_seqs = obs_seqs[:, i:to_trial]
                    batch_problem_seqs = problem_seqs[:, i:to_trial]
                    pC[:, i:to_trial] = model(batch_obs_seqs, batch_problem_seqs, A)

                ypred = pC.flatten()
                ytrue = obs_seqs.flatten()
                mask_ix = mask_seqs.flatten()
                    
                ypred = ypred[mask_ix].cpu().numpy()
                ytrue = ytrue[mask_ix].cpu().numpy()

                sample_ypred.append(ypred)
                all_ytrue.append(ytrue)
            sample_ypred = np.hstack(sample_ypred)
            ytrue = np.hstack(all_ytrue)
            all_ypred.append(sample_ypred)
        ypred = np.mean(np.vstack(all_ypred), axis=0)
        
    model.train()

    return ytrue, ypred

class ExpMovAvgModel(nn.Module):
    def __init__(self, n_problems, n_latent_kcs, n_hidden):
        super(ExpMovAvgModel, self).__init__()
        
        self.n_problems = n_problems 
        self.n_latent_kcs = n_latent_kcs

        self.problem_embd = nn.Embedding(n_problems, n_hidden)
        self.kc_embd = nn.Embedding(n_latent_kcs, n_hidden)
        self.kc_guess = nn.Linear(n_hidden, 1)
        self.kc_slip = nn.Linear(n_hidden, 1)
        self.kc_lam = nn.Linear(n_hidden, 1)
        self.R = nn.Parameter(th.randn(n_hidden, n_hidden))
        self._hard = False
    def sample_A(self, tau, hard_samples):
        """
           
            tau: Gumbel-softmax temperature
            hard_samples: return hard samples or not
            Output:
                Assignment matrix P x Latent KCs
        """
        membership_logits = self._compute_membership_logits()

        # sample
        A = nn.functional.gumbel_softmax(membership_logits, hard=hard_samples, tau=tau, dim=1)
        self._hard = hard_samples
        return A 
    
    def _compute_membership_logits(self):

        problem_embeddings = self.problem_embd.weight # Pxn_hidden
        
        # compute logits
        membership_logits = (problem_embeddings @ self.R) @ self.kc_embd.weight.T  # Problems x Latent KCs
        
        return membership_logits

    def forward(self, y, problem_seq, A):
        """
            problem_seq: [n_batch, t]
            A: n_problems x n_latent_kcs
            y: [n_batch, t]
        """

        #
        # compute problem "similarity" based on KC membership
        #
        x = A[problem_seq,:] # BxTxn_kcs 
        sim_mat = th.bmm(x, th.permute(x, (0, 2, 1))) # BxTxT
        
        # instantiate mask and apply to similarity matrix
        mask = th.ones_like(sim_mat).tril(diagonal=-1)
        sim_mat = sim_mat * mask 
        
        # compute effective delta_t based on similarities
        rev_idx = th.arange(sim_mat.shape[1]-1, end=-1, step=-1)
        delta_t = sim_mat[:, :, rev_idx].cumsum(2)[:, :, rev_idx] # BxTxT
        
        #
        # compute the attention weights BxTxT
        #

        # compute KC representations
        kc_embd = x @ self.kc_embd.weight # BxTxlatent_dim
        kc_lam = th.exp(self.kc_lam(kc_embd)) # BxTx1
        weight = sim_mat * th.exp(-kc_lam * delta_t)
        
        # B x T x T
        weight_normed = weight / (weight.sum(dim=2, keepdims=True) + 1e-6)
        
        # Bx1xT * BxTxT = BxT
        h = (y[:,None,:] * weight_normed).sum(2)
        
        # BxT
        guess = th.sigmoid(self.kc_guess(kc_embd))[:,:,0]
        slip = th.sigmoid(self.kc_slip(kc_embd))[:,:,0]

        yhat = h * (1-slip) + (1-h) * guess 

        return th.clamp(yhat, 0.01, 0.99) 



if __name__ == "__main__":
    cfg_path = sys.argv[1]
    dataset_name = sys.argv[2]
    output_path = sys.argv[3]

    main(cfg_path, dataset_name, output_path)
