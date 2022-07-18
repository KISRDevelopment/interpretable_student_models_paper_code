import numpy as np
import sklearn.metrics
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

def to_student_sequences(df):
    seqs = defaultdict(lambda: {
        "obs" : [],
        "kc" : []
    })
    for r in df.itertuples():
        seqs[r.student]["obs"].append(r.correct)
        seqs[r.student]["kc"].append(r.skill)
    return seqs

def train(train_seqs, valid_seqs, n_kcs, device, learning_rate, epochs, patience, n_batch_seqs):

    model = BktModel(n_kcs)
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
            batch_mask_seqs = pad_sequence([th.tensor(s['obs']) for s in batch_seqs], batch_first=True, padding_value=-1) > -1

            output = model(batch_obs_seqs.to(device), batch_kc_seqs.to(device)).cpu()
            
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

        auc_roc = sklearn.metrics.roc_auc_score(ytrue, ypred)
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

            output = model(batch_obs_seqs.to(device), batch_kc_seqs.to(device)).cpu()
                
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
    seqs = to_student_sequences(df)
    
    splits = np.load("data/splits/gervetetal_statics.npy")
    split = splits[0, :]

    train_ix = split == 2
    valid_ix = split == 1
    test_ix = split == 0

    train_students = set(df[train_ix]['student'])
    valid_students = set(df[valid_ix]['student'])
    test_students = set(df[test_ix]['student'])

    train_seqs = [seqs[s] for s in train_students]
    valid_seqs = [seqs[s] for s in valid_students]
    test_seqs = [seqs[s] for s in test_students]

    n_kcs = int(np.max(df['skill']) + 1)
    model = train(train_seqs, 
                  valid_seqs,
                  n_kcs,
                  device='cuda:0', 
                  learning_rate=0.5, 
                  epochs=1, 
                  patience=10,
                  n_batch_seqs=500)

    ytrue_test, ypred_test = predict(model, test_seqs, 500, 'cuda:0')
    auc_roc = sklearn.metrics.roc_auc_score(ytrue_test, ypred_test)

    print("Test auc: %0.2f" % (auc_roc))

if __name__ == "__main__":
    main()