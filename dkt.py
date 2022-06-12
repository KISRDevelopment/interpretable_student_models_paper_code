from socket import TIPC_HIGH_IMPORTANCE
import torch as th
import torch.nn as nn
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor
import numpy as np
import sequences
import pandas as pd 
import sklearn.metrics
import copy 

class DKTModel(nn.Module):
    def __init__(self, n_kcs, n_hidden):
        super(DKTModel, self).__init__()
        
        self.n_kcs = n_kcs 
        self.cell = nn.LSTM(2 * n_kcs, n_hidden, num_layers=1, batch_first=True)
        self.ff = nn.Linear(n_hidden, n_kcs)
        self.dropout = nn.Dropout(0)

    def forward(self, cell_input, curr_kc):
        """
            input: [n_batch, t, 2*n_kcs]
            curr_kc: [n_batch, t, n_kcs]
        """
        
        cell_output, last_state = self.cell(cell_input)
        cell_output = self.dropout(cell_output)

        kc_probs = th.sigmoid(self.ff(cell_output)) # [n_batch, t, n_kcs]
        probs = (kc_probs * curr_kc).sum(dim=2) # [n_batch, t]
        
        return probs, last_state

    def forward_from_state(self, cell_input, curr_kc, state):
        """
            input: [n_batch, t, 2*n_kcs]
            curr_kc: [n_batch, t, n_kcs]
            state: ([n_layers, n_batch, n_hidden], [n_layers, n_batch, n_hidden])
        """
        
        cell_output, last_state = self.cell(cell_input, state)
        cell_output = self.dropout(cell_output)
        
        kc_probs = th.sigmoid(self.ff(cell_output)) # [n_batch, t, n_kcs]
        probs = (kc_probs * curr_kc).sum(dim=2) # [n_batch, t]
        
        return probs, last_state


def train(train_seqs, valid_seqs, n_kcs,
    n_hidden=100,
    epochs=100, 
    n_batch_seqs=50, 
    n_batch_trials=100, 
    learning_rate=1e-1,
    patience=10):
    
    loss_fn = nn.BCELoss(reduction='none')

    model = DKTModel(n_kcs, n_hidden)

    optimizer = th.optim.NAdam(model.parameters(), lr=learning_rate)

    best_val_auc = 0.5 
    best_val_loss = np.inf 
    best_state = None 
    epochs_since_last_best = 0

    for e in range(epochs):
        np.random.shuffle(train_seqs)
        losses = []

        for seqs, new_seqs in sequences.iterate_batched(train_seqs, n_batch_seqs, n_batch_trials):

            cell_input, curr_skill, curr_correct, mask = transform(seqs, n_kcs)
            
            if new_seqs:
                probs, state = model(cell_input, curr_skill)
            else:
                hn, cn = state[0].detach(), state[1].detach()

                # trim the state
                n_state_size = hn.shape[1]
                n_diff = n_state_size - len(seqs)

                if n_diff > 0:
                    hn = hn[:,n_diff:,:]
                    cn = cn[:,n_diff:,:]
                
                probs, state = model.forward_from_state(cell_input, curr_skill, (hn, cn))
            
            loss = loss_fn(probs, curr_correct)
            loss = loss * mask
            loss = loss.sum() / mask.sum()

            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with th.no_grad():
            ytrue_valid, ypred_valid = predict(model, valid_seqs)
            valid_loss = loss_fn(th.tensor(ypred_valid), th.tensor(ytrue_valid)).mean().numpy()
            
            auc_roc = sklearn.metrics.roc_auc_score(ytrue_valid, ypred_valid)
            print("%d Train loss: %0.4f, Valid loss: %0.4f, auc: %0.2f" % (e, np.mean(losses), valid_loss, auc_roc))

            
            #if auc_roc > best_val_auc:
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_auc = auc_roc
                best_state = copy.deepcopy(model.state_dict())
                epochs_since_last_best = 0
            else:
                epochs_since_last_best += 1

            if epochs_since_last_best >= patience:
                break

    model.load_state_dict(best_state)
    return model 

def predict(model, test_seqs, n_batch_seqs=50, n_batch_trials=100):

    with th.no_grad():
        final_probs = []

        all_probs = []
        all_labels = []

        for seqs, new_seqs in sequences.iterate_batched(test_seqs, n_batch_seqs, n_batch_trials):
            cell_input, curr_skill, curr_correct, mask = transform(seqs, model.n_kcs)
            
            if new_seqs:
                probs, state = model(cell_input, curr_skill)
            else:
                hn, cn = state
                # trim the state
                n_state_size = hn.shape[1]
                n_diff = n_state_size - len(seqs)
                if n_diff > 0:
                    hn = hn[:,n_diff:,:]
                    cn = cn[:,n_diff:,:]
                
                probs, state = model.forward_from_state(cell_input, curr_skill, (hn, cn))
            
            probs = probs.flatten()
            curr_correct = curr_correct.flatten()
            mask = mask.flatten().bool()

            probs = probs[mask]
            curr_correct = curr_correct[mask]
            all_probs.append(probs)
            all_labels.append(curr_correct)
        
        all_probs = th.concat(all_probs).numpy()
        all_labels = th.concat(all_labels).numpy()

    return all_labels, all_probs

def transform(subseqs, n_kcs):
    n_batch = len(subseqs)
    n_trials = len(subseqs[0])

    cell_input = np.zeros((n_batch, n_trials, 2*n_kcs))
    curr_skill = np.zeros((n_batch, n_trials, n_kcs))
    correct = np.zeros((n_batch, n_trials), dtype=int)
    included = np.zeros((n_batch, n_trials), dtype=int)
    
    for s, seq in enumerate(subseqs):
        for t, elm in enumerate(seq):
            
            prev_trial = elm[0]
            curr_trial = elm[1]

            if prev_trial is not None:
                prev_skill = prev_trial['skill']
                prev_corr = prev_trial['correct']
                cell_input[s, t, prev_corr*n_kcs + prev_skill] = 1

            if curr_trial is not None:
                cs = curr_trial['skill']
                curr_skill[s, t, cs] = 1
                correct[s, t] = curr_trial['correct']
                included[s,t] = 1

    return th.tensor(cell_input).float(), th.tensor(curr_skill).float(), th.tensor(correct).float(), th.tensor(included).float()

def main():
    df = pd.read_csv("data/datasets/gervetetal_statics.csv")
    splits = np.load("data/splits/gervetetal_statics.npy")

    all_ytrue = []
    all_ypred = []
    for s in range(1):
        split = splits[s, :]

        train_ix = split == 2
        valid_ix = split == 1
        test_ix = split == 0

        train_df = df[train_ix]
        valid_df = df[valid_ix]
        test_df = df[test_ix]

        train_students = set(train_df['student'])
        valid_students = set(valid_df['student'])

        train_seqs = sequences.make_sequences(df, train_students)
        valid_seqs = sequences.make_sequences(df, valid_students)

        n_obs_kcs = int(np.max(df['skill']) + 1)
        model = train(train_seqs, valid_seqs, 
            n_kcs=n_obs_kcs, 
            n_hidden=200,
            learning_rate=0.001, 
            epochs=100, 
            patience=5,
            n_batch_seqs=100, 
            n_batch_trials=50)

        test_students = set(test_df['student'])
        test_seqs = sequences.make_sequences(df, test_students)
        ytrue_test, ypred_test = predict(model, test_seqs)
        
        all_ytrue.extend(ytrue_test)
        all_ypred.extend(ypred_test)
    
    all_ytrue = np.array(all_ytrue)
    all_ypred = np.array(all_ypred)

    auc_roc = sklearn.metrics.roc_auc_score(all_ytrue, all_ypred)
    test_loss = -np.mean(all_ytrue * np.log(all_ypred) + (1-all_ytrue) * np.log(1-all_ypred))

    print("Test loss: %0.4f, auc: %0.2f" % (test_loss, auc_roc))


if __name__ == "__main__":
    main()
    