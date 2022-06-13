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
        self.dropout = nn.Dropout(0.5)

    def forward(self, cell_input, curr_kc, state=None):
        """
            input: [n_batch, t, 2*n_kcs]
            curr_kc: [n_batch, t, n_kcs]
        """
        
        # if state is None:
        #     assert cell_input[:,0,:].sum() == 0
        
        # # [n_batch, t, 2*n_kcs + 1]
        # cell_input = th.cat((cell_input, th.zeros((cell_input.shape[0], cell_input.shape[1],1))), dim=2)
        # if state is None:
        #     cell_input[:, 0, cell_input.shape[2]-1] = 1
        
        
        cell_output, last_state = self.cell(cell_input, state)
        cell_output = self.dropout(cell_output)

        kc_logits = self.ff(cell_output) # [n_batch, t, n_kcs]
        logits = (kc_logits * curr_kc).sum(dim=2) # [n_batch, t]
        
        return logits, last_state


def train(train_seqs, valid_seqs, n_kcs,
    n_hidden=100,
    epochs=100, 
    n_batch_seqs=50, 
    n_batch_trials=100, 
    learning_rate=1e-1,
    patience=10):
    
    loss_fn = nn.BCEWithLogitsLoss()

    model = DKTModel(n_kcs, n_hidden)
    model = model.cuda()

    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = np.inf 
    best_state = None 
    epochs_since_last_best = 0
    best_val_auc = 0.5

    for e in range(epochs):
        np.random.shuffle(train_seqs)
        losses = []

        model.train()

        for seqs, new_seqs in sequences.iterate_batched(train_seqs, n_batch_seqs, n_batch_trials):

            cell_input, curr_skill, curr_correct, mask = transform(seqs, n_kcs)
            
            cell_input = cell_input.cuda()
            curr_skill = curr_skill.cuda()

            if new_seqs:
                logits, state = model(cell_input, curr_skill)
            else:
                hn, cn = state[0].detach(), state[1].detach()

                # trim the state
                n_state_size = hn.shape[1]
                n_diff = n_state_size - len(seqs)

                if n_diff > 0:
                    hn = hn[:,n_diff:,:]
                    cn = cn[:,n_diff:,:]
                
                logits, state = model(cell_input, curr_skill, (hn, cn))
            
            logits = logits.cpu()

            logits = logits.view(-1)
            curr_correct = curr_correct.view(-1)
            mask = mask.view(-1).bool()

            logits = logits[mask]
            curr_correct = curr_correct[mask]

            loss = loss_fn(logits, curr_correct)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            
        model.eval()

        with th.no_grad():

            ytrue_valid, logit_ypred_valid = predict(model, valid_seqs)
            valid_loss = loss_fn(logit_ypred_valid, ytrue_valid)
            
            auc_roc = sklearn.metrics.roc_auc_score(ytrue_valid, logit_ypred_valid)
            print("%d Train loss: %0.4f, Valid loss: %0.4f, auc: %0.6f" % (e, np.mean(losses), valid_loss, auc_roc))

            if auc_roc > best_val_auc:
            #if valid_loss < best_val_loss:
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

def predict(model, test_seqs, n_batch_seqs=50, n_batch_trials=0):
    model.eval()
    with th.no_grad():

        all_logits = []
        all_labels = []

        for seqs, new_seqs in sequences.iterate_batched(test_seqs, n_batch_seqs, n_batch_trials):
            cell_input, curr_skill, curr_correct, mask = transform(seqs, model.n_kcs)
            
            cell_input = cell_input.cuda()
            curr_skill = curr_skill.cuda()

            if new_seqs:
                logits, state = model(cell_input, curr_skill)
            else:
                hn, cn = state
                # trim the state
                n_state_size = hn.shape[1]
                n_diff = n_state_size - len(seqs)
                if n_diff > 0:
                    hn = hn[:,n_diff:,:]
                    cn = cn[:,n_diff:,:]
                
                logits, state = model(cell_input, curr_skill, (hn, cn))
            
            logits = logits.cpu()
            
            logits = logits.flatten()
            curr_correct = curr_correct.flatten()
            mask = mask.flatten().bool()

            logits = logits[mask]
            curr_correct = curr_correct[mask]
            all_logits.append(logits)
            all_labels.append(curr_correct)
        
        all_logits = th.concat(all_logits)
        all_labels = th.concat(all_labels)

    return all_labels, all_logits

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

    train_df = pd.read_csv("/home/mmkhajah/Projects/learner-performance-prediction/data/statics/preprocessed_data_train.csv", sep="\t")
    train_students = list(set(train_df['user_id']))
    np.random.shuffle(train_students)

    n_valid = int(len(train_students) * 0.2)
    valid_students = set(train_students[:n_valid])
    train_students = set(train_students[n_valid:])


    test_df = pd.read_csv("/home/mmkhajah/Projects/learner-performance-prediction/data/statics/preprocessed_data_test.csv", sep="\t")
    test_students = set(test_df['user_id'])

    all_ytrue = []
    all_ypred = []
    for s in range(1):
        split = splits[s, :]

        # train_ix = split == 2
        # valid_ix = split == 1
        # test_ix = split == 0

        # train_df = df[train_ix]
        # valid_df = df[valid_ix]
        # test_df = df[test_ix]

        # train_students = set(train_df['student'])
        # valid_students = set(valid_df['student'])

        train_seqs = sequences.make_sequences(df, train_students)
        valid_seqs = sequences.make_sequences(df, valid_students)

        n_obs_kcs = int(np.max(df['skill']) + 1)
        model = train(train_seqs, valid_seqs, 
            n_kcs=n_obs_kcs, 
            n_hidden=200,
            learning_rate=0.01, 
            epochs=100, 
            patience=20,
            n_batch_seqs=100, 
            n_batch_trials=50)

        #test_students = set(test_df['student'])
        test_seqs = sequences.make_sequences(df, test_students)
        ytrue_test, logit_ypred_test = predict(model, test_seqs)
        
        all_ytrue.extend(ytrue_test.numpy())
        all_ypred.extend(logit_ypred_test.numpy())
    
    all_ytrue = np.array(all_ytrue)
    all_ypred = np.array(all_ypred)

    auc_roc = sklearn.metrics.roc_auc_score(all_ytrue, all_ypred)
    
    print("Test auc: %0.6f" % (auc_roc))


if __name__ == "__main__":
    main()
    