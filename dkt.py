import torch as th
import torch.nn as nn
import numpy as np
import dkt_sequences as sequences
import pandas as pd
import metrics
import copy 
import json 

class DKTModel(nn.Module):
    def __init__(self, n_kcs, n_hidden, n_kc_embd=10):
        super(DKTModel, self).__init__()
        
        self.n_kcs = n_kcs 
        self.kc_embd = nn.Linear(n_kcs, n_kc_embd)
        self.cell = nn.LSTM(n_kc_embd + 1, n_hidden, num_layers=1, batch_first=True)
        self.ff = nn.Linear(n_hidden, n_kcs)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input, curr_kc, state=None):
        """
            input: [n_batch, t, n_kcs + 1]
            curr_kc: [n_batch, t, n_kcs]
        """
        
        # if state is None:
        #     assert cell_input[:,0,:].sum() == 0
        
        # # [n_batch, t, 2*n_kcs + 1]
        # cell_input = th.cat((cell_input, th.zeros((cell_input.shape[0], cell_input.shape[1],1))), dim=2)
        # if state is None:
        #     cell_input[:, 0, cell_input.shape[2]-1] = 1
        
        kc_embd = self.kc_embd(input[:, :, :self.n_kcs]) # n_batch,t,n_kc_embd
        prev_output = input[:,:, [self.n_kcs]] # n_batch,t,1

        cell_input = th.concat((kc_embd, prev_output), dim=2)
        cell_output, last_state = self.cell(cell_input, state)
        #cell_output = self.dropout(cell_output)

        kc_logits = self.ff(cell_output) # [n_batch, t, n_kcs]
        logits = (kc_logits * curr_kc).sum(dim=2) # [n_batch, t]
        
        return logits, last_state


def train(train_seqs, valid_seqs, n_kcs,
    n_hidden=100,
    epochs=100, 
    n_batch_seqs=50, 
    n_batch_trials=100, 
    learning_rate=1e-1,
    patience=10,
    balanced_loss=False):
    
    model = DKTModel(n_kcs, n_hidden)
    model = model.cuda()

    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

    best_state = None 
    epochs_since_last_best = 0
    best_val_auc = 0.5

    for e in range(epochs):
        np.random.shuffle(train_seqs)
        losses = []

        model.train()

        for from_seq in range(0, len(train_seqs), n_batch_seqs):
            to_seq = from_seq + n_batch_seqs
            batch_seqs = train_seqs[from_seq:to_seq]
            batch_seqs = sequences.pad_to_max(batch_seqs)
            batch_seqs = sequences.make_prev_curr_sequences(batch_seqs)

            n_seqs = len(batch_seqs)
            n_trials = len(batch_seqs[0])

            batch_logits = th.zeros((n_seqs, n_trials)).cuda()
            batch_label = th.zeros((n_seqs, n_trials)).cuda()
            batch_mask = th.zeros((n_seqs, n_trials)).cuda()

            for i in range(0, n_trials, n_batch_trials):
                to_trial = i + n_batch_trials
                block = [s[i:to_trial] for s in batch_seqs]
                
                cell_input, curr_skill, curr_correct, mask = transform(block, n_kcs)
                
                cell_input = cell_input.cuda()
                curr_skill = curr_skill.cuda()
                
                if i == 0:
                    logits, state = model(cell_input, curr_skill)
                else:
                    hn, cn = state[0].detach(), state[1].detach()
                    logits, state = model(cell_input, curr_skill, (hn, cn))

                batch_logits[:, i:to_trial] = logits 
                batch_label[:, i:to_trial] = curr_correct
                batch_mask[:, i:to_trial] = mask 
        
            batch_logits = batch_logits.view(-1).cpu()
            batch_label = batch_label.view(-1).cpu()
            batch_mask = batch_mask.view(-1).bool().cpu()

            batch_logits = batch_logits[batch_mask]
            batch_label = batch_label[batch_mask]

            if balanced_loss:
                pos_weight = th.tensor([(batch_label==0).sum()/(batch_label==1).sum()])
            else:
                pos_weight = th.tensor([1])
                
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            loss = loss_fn(batch_logits, batch_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())


        model.eval()

        with th.no_grad():

            ytrue_valid, logit_ypred_valid = predict(model, valid_seqs, n_batch_trials=n_batch_trials*10)
            if balanced_loss:
                pos_weight = th.tensor([(ytrue_valid==0).sum()/(ytrue_valid==1).sum()])
            else:
                pos_weight = th.tensor([1])
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            valid_loss = loss_fn(logit_ypred_valid, ytrue_valid)
            
            m = metrics.calculate_metrics(ytrue_valid.numpy(), th.sigmoid(logit_ypred_valid).numpy())
            auc_roc = m['auc_roc']
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

def predict(model, test_seqs, n_batch_seqs=50, n_batch_trials=100000):
    model.eval()
    with th.no_grad():

        all_logits = []
        all_labels = []

        for seqs, new_seqs in sequences.iterate_batched(test_seqs, n_batch_seqs, n_batch_trials, pad_to_maximum=True):
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

    cell_input = np.zeros((n_batch, n_trials, n_kcs+1))
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
                cell_input[s, t, prev_skill] = 1
                cell_input[s, t, n_kcs] = prev_corr

            if curr_trial is not None:
                cs = curr_trial['skill']
                curr_skill[s, t, cs] = 1
                correct[s, t] = curr_trial['correct']
                included[s,t] = 1

    return th.tensor(cell_input).float(), th.tensor(curr_skill).float(), th.tensor(correct).float(), th.tensor(included).float()

def main(cfg_path, dataset_name, output_path):
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    


    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    df['skill'] = df['problem']
    splits = np.load("data/splits/%s.npy" % dataset_name)

    all_ytrue = []
    all_ypred = []
    all_embdeddings = []

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

        train_seqs = sequences.make_sequences(df, train_students)
        valid_seqs = sequences.make_sequences(df, valid_students)

        n_obs_kcs = int(np.max(df['skill']) + 1)
        model = train(train_seqs, valid_seqs, 
            n_kcs=n_obs_kcs, 
            **cfg)
        state_dict = model.state_dict()
        
        kc_embddings = state_dict['kc_embd.weight'].T.cpu()
        all_embdeddings.append(kc_embddings.numpy())

        test_students = set(test_df['student'])
        test_seqs = sequences.make_sequences(df, test_students)
        ytrue_test, logit_ypred_test = predict(model, test_seqs)
        
        ytrue_test = ytrue_test.numpy()
        ypred_test = th.sigmoid(logit_ypred_test).numpy()

        results.append(metrics.calculate_metrics(ytrue_test, ypred_test))
        all_ytrue.extend(ytrue_test)
        all_ypred.extend(ypred_test)

    all_embdeddings = np.array(all_embdeddings)

    all_ytrue = np.array(all_ytrue)
    all_ypred = np.array(all_ypred)

    overall_metrics = metrics.calculate_metrics(all_ytrue, all_ypred)
    results.append(overall_metrics)

    results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(splits.shape[0])] + ['Overall'])
    print(results_df)

    results_df.to_csv(output_path)

    params_output_path = output_path.replace('.csv', '.npy')
    np.save(params_output_path, all_embdeddings)

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])

    