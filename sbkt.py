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
class BKTCell(jit.ScriptModule):
    def __init__(self, n_kcs, device='cpu'):
        super(BKTCell, self).__init__()

        self.n_kcs = n_kcs
        self.device = device
        self.kc_logits = nn.Embedding(n_kcs, 5, device=device) # learning, forgetting, guessing, and slipping, initial logits
        
    @jit.script_method
    def forward(self, input: Tuple[Tensor, Tensor, Tensor], state: Tensor) -> Tuple[Tensor, Tensor]:
        """
            state: p(h_(t-1) = 1| y_1, ..., y_(t-2)) [n_batch, n_kcs]
            input: 
                previous KC [n_batch]
                current KC [n_batch]
                previous correct [n_batch]
        """
        
        prev_kc, curr_kc, prev_corr = input 
        
        batch_ix = th.arange(prev_kc.shape[0])
        
        prev_logits = self.kc_logits(prev_kc) # [n_batch, 5]
        prev_probs = th.sigmoid(prev_logits) # [n_batch, 5]
        curr_logits = self.kc_logits(curr_kc) #[n_batch, 5]
        curr_probs = th.sigmoid(curr_logits)
        
        # compute probability of correctness given h (0 or 1)
        p_correct_given_h = prev_probs[:, [2, 3]] # [n_batch, 2]
        
        # compute probability of previous steps' output given h [n_batch, 2]
        p_output_given_h = th.pow(p_correct_given_h, prev_corr[:,None]) * th.pow(1-p_correct_given_h, 1-prev_corr[:,None])
        
        # compute filtering distribution p(h_(t-1) = 1 | y_1 .. y_(t-1))
        skill_state = state[batch_ix, prev_kc] # [n_batch]
        # [n_batch]
        filtering = (p_output_given_h[:,1] * skill_state) / (p_output_given_h[:,0]*(1-skill_state) + p_output_given_h[:,1]*skill_state)
        
        # compute predictive distribution p(h_t=1|y_1...y_(t-1))
        p_learning = prev_probs[:,0]
        p_forgetting = prev_probs[:,1]
        predictive = p_learning * (1-filtering) + (1-p_forgetting) * filtering
        
        # update relevant entry
        state[batch_ix,prev_kc] = predictive
        
        # grab  predictive state at current time step
        curr_state = state[batch_ix, curr_kc] #[n_batch]
        
        p_correct_given_curr_h = curr_probs[:, [2, 3]] # [n_batch, 2]
        p_curr_correct = p_correct_given_curr_h[:,0] * (1-curr_state) + p_correct_given_curr_h[:,1] * curr_state
        
        return p_curr_correct, state
    
    @jit.script_method
    def forward_first_trial(self, curr_kc: Tensor) -> Tuple[Tensor, Tensor]:
        """
            curr_kc: [n_batch]
        """
        logits = self.kc_logits(th.arange(self.n_kcs, device=self.device))
        
        state = th.tile(th.sigmoid(logits[:,[4]].T), (curr_kc.shape[0], 1)) # [n_batch, n_kcs]
        
        return self.forward_first_trial_from_state(curr_kc, state)
    
    @jit.script_method
    def forward_first_trial_from_state(self, curr_kc: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
            curr_kc: [n_batch]
        """
        batch_ix = th.arange(curr_kc.shape[0], device=self.device)
        
        logits = self.kc_logits(th.arange(self.n_kcs, device=self.device))
        
        curr_logits = self.kc_logits(curr_kc) #[n_batch, 5]
        curr_probs = th.sigmoid(curr_logits)
        
        curr_state = state[batch_ix, curr_kc] #[n_batch]
        
        p_correct_given_curr_h = curr_probs[:, [2, 3]] # [n_batch, 2]
        p_curr_correct = p_correct_given_curr_h[:,0] * (1-curr_state) + p_correct_given_curr_h[:,1] * curr_state
        
        return p_curr_correct, state 
    
class BKTLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(BKTLayer, self).__init__()
        self.cell = cell(*cell_args)
        
    @jit.script_method
    def forward(self, prev_kc: Tensor, curr_kc: Tensor, prev_corr: Tensor) -> Tuple[Tensor, Tensor]:
        """
            prev_kc: [n_batch, t]
            curr_kc: [n_batch, t]
            prev_corr: [n_batch, t]
        """
        outputs = th.jit.annotate(List[Tensor], [])
        
        pc, state = self.cell.forward_first_trial(curr_kc[:,0])
        outputs += [pc]
        
        for i in range(1, prev_kc.shape[1]):
            pc, state = self.cell((prev_kc[:,i], curr_kc[:, i], prev_corr[:,i]), state)
            
            outputs += [pc]
        return th.stack(outputs).T, state

    @jit.script_method
    def forward_from_state(self, prev_kc: Tensor, curr_kc: Tensor, prev_corr: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
            prev_kc: [n_batch, t]
            curr_kc: [n_batch, t]
            prev_corr: [n_batch, t]
            state: [n_batch, n_kcs]
        """
        outputs = th.jit.annotate(List[Tensor], [])
        
        pc, state = self.cell.forward_first_trial_from_state(curr_kc[:,0], state)
        outputs += [pc]
        
        for i in range(1, prev_kc.shape[1]):
            pc, state = self.cell((prev_kc[:,i], curr_kc[:, i], prev_corr[:,i]), state)
            
            outputs += [pc]
        return th.stack(outputs).T, state

class BKTModel(nn.Module):
    def __init__(self, n_kcs, device='cpu'):
        super(BKTModel, self).__init__()
        
        self.bktlayer = BKTLayer(BKTCell, n_kcs, device)

    def forward(self, prev_kc, curr_kc, prev_corr):
        probs, state = self.bktlayer(prev_kc, curr_kc, prev_corr)
        
        return probs, state

    def forward_from_state(self, prev_kc, curr_kc, prev_corr, state):
        probs, state = self.bktlayer.forward_from_state(prev_kc, curr_kc, prev_corr, state)
        
        return probs, state

def train(train_seqs, valid_seqs, n_kcs, 
    epochs=100, 
    n_batch_seqs=50, 
    n_batch_trials=100, 
    learning_rate=1e-1,
    patience=50,
    device='cpu'):
    
    loss_fn = nn.BCELoss(reduction='none')

    model = BKTModel(n_kcs, device)

    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        np.random.shuffle(train_seqs)
        print("Epoch %d" % e, end=' ')
        losses = []

        for seqs, new_seqs in sequences.iterate_batched(train_seqs, n_batch_seqs, n_batch_trials):

            curr_skill, curr_correct, mask, _ = transform(seqs)
            prev_skill, prev_correct, _, _ = transform(seqs, prev_trial=True)
            

            if new_seqs:
                prev_skill = prev_skill.to(device)
                curr_skill = curr_skill.to(device)
                prev_correct = prev_correct.to(device)

                probs, state = model(prev_skill, curr_skill, prev_correct)
            else:
                state = state.detach()

                # trim the state
                n_state_size = state.shape[0]
                n_diff = n_state_size - len(seqs)
                if n_diff > 0:
                    state = state[n_diff:,:]

                prev_skill = prev_skill.to(device)
                curr_skill = curr_skill.to(device)
                prev_correct = prev_correct.to(device)
                state = state.to(device)

                probs, state = model.forward_from_state(prev_skill, curr_skill,  prev_correct, state)
            
            curr_correct = curr_correct.to(device)
            mask = mask.to(device)
            loss = loss_fn(probs, curr_correct)
            loss = loss * mask
            loss = loss.sum() / mask.sum()

            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print("Loss: %f"  % np.mean(losses))

        predict(model, valid_seqs, device=device)

def predict(model, test_seqs, n_batch_seqs=50, n_batch_trials=100, device='cpu'):

    with th.no_grad():
        all_probs = []
        all_labels = []
        for seqs, new_seqs in sequences.iterate_batched(test_seqs, n_batch_seqs, n_batch_trials):
            curr_skill, curr_correct, mask, trial_index = transform(seqs)
            prev_skill, prev_correct, _, _ = transform(seqs, prev_trial=True)
            
            if new_seqs:
                prev_skill = prev_skill.to(device)
                curr_skill = curr_skill.to(device)
                prev_correct = prev_correct.to(device)

                probs, state = model(prev_skill, curr_skill, prev_correct)
            else:
                # trim the state
                n_state_size = state.shape[0]
                n_diff = n_state_size - len(seqs)
                if n_diff > 0:
                    state = state[n_diff:,:]

                prev_skill = prev_skill.to(device)
                curr_skill = curr_skill.to(device)
                prev_correct = prev_correct.to(device)
                state = state.to(device)

                probs, state = model.forward_from_state(prev_skill, curr_skill,  prev_correct, state)
            
            probs = probs.cpu()
            
            probs = probs.flatten()
            curr_correct = curr_correct.flatten()
            mask = mask.flatten().bool()

            probs = probs[mask]
            curr_correct = curr_correct[mask]
            all_probs.append(probs)
            all_labels.append(curr_correct)
        
        all_probs = th.concat(all_probs).numpy()
        all_labels = th.concat(all_labels).numpy()
        
        auc_roc = sklearn.metrics.roc_auc_score(all_labels, all_probs)
        print("Valid auc: %0.2f" % auc_roc)



def transform(subseqs, prev_trial=False):
    n_batch = len(subseqs)
    n_trials = len(subseqs[0])

    correct = np.zeros((n_batch, n_trials), dtype=int)
    skill = np.zeros((n_batch, n_trials), dtype=int)
    included = np.zeros((n_batch, n_trials), dtype=int)
    trial_index = np.zeros((n_batch, n_trials), dtype=int)

    tuple_idx = 0 if prev_trial else 1 
    for s, seq in enumerate(subseqs):
        for t, elm in enumerate(seq):
            
            trial = elm[tuple_idx]
            if trial is None:
                correct[s, t] = 0
                included[s, t] = False 
                trial_index[s, t] = -1
            else:
                correct[s, t] = trial['correct']
                skill[s, t] = trial['skill']
                included[s, t] = True 
                trial_index[s, t] = trial['__index__']
    
    return th.tensor(skill), th.tensor(correct).float(), th.tensor(included).float(), trial_index

def main():
    df = pd.read_csv("data/datasets/assistment2009c.csv")
    splits = np.load("data/splits/assistment2009c.npy")
    split = splits[0, :]

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

    n_kcs = int(np.max(df['skill']) + 1)

    train(train_seqs, valid_seqs, n_kcs, device='cpu', learning_rate=0.01, epochs=10)

if __name__ == "__main__":
    main()
    