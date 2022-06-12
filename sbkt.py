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
import torch.distributions as D

class BKTCell(jit.ScriptModule):
    def __init__(self, n_kcs, n_obs_kcs, n_abilities):
        super(BKTCell, self).__init__()

        self.n_kcs = n_kcs
        self.n_obs_kcs = n_obs_kcs

        self.kc_logits = nn.Embedding(n_kcs, 5) # learning, forgetting, guessing, and slipping, initial logits
        nn.init.normal_(self.kc_logits.weight, 0, 0.1)

        self.abilities = th.linspace(-3, 3, n_abilities)
        
        # student abilities prior
        self.n_components = 5
        self.component_weights = nn.Parameter(th.randn(self.n_components))
        self.component_mu = nn.Parameter(th.randn(self.n_components))
        self.component_log_var = nn.Parameter(th.randn(self.n_components))
        
    @jit.script_method
    def forward(self, prev_kc: Tensor,          # previous KC [n_batch] (n_obs_kcs)
                      curr_kc: Tensor,          # current KC [n_batch] (n_obs_kcs)
                      prev_corr: Tensor,        # previous correct [n_batch] (n_obs_kcs)
                      A: Tensor,                # assignment matrix [n_obs_kcs, n_kcs]
                      full_state: Tuple[Tensor, Tensor, Tensor] 
                            # state: predictive state distribution p(h_(t-1) = 1|y_1,...,y_(t-2),a_s) [n_batch, n_obs_kcs, n_abilities]
                            # ability_state: unnormed log posterior over abilities log(p(y_1,...,y_(t-2),a_s)) [n_batch, n_abilities],
                            # probability correct answer on previous trial p(y_(t-1)=1|y_1...(t-2),a_s) [n_batch, n_abilities]
                ) -> Tuple[Tensor, Tensor, Tensor]:
        
        state, ability_state, prev_prob_corr_given_alpha = full_state

        #
        # update the unnormed posterior over student abilities
        #

        # compute loglikelihood of previous trial
        prev_trial_logprob = prev_corr[:,None] * th.log(prev_prob_corr_given_alpha) + (1-prev_corr[:,None]) * th.log(1-prev_prob_corr_given_alpha)

        # perform update, ability_state is now: log(p(y_1,...,y_(t-1), a_s))
        ability_state = prev_trial_logprob + ability_state

        #
        # update hmm state
        #
        batch_ix = th.arange(prev_kc.shape[0])
        
        prev_A = A[prev_kc,:] # [n_batch, n_kcs]
        prev_logits = th.matmul(prev_A, self.kc_logits.weight)# [n_batch, 5]
        
        curr_A = A[curr_kc, :] # [n_batch, n_kcs]
        curr_logits = th.matmul(curr_A, self.kc_logits.weight) # [n_batch, 5]
        
        # compute probability of correctness given h
        batch_abilities = th.tile(self.abilities, (prev_kc.shape[0],1)) # [n_batch, n_abilities]
        p_correct_given_h0 = th.sigmoid(prev_logits[:, [2]] + batch_abilities) # [n_batch, n_abilities]
        p_correct_given_h1 = th.sigmoid(prev_logits[:, [3]] + batch_abilities) # [n_batch, n_abilities]

        # compute probability of previous steps' output given h [n_batch, n_abilities]
        p_output_given_h0 = th.pow(p_correct_given_h0, prev_corr[:,None]) * th.pow(1-p_correct_given_h0, 1-prev_corr[:,None])
        p_output_given_h1 = th.pow(p_correct_given_h1, prev_corr[:,None]) * th.pow(1-p_correct_given_h1, 1-prev_corr[:,None])
        
        # compute filtering distribution p(h_(t-1) = 1 | y_1 .. y_(t-1),a_s)
        skill_state = state[batch_ix, prev_kc, :] # [n_batch, n_abilities]

        # [n_batch, n_abilities]
        filtering = (p_output_given_h1 * skill_state) / \
            (p_output_given_h0*(1-skill_state) + p_output_given_h1*skill_state)
        
        # compute predictive distribution p(h_t=1|y_1...y_(t-1), a_s)
        p_learning = th.sigmoid(prev_logits[:,0,None])
        p_forgetting = th.sigmoid(prev_logits[:,1,None])
        predictive = p_learning * (1-filtering) + (1-p_forgetting) * filtering # [n_batch, n_abilities]
        
        # update relevant entries
        states_to_update = th.matmul(prev_A, A.T)[:,:,None] # [n_batch, n_obs_kcs, 1]

        # [n_batch, n_obs_kcs, n_abilities]
        state = state * (1-states_to_update) + states_to_update * predictive[:,None,:]
        
        # grab  predictive state at current time step
        curr_state = state[batch_ix, curr_kc, :] # [n_batch, n_abilities]
        
        p_correct_given_curr_h0 = th.sigmoid(curr_logits[:, [2]] + batch_abilities)
        p_correct_given_curr_h1 = th.sigmoid(curr_logits[:, [3]] + batch_abilities)
        
        # [n_batch, n_abilities]
        p_curr_correct_given_alpha = p_correct_given_curr_h0 * (1-curr_state) + p_correct_given_curr_h1 * curr_state
        
        return state, ability_state, p_curr_correct_given_alpha
    
    @jit.script_method
    def forward_first_trial(self, curr_kc: Tensor, A: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
            curr_kc: [n_b+atch]
            A: [n_obs_kcs, n_kcs]
        """
        logits = th.matmul(A, self.kc_logits.weight) # [n_obs_kcs, 5]

        state = th.tile(th.sigmoid(logits[:,4]), (curr_kc.shape[0], 1))[:,:,None] # [n_batch, n_kcs, 1]
        state = th.tile(state, (self.abilities.shape[0],)) # [n_batch, n_kcs, n_abilities]
        
        # initialize abilities prior log(a_s)
        abilities_prior = self.gmm_logpdf(self.abilities, self.component_weights, self.component_mu, self.component_log_var)
        ability_state = th.tile(abilities_prior, (curr_kc.shape[0],1)) # [n_batch, n_abilities]

        return self.forward_first_trial_from_state(curr_kc, A, state, ability_state)
    
    @jit.script_method
    def forward_first_trial_from_state(self, curr_kc: Tensor, A: Tensor, state: Tensor, ability_state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
            curr_kc: [n_batch] (n_obs_kcs)
            state: [n_batch, n_obs_kcs]
            ability_state: [n_batch,n_abilities]
        """
        batch_ix = th.arange(curr_kc.shape[0])
        
        logits = th.matmul(A, self.kc_logits.weight) # [n_obs_kcs, 5]

        curr_logits = logits[curr_kc,:] # [n_batch, 5]
        
        curr_state = state[batch_ix, curr_kc, :] # [n_batch, n_abilities]
        
        batch_abilities = th.tile(self.abilities, (curr_kc.shape[0], 1)) # [n_batch, n_abilities]

        p_correct_given_curr_h0 = th.sigmoid(curr_logits[:, [2]] + batch_abilities)  # [n_batch, n_abilities]
        p_correct_given_curr_h1 = th.sigmoid(curr_logits[:, [3]] + batch_abilities)  # [n_batch, n_abilities]

        # p(y1|a_s)
        # [n_batch, n_abilities]
        p_curr_correct_given_alpha = p_correct_given_curr_h0 * (1-curr_state) + p_correct_given_curr_h1 * curr_state
        
        # marginalize out abilities [n_batch,]
        #p_curr_correct = (p_curr_correct_given_alpha * abilities_state).sum(1)

        return state, ability_state, p_curr_correct_given_alpha
    
    @jit.script_method
    def gmm_logpdf(self, x, weights_unnormed, mus, log_vars):
        """
            x: [n,]
            weights_unnormed, mus, log_stds: [m,]
        """
        
        x = th.tile(x, (weights_unnormed.shape[0],1)) # [m, n]

        weights = th.softmax(weights_unnormed, dim=0)[:,None]
        mus = mus[:,None]
        dvars = th.exp(log_vars[:,None])
        
        # [m, n]
        logpdf = 0.5 * th.square(x - mus) / dvars - th.log(th.sqrt(2 * th.pi * dvars))  
        logpdf = logpdf + th.log(weights)

        return th.logsumexp(logpdf, dim=0) 

class BKTLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(BKTLayer, self).__init__()
        self.cell = cell(*cell_args)
        
        self.kc_membership_logits = nn.Embedding(self.cell.n_obs_kcs, self.cell.n_kcs)
        nn.init.xavier_normal_(self.kc_membership_logits.weight)

        if self.cell.n_kcs == self.cell.n_obs_kcs:
            self._A = th.eye(self.cell.n_kcs)
        else:
            self._A = None 

    @jit.script_method
    def sample_A(self, tau: float):
        # for when n_kcs = n_obs_kcs
        if self._A is not None:
            return self._A 
        
        return nn.functional.gumbel_softmax(self.kc_membership_logits.weight, hard=True, tau=tau, dim=1)

    @jit.script_method
    def forward(self, prev_kc: Tensor, curr_kc: Tensor, prev_corr: Tensor, A: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
            prev_kc: [n_batch, t]
            curr_kc: [n_batch, t]
            prev_corr: [n_batch, t]
        """
        outputs = th.jit.annotate(List[Tensor], [])
        
        state, ability_state, pc_given_alpha = self.cell.forward_first_trial(curr_kc[:,0], A)
        pc = (th.softmax(ability_state,dim=1) * pc_given_alpha).sum(1)
        outputs += [pc]
        
        for i in range(1, prev_kc.shape[1]):
            state, ability_state, pc_given_alpha = self.cell(prev_kc[:,i], curr_kc[:, i], prev_corr[:,i], A, (state, ability_state, pc_given_alpha))
            pc = (th.softmax(ability_state,dim=1) * pc_given_alpha).sum(1)
            outputs += [pc]
        
        outputs = th.stack(outputs)
        outputs = th.transpose(outputs, 0, 1)
        
        return outputs, (state, ability_state, pc_given_alpha)

    @jit.script_method
    def forward_from_state(self, prev_kc: Tensor, curr_kc: Tensor, prev_corr: Tensor, A: Tensor, full_state: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
            prev_kc: [n_batch, t]
            curr_kc: [n_batch, t]
            prev_corr: [n_batch, t]
        """
        outputs = th.jit.annotate(List[Tensor], [])
        
        state, ability_state, pc_given_alpha = self.cell(prev_kc[:,0], curr_kc[:,0], prev_corr[:,0], A, full_state)
        pc = (th.softmax(ability_state,dim=1) * pc_given_alpha).sum(1)
        outputs += [pc]
        
        for i in range(1, prev_kc.shape[1]):
            state, ability_state, pc_given_alpha = self.cell(prev_kc[:,i], curr_kc[:, i], prev_corr[:,i], A, (state, ability_state, pc_given_alpha))
            pc = (th.softmax(ability_state,dim=1) * pc_given_alpha).sum(1)
            outputs += [pc]
        
        outputs = th.stack(outputs)
        outputs = th.transpose(outputs, 0, 1)
        
        return outputs, (state, ability_state, pc_given_alpha)

class BKTModel(nn.Module):
    def __init__(self, n_kcs, n_obs_kcs):
        super(BKTModel, self).__init__()
        
        self.bktlayer = BKTLayer(BKTCell, n_kcs, n_obs_kcs, 30)
        self._A = None 

    def sample_assignment(self, tau):
        self._A = self.bktlayer.sample_A(tau)

    def forward(self, prev_kc, curr_kc, prev_corr):

        return self.bktlayer(prev_kc, curr_kc, prev_corr, self._A)

    def forward_from_state(self, prev_kc, curr_kc, prev_corr, state):
        return self.bktlayer.forward_from_state(prev_kc, curr_kc, prev_corr, self._A, state)
        
def train(train_seqs, valid_seqs, n_obs_kcs, 
    n_kcs=5,
    epochs=100, 
    n_batch_seqs=50, 
    n_batch_trials=100, 
    learning_rate=1e-1,
    patience=10,
    tau=1.5,
    n_valid_samples=20,
    device='cpu'):
    print("Obs KCS : %d -> Core KCS: %d" % (n_obs_kcs, n_kcs))
    
    loss_fn = nn.BCELoss(reduction='none')

    model = BKTModel(n_kcs, n_obs_kcs)
    model.to(device)
    
    optimizer = th.optim.NAdam(model.parameters(), lr=learning_rate)
    #scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=1e-2, threshold_mode='abs', verbose=True)
    best_val_loss = np.inf 
    best_state = None 
    epochs_since_last_best = 0

    for e in range(epochs):
        np.random.shuffle(train_seqs)
        losses = []

        for seqs, new_seqs in sequences.iterate_batched(train_seqs, n_batch_seqs, n_batch_trials):

            curr_skill, curr_correct, mask, _ = transform(seqs)
            prev_skill, prev_correct, _, _ = transform(seqs, prev_trial=True)
            
            if new_seqs:
                prev_skill = prev_skill.to(device)
                curr_skill = curr_skill.to(device)
                prev_correct = prev_correct.to(device)

                model.sample_assignment(tau)
                probs, full_state = model(prev_skill, curr_skill, prev_correct)
            else:
                
                state, ability_state, pc_given_alpha  = full_state
                state = state.detach()
                ability_state = ability_state.detach()
                pc_given_alpha = pc_given_alpha.detach()

                # trim the state
                n_state_size = state.shape[0]
                n_diff = n_state_size - len(seqs)
                if n_diff > 0:
                    state = state[n_diff:,:]
                    ability_state = ability_state[n_diff:,:]
                    pc_given_alpha = pc_given_alpha[n_diff:,:]

                prev_skill = prev_skill.to(device)
                curr_skill = curr_skill.to(device)
                prev_correct = prev_correct.to(device)
                
                model.sample_assignment(tau)
                probs, full_state = model.forward_from_state(prev_skill, curr_skill, prev_correct, (state, ability_state, pc_given_alpha))
            
            curr_correct = curr_correct.to(device)
            mask = mask.to(device)
            loss = loss_fn(probs, curr_correct)
            loss = loss * mask
            loss = loss.sum() / mask.sum()

            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        valid_loss = evaluate(model, valid_seqs, device=device, n_samples=n_valid_samples, n_batch_seqs=n_batch_seqs*2, n_batch_trials=n_batch_trials)
        print("%d Train loss: %0.4f, Valid loss: %0.4f" % (e, np.mean(losses), valid_loss))

        epochs_since_last_best += 1

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_last_best = 0
        
        #scheduler.step(valid_loss)

        if epochs_since_last_best >= patience:
            break

    model.load_state_dict(best_state)
    return model 

def predict(model, test_seqs, n_batch_seqs=50, n_batch_trials=100, device='cpu', n_samples=5):

    with th.no_grad():
        final_probs = []

        for e in range(n_samples):
            
            model.sample_assignment(tau=1e-6)
            probs, curr_correct = _predict(model, test_seqs, n_batch_seqs, n_batch_trials, device)
            
            final_probs.append(probs[:,None])

    return curr_correct, np.mean(np.hstack(final_probs), axis=1)

def evaluate(model, test_seqs, n_batch_seqs=50, n_batch_trials=100, device='cpu', n_samples=5):
    with th.no_grad():
        losses = []

        for e in range(n_samples):
            
            model.sample_assignment(tau=1e-6)
            probs, curr_correct = _predict(model, test_seqs, n_batch_seqs, n_batch_trials, device)
            loss = -np.mean(curr_correct * np.log(probs) + (1-curr_correct) * np.log(1-probs))

            losses.append(loss)

    return np.mean(losses)

def _predict(model, seqs, n_batch_seqs, n_batch_trials, device):
    all_probs = []
    all_labels = []

    for seqs, new_seqs in sequences.iterate_batched(seqs, n_batch_seqs, n_batch_trials):
        curr_skill, curr_correct, mask, trial_index = transform(seqs)
        prev_skill, prev_correct, _, _ = transform(seqs, prev_trial=True)
                
        if new_seqs:
            prev_skill = prev_skill.to(device)
            curr_skill = curr_skill.to(device)
            prev_correct = prev_correct.to(device)

            probs, full_state = model(prev_skill, curr_skill, prev_correct)
        else:
            state, ability_state, pc_given_alpha  = full_state
            # trim the state
            n_state_size = state.shape[0]
            n_diff = n_state_size - len(seqs)
            if n_diff > 0:
                state = state[n_diff:,:]
                ability_state = ability_state[n_diff:,:]
                pc_given_alpha = pc_given_alpha[n_diff:,:]

            prev_skill = prev_skill.to(device)
            curr_skill = curr_skill.to(device)
            prev_correct = prev_correct.to(device)
            
            probs, full_state = model.forward_from_state(prev_skill, curr_skill, prev_correct, (state, ability_state, pc_given_alpha))
        
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

    return all_probs, all_labels

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
    df = pd.read_csv("data/datasets/gervetetal_statics.csv")
    #df['skill'] = df['core_skill']
    splits = np.load("data/splits/gervetetal_statics.npy")
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

    n_obs_kcs = int(np.max(df['skill']) + 1)
    model = train(train_seqs, valid_seqs, n_obs_kcs, 
        n_kcs=10, 
        device='cpu', 
        learning_rate=0.1, 
        epochs=100, 
        tau=0.5, 
        patience=10,
        n_batch_seqs=200, 
        n_batch_trials=50)

    test_students = set(test_df['student'])
    test_seqs = sequences.make_sequences(df, test_students)
    ytrue_test, ypred_test = predict(model, test_seqs, device='cpu',n_samples=20)
    auc_roc = sklearn.metrics.roc_auc_score(ytrue_test, ypred_test)
    test_loss = -np.mean(ytrue_test * np.log(ypred_test) + (1-ytrue_test) * np.log(1-ypred_test))

    print("Test loss: %0.4f, auc: %0.2f" % (test_loss, auc_roc))


if __name__ == "__main__":
    main()
    