import numpy as np 
import pandas as pd
import generate_skill_discovery_data
import split_dataset
import os 
import torch.nn as nn 
import torch as th 
import torch.nn.functional as F
import sklearn.metrics 
import copy 

def main():
    
    n_epochs = 10000
    n_patience = 50

    n_students = 1000
    n_problems_per_skill = 10
    n_skills = 5
    learning_rate = 0.1
    sparsity_reg = 0.1
    tau = 1.5

    use_actual_labels = False 

    n_problems = n_problems_per_skill * n_skills
    device = 'cuda:0'
    
    df, probs, actual_labels = generate_skill_discovery_data.main(n_problems_per_skill=n_problems_per_skill, 
        n_students=n_students, 
        n_skills=n_skills,
        no_bkt=True)
    
    splits = split_dataset.main(df, 5, 5)

    split = splits[0, :]

    train_ix = split == 2
    valid_ix = split == 1
    test_ix = split == 0

    train_df = df[train_ix]
    valid_df = df[valid_ix]
    test_df = df[test_ix]

    if use_actual_labels:
        A = F.one_hot(th.tensor(actual_labels), num_classes=n_skills).float().to(device)
        model = SimpleModel(n_problems, n_skills, A)
    else:
        model = SimpleModel(n_problems, n_skills)

    model.to(device)
    
    optimizer = th.optim.NAdam(model.parameters(), lr=learning_rate)

    
    valid_problem = th.tensor(valid_df['problem'].to_numpy()).long().to(device)
    valid_corr = th.tensor(valid_df['correct'].to_numpy()).float()
    
    best_auc_roc = 0.
    waited = 0

    for e in range(n_epochs):
        train_df = train_df.sample(frac=1)
        train_problem = th.tensor(train_df['problem'].to_numpy()).long().to(device)
        train_corr = th.tensor(train_df['correct'].to_numpy()).float()

        logit_pC = model(train_problem, tau).cpu()

        train_loss = F.binary_cross_entropy_with_logits(logit_pC, train_corr)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with th.no_grad():
            logit_pC = model(valid_problem, 1e-6).cpu()

            auc_roc = sklearn.metrics.roc_auc_score(valid_corr.numpy(), logit_pC.numpy())

            pred_labels = model.get_predicted_labels().cpu().numpy()
            rand_index = sklearn.metrics.adjusted_rand_score(actual_labels, pred_labels)

            new_best = auc_roc > best_auc_roc
            print("%4d Train loss: %8.4f, Valid AUC-ROC: %0.2f, Rand: %0.2f %s" % (e, train_loss.item(), 
                auc_roc, rand_index, '***' if new_best else ''))

            if auc_roc > best_auc_roc:
                best_auc_roc = auc_roc
                best_state = copy.deepcopy(model.state_dict())
                waited = 0
            else:
                waited += 1
            
            if waited >= n_patience:
                break 
        model.train()

    model.load_state_dict(best_state)
    with th.no_grad():

        print(th.max(th.softmax(model.kc_membership_logits.weight, dim=1),1)[0].cpu().numpy())

class SimpleModel(nn.Module):
    def __init__(self, n_problems, n_latent_kcs, predefined_labels=None):
        super(SimpleModel, self).__init__()
        
        self.kc_membership_logits = nn.Embedding(n_problems, n_latent_kcs)
        self.kc_logit_pC = nn.Parameter(th.rand(n_latent_kcs))
        self.predefined_labels = predefined_labels
    
    def get_predicted_labels(self):
        if self.predefined_labels is None:
            return th.argmax(self.kc_membership_logits.weight, dim=1)
        
        return th.argmax(self.predefined_labels, dim=1)

    def forward(self, problem, tau):
        
        if self.predefined_labels is None:
            kc_membership_probs = nn.functional.gumbel_softmax(self.kc_membership_logits.weight, 
                hard=False, 
                tau=tau, dim=1)
        else:
            kc_membership_probs = self.predefined_labels

        # B x K
        mixture = kc_membership_probs[problem,:]

        return th.matmul(mixture, self.kc_logit_pC)

if __name__ == "__main__":
    main()