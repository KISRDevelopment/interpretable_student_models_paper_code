import torch as th 
import numpy as np 
import torch.nn as nn 

def main():

    model = NeuralClustering(n_filters=10, max_clusters=5, item_dim=5, hidden_dim=5, device='cpu')

    for r in range(100):
        item_reps = th.rand((40, 5))
        A = model.sample(item_reps, True, 1.5)
        assert A.sum() == item_reps.shape[0]
        assert (A.sum(1) == 1).sum() == item_reps.shape[0]

    
    
class NeuralClustering(nn.Module):

    def __init__(self, n_filters, max_clusters, item_dim, hidden_dim, device):
        super(NeuralClustering, self).__init__()

        filters = []
        for i in range(n_filters):
            f = nn.Sequential(
                nn.Linear(item_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim))
            filters.append(f)
        self.filters = nn.ModuleList(filters)

        self.fo = nn.Linear(n_filters, 1)
        
        self.max_clusters = max_clusters

        self.device = device

    def forward(self, item_reps, assigned):
        """
            item_reps: N x d
            assigned: N 
            Output:
                logits Nx2
        """
        sims = []
        for f in self.filters:
            u = f(item_reps) # N x L
            h = (u * (1-assigned[:,None])).sum(0) / ((1-assigned).sum()+1e-6) # sum non-assigned stuff only
            
            sim = (u @ h) # N
            sims.append(sim)
        
        sims = th.vstack(sims).T # N x filters
        
        logits = self.fo(sims) # Nx1

        return th.hstack((logits, -logits)) # Nx2 
        
    def sample(self, item_reps, hard_samples, tau):
        n_items = item_reps.shape[0]

        A = th.zeros((self.max_clusters, n_items)).float().to(self.device)

        assigned = th.zeros(n_items).float().to(self.device)
        
        for i in range(self.max_clusters):
            if i == self.max_clusters-1:
                A[i, :] = 1-assigned
            else:
                logits = self.forward(item_reps, assigned)
                sampled = nn.functional.gumbel_softmax(logits, hard=hard_samples, tau=tau, dim=1) # Nx2
                to_assign = sampled[:, 0] * (1-assigned) # N
                A[i, :] = to_assign
                assigned = assigned + to_assign 
        
        return A.T 

if __name__ == "__main__":
    main()