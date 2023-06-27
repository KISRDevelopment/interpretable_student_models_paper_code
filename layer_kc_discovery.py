import numpy as np
import metrics
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from typing import List


# class AltSimpleKCDiscovery(nn.Module):

#     def __init__(self, n_problems, n_kcs, device):
#         super().__init__()
        
#         self.n_problems = n_problems
#         self.n_kcs = n_kcs 

#         self.logit_exp_vals = nn.Parameter(th.randn(n_problems)) # P
#         self.log_gamma = nn.Parameter(th.randn(n_problems)) # P
#         self.bins = th.arange(n_kcs).to(device) # K
        
#     def sample_A(self, tau, hard):
#         logits = self.get_logits()
#         return nn.functional.gumbel_softmax(logits, hard=hard, tau=tau, dim=1)

#     def get_logits(self):
#         # Px1 - 1xK = PxK
#         exp_vals = self.logit_exp_vals[:,None].sigmoid()*self.n_kcs
        
#         diffs = (exp_vals - self.bins[None,:]).abs()
#         logits = -self.log_gamma.exp()[:,None] * diffs # PxK
        
#         return logits 

class SimpleKCDiscovery(nn.Module):

    def __init__(self, n_problems, n_kcs, initial_kcs):
        super().__init__()
        
        self.n_problems = n_problems
        self.n_kcs = n_kcs 

        logits = th.randn((n_problems, n_kcs))
        logits[:,initial_kcs:] = -10
        self._logits = nn.Parameter(logits)

    def sample_A(self, tau, hard):
        return nn.functional.gumbel_softmax(self._logits, hard=hard, tau=tau, dim=1)

    def get_logits(self):
        return self._logits

class FeaturizedKCDiscovery(nn.Module):

    def __init__(self, problem_feature_mat, n_kcs):
        super().__init__()

        self._proj = nn.Linear(problem_feature_mat.shape[1], n_kcs)
        self.problem_feature_mat = problem_feature_mat
    def sample_A(self, tau, hard):
        logits = self.get_logits()
        return nn.functional.gumbel_softmax(logits, hard=hard, tau=tau, dim=1)
    
    def get_logits(self):
        return self._proj(self.problem_feature_mat)
        
if __name__ == "__main__":

    module = SimpleKCDiscovery(10, 5)

    A = module.sample_A(0.5, True)
    print(A)