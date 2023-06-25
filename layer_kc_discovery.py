import numpy as np
import metrics
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from typing import List

class SimpleKCDiscovery(nn.Module):

    def __init__(self, n_problems, n_kcs):
        super().__init__()
        
        self.n_problems = n_problems
        self.n_kcs = n_kcs 

        weight_matrix = th.randn((n_problems, n_kcs))
        self._logits = nn.Parameter(weight_matrix)

        self.trans_logits = nn.Parameter(th.randn(n_kcs, 2, 2))
        self.obs_logits = nn.Parameter(th.randn(n_kcs, 2, 2))
        self.init_logits = nn.Parameter(th.randn(n_kcs, 2))

    def get_params(self, tau, hard):
        A = nn.functional.gumbel_softmax(self._logits, hard=hard, tau=tau, dim=1)
        return A, self.trans_logits, self.obs_logits, self.init_logits
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