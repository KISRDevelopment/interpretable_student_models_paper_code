import numpy as np
import metrics
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from typing import List

class SimpleKCDiscovery(nn.Module):

    def __init__(self, initial_A):
        super().__init__()
        
        initial_A = th.tensor(initial_A)

        n_problems, n_kcs = initial_A.shape 

        self.n_problems = n_problems
        self.n_kcs = n_kcs 

        self._logits = nn.Parameter(initial_A * 5 + (1-initial_A) * -5)

    def sample_A(self, tau, hard):
        return nn.functional.gumbel_softmax(self._logits, hard=hard, tau=tau, dim=1)
        
if __name__ == "__main__":

    module = SimpleKCDiscovery(10, 5)

    A = module.sample_A(0.5, True)
    print(A)