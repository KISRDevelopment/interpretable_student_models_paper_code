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

        self._logits = nn.Parameter(th.randn((n_problems, n_kcs)))

    def sample_A(self, tau, hard):
        return nn.functional.gumbel_softmax(self._logits, hard=hard, tau=tau, dim=1)
    
if __name__ == "__main__":

    module = SimpleKCDiscovery(10, 5)

    A = module.sample_A(0.5, True)
    print(A)