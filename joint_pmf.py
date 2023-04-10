import torch as th 
import numpy as np 
import itertools
import torch.nn.functional as F 
import torch.jit as jit
from torch import Tensor

def main():
    n_vars = 3

    joint_pmf_layer = JointPMF(n_vars)

    logit_probs = th.rand((5, n_vars))
    
class JointPMF(jit.ScriptModule):
    
    def __init__(self, n_vars):
        super(JointPMF, self).__init__()
        
        self.n_vars = n_vars 
        self.indexer = make_indexer(n_vars)

    @jit.script_method
    def forward(self, logit_probs: Tensor) -> Tensor:
        """
            Calculates the log joint PMF of N independent Bernoullis.
            logit_probs: BxN 
            Outputs:
                joint_logprobs: Bx2**N
        """
        log_probs = F.softplus(logit_probs, beta=-1)
        log_nprobs = -logit_probs + log_probs

        eff_logprobs = th.hstack((log_nprobs, log_probs))
        joint_logprobs = eff_logprobs[:,self.indexer].sum(dim=-1)

        return joint_logprobs


def make_indexer(n_vars):
    """
        Generates the indexing matrix for N independent Bernoulli
        random variables. Assumes it will index into a vector of 
        probabilities [1-mu1, 1-mu2, ..., mu1, mu2, ...]
    """

    #
    # First, generate a binary matrix of all combinations of variables
    #
    a = list(range(2))
    result = np.array(list(itertools.product(*n_vars*[a])))
    
    #
    # Second, convert into indexing matrix 
    #
    offsets = np.arange(n_vars)[None, :]
    indexer = offsets + n_vars * result 

    return th.tensor(indexer).long()

if __name__ == "__main__":
    main()
