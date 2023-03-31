import numpy as np 
import torch as th 


def main():

    problem_embd = th.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1]
    ]).float()
    
    problem_seq = th.tensor([1, 0, 3, 4, 2, 0])
    y =   th.tensor([1, 1, 1, 0, 1, 1]).float()
    
    # compute cosine similarity (t x t)
    mag = th.linalg.vector_norm(problem_embd, dim=1)
    denom = mag[problem_seq, None] @ mag[problem_seq, None].T
    sim_mat = (problem_embd[problem_seq, :] @ problem_embd[problem_seq, :].T) / denom
    
    
    # instantiate mask and apply to similarity matrix
    mask = th.ones_like(sim_mat).tril(diagonal=-1)
    sim_mat = sim_mat * mask 
    
    # compute effective delta_t based on similarities
    rev_idx = th.arange(sim_mat.shape[0]-1, end=-1, step=-1)
    delta_t = sim_mat[:, rev_idx].cumsum(1)[:, rev_idx]
    
    # compute the attention weights (t x t)
    weight = sim_mat * th.exp(-0.1 * delta_t)
    
    # 1 x t * t x t
    yhat = (y[None, :] * weight).sum(1, keepdims=True) / (weight.sum(dim=1, keepdims=True) + 1e-6)
    print(yhat)



if __name__ == "__main__":
    main()
