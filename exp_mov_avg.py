import numpy as np 
import torch as th 


def main():

    # PxD
    problem_embd = th.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1]
    ]).float()
    
    # B x T
    problem_seq = th.tensor([
        [1, 0, 3, 4, 2, 0],
        [1, 2, 3, 4, 0, 0],
        [1, 0, 3, 4, 2, 0]
    ])
    y = th.tensor([
        [1, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 1]
    ]).float()
    
    # compute problem cosine similarity
    mag = th.linalg.vector_norm(problem_embd, dim=1) # P
    mag = mag[problem_seq] # BxT
    denom = th.bmm(mag[:,:,None], mag[:,None,:]) # BxTx1 * Bx1xT = BxTxT
    
    x = problem_embd[problem_seq, :] # BxTxD
    sim_mat = th.bmm(x, th.permute(x, (0, 2, 1))) / denom # BxTxT
    
    # instantiate mask and apply to similarity matrix
    mask = th.ones_like(sim_mat).tril(diagonal=-1)
    sim_mat = sim_mat * mask 
    
    # compute effective delta_t based on similarities
    rev_idx = th.arange(sim_mat.shape[1]-1, end=-1, step=-1)
    delta_t = sim_mat[:, :, rev_idx].cumsum(1)[:, :, rev_idx] # BxTxT
    
    # compute the attention weights BxTxT
    weight = sim_mat * th.exp(-0.0 * delta_t)
    
    # Bx1xT * BxTxT = BxT
    yhat = (y[:,None,:] * weight).sum(2) / (weight.sum(dim=2) + 1e-6)
    print(yhat)



if __name__ == "__main__":
    main()
