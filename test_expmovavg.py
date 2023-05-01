import numpy as np 

def main():
    np.random.seed(53345)
    alpha = 0.4
    beta = 0.1
    h1 = 0.35

    y = np.array([[1, 0, 0, 1.0], [0, 1, 0, 1.0]])
    y_pred = np.zeros_like(y).astype(float)

    for s in range(y.shape[0]):
        y_pred[s, 0] = h1

        h = h1
        for i in range(1, y_pred.shape[1]):
            h = y[s, i-1] * alpha + (y[s, i-1] * (1 - alpha - beta) + beta) * h
            y_pred[s, i] = h 
    
    
    #
    # non-recursive
    #
    seqs = y
    
    #
    # create matrix of Fs (one row for each time step) TxT
    #
    fs = seqs * (1- alpha - beta) + beta  # BxT
    t = seqs.shape[1]
    Fs = np.tril(np.ones((t, t)), -1) * fs[:,None,:] + np.triu(np.ones((t, t)), 0)
    print("Log Fs:")
    logFs = np.log(Fs)
    print(logFs) # BxTxT
    
    #
    # Compute reverse cumulative product row-wise
    #
    rev_idx = np.arange(t-1, -1, -1)
    rev_cumsum = np.cumsum(logFs[:, :, rev_idx], axis=2)[:, :,rev_idx]
    print("Reverse cumulative sum:")
    print(rev_cumsum) # BxTxT
    
    #
    # Multiply by alphas and observations
    #
    alphas = np.ones(t) * alpha 
    alphas[0] = 1
    xs = np.roll(seqs, 1, axis=1) # BxT
    xs[:,0] = h1 
    print("Xs:")
    print(xs)
    
    mask = np.tril(np.ones(t), 0)
    
    R = np.exp(rev_cumsum + np.log(alphas[None,None,:])) * xs[:,None,:] * mask[None,:,:]
    print(R)

    print("Seq:")
    print(y)
    
    print("Reference:")
    print(y_pred)

    print("New:")
    print(np.sum(R, axis=2))

if __name__ == "__main__":
    main()