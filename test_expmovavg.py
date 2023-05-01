import numpy as np 

def main():
    np.random.seed(53345)
    alpha = 0.63
    beta = 0.3
    h1 = 0.1

    y = np.array([1, 0, 0, 1.0])
    
    y_pred = np.zeros_like(y).astype(float)
    y_pred[0] = h1

    h = h1
    for i in range(1, y_pred.shape[0]):
        h = y[i-1] * alpha + (y[i-1] * (1 - alpha - beta) + beta) * h
        y_pred[i] = h 
    
    print("Reference:")
    print(y)
    print(y_pred)

    print("Non-recursive:")

    #
    # create matrix of Fs (one row for each time step) TxT
    #
    fs = y * (1- alpha - beta) + beta 
    t = y.shape[0]
    Fs = np.tril(np.ones((t, t)), -1) * fs[None,:] + np.triu(np.ones((t, t)), 0)
    print("Fs:")
    print(Fs)

    #
    # Compute reverse cumulative product row-wise
    #
    rev_idx = np.arange(t-1, -1, -1)
    rev_cumprod = np.cumprod(Fs[:, rev_idx], axis=1)[:,rev_idx]
    print("Reverse cumulative product:")
    print(rev_cumprod)

    #
    # Multiply by alphas and observations
    #
    alphas = np.ones(t) * alpha 
    alphas[0] = 1
    xs = np.roll(y, 1)
    xs[0] = h1 
    print("Xs:")
    print(xs)

    print("Mask:")
    mask = np.tril(np.ones(t), 0)
    print(mask)

    R = rev_cumprod * alphas[None,:] * xs[None,:] * mask
    print("R:")
    print(R)
    print(np.sum(R, axis=1))
if __name__ == "__main__":
    main()