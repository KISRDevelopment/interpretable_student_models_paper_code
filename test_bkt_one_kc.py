import numpy as np 
import pandas as pd 
import numpy.random as rng 
from collections import defaultdict
import model_brute_force_bkt
import metrics

def main():

    params = [0.01, 0.1, 0.2, 0.2, 0.25]
    seqs = generate_data(500, 200, *params)
    
    loglik = 0.0
    all_probs = []
    all_y = []
    for seq in seqs:
        probs = model_brute_force_bkt.forward_bkt(seq, *params)
        y = seq[:,1]
        ll = np.sum(y * np.log(probs) + (1-y) * np.log(1-probs))
        loglik += ll 

        all_probs.extend(probs)
        all_y.extend(y)
    
    all_probs = np.array(all_probs)
    all_y = np.array(all_y)
    
    print("LL: %f" % loglik)

    r = metrics.calculate_metrics(all_y, all_probs)
    print(r)

def generate_data(n_students, n_trials, pL, pF, pG, pS, pI0):
    
    # generate trials
    #cols = defaultdict(list)

    seqs = []
    for s in range(n_students):
        seq = []
        
        # initialize state (n_skills,)
        state = rng.binomial(1, pI0)

        for t in range(n_trials):
            
            if state == 1:
                pC = 1-pS
            else:
                pC = pG
            
            ans = rng.binomial(1, pC)
            seq.append((s, ans))

            # transition state
            if state == 0:
                state = rng.binomial(1, pL)
            else:
                state = 1 - rng.binomial(1, pF)
        seqs.append(np.array(seq))
    return seqs

if __name__ == "__main__":
    main()
