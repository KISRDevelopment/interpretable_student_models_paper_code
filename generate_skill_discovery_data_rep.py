import numpy as np 
import pandas as pd 
import numpy.random as rng 
from collections import defaultdict
import split_dataset
from scipy.stats import qmc
import itertools
import sklearn.datasets

def main(n_students, 
         n_problems_per_skill, 
         n_skills,
         n_features, 
         seed=None, 
         block_kcs=False, 
         has_forgetting=True):
    
    if seed is not None:
        rng.seed(seed)

    # pI, pL, pF, pG, pS
    pIs = [0.1, 0.25, 0.5, 0.75, 0.9]
    pLs = [0.01, 0.05, 0.1, 0.2] 
    pFs = [0.01, 0.05, 0.1, 0.2] if has_forgetting else [0.0]
    pGs = [0.01, 0.05, 0.1, 0.2]
    pSs = [0.01, 0.05, 0.1, 0.2]
    all_prob_combs = np.array(list(itertools.product(pIs, pLs, pFs, pGs, pSs)))
    print("Choosing from %d combinations" % all_prob_combs.shape[0])
    probs = all_prob_combs[rng.choice(all_prob_combs.shape[0], replace=False, size=n_skills), :]
    print(probs)
    
    # generate assignments based on problem representation
    X, A = sklearn.datasets.make_blobs(n_samples=n_skills * n_problems_per_skill, 
        centers=n_skills, 
        n_features=n_features, 
        cluster_std=1, 
        center_box=[-1, 1]) # N_problems * N_features and N_problems
    print("Dataset size: %d students" % n_students)
         
    # generate trials
    cols = defaultdict(list)
    
    # generate problem sequence
    if block_kcs:
        problem_seq = np.argsort(A)
        print(A[problem_seq])

    kc_one = 0
    kc_zero = 0
    for s in range(n_students):

        # initialize state (n_skills,)
        state = rng.binomial(1, probs[:,0])
        
        # generate problem sequence if different for each student
        if not block_kcs:
            problem_seq = np.random.permutation(A.shape[0])
        
        for t in range(problem_seq.shape[0]):
            problem = problem_seq[t]
            kc = A[problem]
            
            kc_state = state[kc]
            _, pL, pF, pG, pS = probs[kc, :]
            
            # get p(correct|state)
            if kc_state == 1:
                pC = 1-pS
                kc_one += 1
            else:
                pC = pG
                kc_zero += 1
            
            ans = rng.binomial(1, pC)
            cols["student"].append(s)
            cols["correct"].append(ans)
            cols["skill"].append(kc)
            cols["problem"].append(problem)
                
            # transition state
            if kc_state == 0:
                state[kc] = rng.binomial(1, pL)
            else:
                state[kc] = 1 - rng.binomial(1, pF)

    print(kc_zero, kc_one)

    df = pd.DataFrame(cols)
        
    print(np.mean(df['correct']))

    return df, probs, A, X 

if __name__ == "__main__":
    import sys 
    n_students = int(sys.argv[1]) 
    n_problems_per_skill = int(sys.argv[2]) 
    n_skills = int(sys.argv[3])
    n_features = int(sys.argv[4])
    dataset_name = sys.argv[5]
    block_kcs = sys.argv[6] == '1'

    df, probs, A, X = main(n_students, n_problems_per_skill, n_skills, n_features=n_features, block_kcs=block_kcs)

    df.to_csv("data/datasets/%s.csv" % dataset_name, index=False)
    
    np.save("data/datasets/%s.embeddings.npy" % dataset_name, X)

    splits = split_dataset.main(df, 5, 5)
    np.save("data/splits/%s.npy" % dataset_name, splits)
