import numpy as np 
import pandas as pd 
import numpy.random as rng 
from collections import defaultdict
import split_dataset
from scipy.stats import qmc
import itertools
def main(n_students, n_problems_per_skill, n_skills, seed=None, same_order=False):
    if seed is not None:
        np.random.seed(seed)

    # pI, pL, pF, pG, pS
    pIs = [0.1, 0.25, 0.5, 0.75, 0.9]
    pLs = pIs 
    pFs = pIs
    pGs = np.array([0.1, 0.2, 0.3, 0.4])
    pSs = np.array([0.1, 0.2, 0.3, 0.4])
    all_prob_combs = np.array(list(itertools.product(pIs, pLs, pFs, pGs, pSs)))
    print("Choosing from %d combinations" % all_prob_combs.shape[0])
    probs = all_prob_combs[rng.choice(all_prob_combs.shape[0], replace=False, size=n_skills), :]
    print(probs)
    
    # generate assignments
    kcs = np.repeat(np.arange(n_skills), (n_problems_per_skill,))
    problems = np.random.permutation(n_problems_per_skill * n_skills)
    A = np.zeros(problems.shape[0],dtype=int)
    A[problems] = kcs 
    
    print("Dataset size: %d students" % n_students)
         
    # generate trials
    cols = defaultdict(list)
    
    # generate problem sequence
    if same_order:
        problem_seq = np.random.permutation(problems.shape[0])
    
    n_state = 0
    for s in range(n_students):

        # initialize state (n_skills,)
        state = rng.binomial(1, probs[:,0])
        
        # generate problem sequence if different for each student
        if not same_order:
            problem_seq = np.random.permutation(problems.shape[0])
        
        for t in range(problem_seq.shape[0]):
            problem = problem_seq[t]
            kc = A[problem]
            
            kc_state = state[kc]
            _, pL, pF, pG, pS = probs[kc, :]
                
            # get p(correct|state)
            if kc_state == 1:
                pC = 1-pS
                n_state += 1
            else:
                pC = pG
            
            
            ans = rng.binomial(1, pC)
            cols["student"].append(s)
            cols["correct"].append(ans)
            cols["skill"].append(kc)
            cols["problem"].append(problem)
                
            # transition state
            if kc_state == 0:
                state[kc] = rng.binomial(1, pL)
            else:
                state[kc] = rng.binomial(1, 1-pF)

    df = pd.DataFrame(cols)
        
    print(np.mean(df['correct']))

    return df, probs, A 

if __name__ == "__main__":
    import sys 
    n_students = int(sys.argv[1]) 
    n_problems_per_skill = int(sys.argv[2]) 
    n_skills = int(sys.argv[3])
    dataset_name = sys.argv[4]

    df, probs, A = main(n_students, n_problems_per_skill, n_skills)

    df.to_csv("data/datasets/%s.csv" % dataset_name, index=False)
    splits = split_dataset.main(df, 5, 5)
    np.save("data/splits/%s.npy" % dataset_name, splits)
