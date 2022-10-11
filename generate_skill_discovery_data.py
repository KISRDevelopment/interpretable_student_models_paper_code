import numpy as np 
import pandas as pd 
import numpy.random as rng 
from collections import defaultdict
import split_dataset
from scipy.stats import qmc
def main():
    
    n_students = 500
    n_problems_per_skill = 50
    n_skills = 10
    
    # pI, pL, pF, pG, pS
    sampler = qmc.Sobol(d=5, scramble=True)
    probs = sampler.random_base2(m=6)[:n_skills,:]
    print(probs)
    
    # generate assignments
    kcs = np.repeat(np.arange(n_skills), (n_problems_per_skill,))
    problems = np.random.permutation(n_problems_per_skill * n_skills)
    A = np.zeros(problems.shape[0],dtype=int)
    A[problems] = kcs 
    
    np.savez("data/skill_discovery_data_params.npz", probs=probs, A=A)

    print("Dataset size: %d students" % n_students)
         
    # generate trials
    cols = defaultdict(list)
    
    for s in range(n_students):

        # initialize state (n_skills,)
        state = rng.binomial(1, probs[:,0])
        
        # generate problem sequence
        problem_seq = np.random.permutation(problems.shape[0])

        for t in range(problem_seq.shape[0]):
            problem = problem_seq[t]
            kc = A[problem]

            kc_state = state[kc]
            _, pL, pF, pG, pS = probs[kc, :]
                
            # get p(correct|state)
            if kc_state == 1:
                pC = 1-pS
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
                state[kc] = 1 - rng.binomial(1, pF)
        
    df = pd.DataFrame(cols)
        
    print(np.mean(df['correct']))

    dataset_name = "sd_%d" % n_students
    df.to_csv("data/datasets/%s.csv" % dataset_name, index=False)
    split_dataset.main("data/datasets/%s.csv" % dataset_name, 
        "data/splits/%s.npy" % dataset_name, 5, 5)
    
if __name__ == "__main__":
    main()
