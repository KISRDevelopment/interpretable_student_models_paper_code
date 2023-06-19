import numpy as np 
import pandas as pd 
import numpy.random as rng 
from collections import defaultdict
import split_dataset

def main():
    rng.seed(96789)

    n_skills = 25
    ns_students = [10, 100, 1000, 3000]
    ns_trials_per_skill = [10, 50, 100, 500]

    # pI, pL, pF, pG, pS
    probs = rng.random((n_skills, 5))
    np.save("data/perf_data_probs.npy", probs)

    for n_students in ns_students:
        for n_trials_per_skill in ns_trials_per_skill:
            df = generate(probs, n_students, n_trials_per_skill)

            print("Students: %d, Trials per Skill: %d, Total Trials: %d" % (n_students, n_trials_per_skill, n_students*n_skills*n_trials_per_skill))
            print("Mean correct: %0.2f" % np.mean(df['correct']))

            dataset_name = "perf_%d_%d" % (n_students, n_trials_per_skill)
            df.to_csv("data/datasets/%s.csv" % dataset_name, index=False)
            full_splits = split_dataset.main(df, 5, 5)
            np.save("data/splits/%s.npy" % dataset_name, full_splits)
         

def generate(probs, n_students, n_trials_per_skill):

    n_skills = probs.shape[0]

    # skill sequence
    kc_seq = np.repeat(np.arange(n_skills), (n_trials_per_skill,))

    cols = defaultdict(list)
    for s in range(n_students):

        # initialize state (n_skills,)
        state = rng.binomial(1, probs[:,0])
        
        for t in range(kc_seq.shape[0]):
            kc = kc_seq[t]

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
            cols["problem"].append(0)
            
            # transition state
            if kc_state == 0:
                state[kc] = rng.binomial(1, pL)
            else:
                state[kc] = 1 - rng.binomial(1, pF)
    
    df = pd.DataFrame(cols)
    
    return df 
    
if __name__ == "__main__":
    main()
