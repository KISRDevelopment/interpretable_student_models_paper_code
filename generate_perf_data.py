import numpy as np 
import pandas as pd 
import numpy.random as rng 
from collections import defaultdict
import split_dataset

def main():
    rng.seed(6456)

    ns_students = [
        # 8,
        # 16, 
        # 32,
        # 64, 
        # 128, 
        # 256,
        # 512,
        # 1024,
        # 2048,
        # 4096,
        8192
    ]
    n_skills = 50
    n_trials_per_skill = 10

    # pI, pL, pF, pG, pS
    probs = rng.random((n_skills, 5))

    kc_seq = np.repeat(np.arange(n_skills), (n_trials_per_skill,))
    
    for n_students in ns_students:

        print("Dataset size: %d students" % n_students)
         
        # generate trials
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
        
        print(np.mean(df['correct']))

        dataset_name = "perf_%d" % n_students
        df.to_csv("data/datasets/%s.csv" % dataset_name, index=False)
        split_dataset.main("data/datasets/%s.csv" % dataset_name, 
            "data/splits/%s.npy" % dataset_name, 5, 5)
if __name__ == "__main__":
    main()
