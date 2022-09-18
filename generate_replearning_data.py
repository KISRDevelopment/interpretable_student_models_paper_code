import numpy as np 
import pandas as pd 
import numpy.random as rng 
from collections import defaultdict
import split_dataset

def main():
    rng.seed(6456)

    d = np.load("tmp/features_grid.npz")
    data = d['data']
    difficulties = (d['difficulties'] - np.min(d['difficulties'])) / (np.max(d['difficulties']) - np.min(d['difficulties']))
    offsets = 3 - 6 * difficulties

    n_students = 1000
    n_trials_per_student = 50
    p_learning = 0.2
    p_forgetting = 0.2
    p_initial = 0.2
    offset_correct_h0 = -1
    offset_correct_h1 = 1

    # generate trials
    cols = defaultdict(list)
    for s in range(n_students):

        # initialize state
        state = rng.binomial(1, p_initial)

        # generate an ordering of problems
        problem_instance_seq = rng.permutation(data.shape[0])
        problem_offsets = offsets[problem_instance_seq]
        
        for t in range(n_trials_per_student):
            # get p(correct|state)

            if state == 1:
                logit_pC = offset_correct_h1 + problem_offsets[t]
            else:
                logit_pC = offset_correct_h0 + problem_offsets[t]
            
            pC = 1/(1+np.exp(-logit_pC))
            ans = rng.binomial(1, pC)
            cols["student"].append(s)
            cols["correct"].append(ans)
            cols["skill"].append(0)
            cols["problem"].append(problem_instance_seq[t])
            
            # transition state
            if state == 0:
                state = rng.binomial(1, p_learning)
            else:
                state = 1 - rng.binomial(1, p_forgetting)
    
    df = pd.DataFrame(cols)
    #print(df)
    print(np.mean(df['correct']))

    df.to_csv("data/datasets/rep_learning.csv", index=False)

    split_dataset.main("data/datasets/rep_learning.csv", "data/splits/rep_learning.npy", 5, 5)
if __name__ == "__main__":
    main()
