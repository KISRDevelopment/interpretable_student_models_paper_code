import numpy as np 
import pandas as pd 
import numpy.random as rng 
from collections import defaultdict
import split_dataset
from scipy.stats import qmc
import itertools
import sklearn.datasets
import sys 
from sklearn.neighbors import NearestNeighbors
import os 

def main():


    n_students = 100
    n_problems_per_skill = 10
    n_features = 50
    target_nn_acc = 0.85
    std_range = np.linspace(0, 1, 50)
    ns_skills = [1, 5, 25, 50]
    
    for n_skills in ns_skills:
        print("****** Skills = %d ******" % n_skills)
        
        #
        # generate skill parameters
        #
        probs = generate_skill_params(n_skills)

        #
        # generate answers
        #
        skc = generate_seqs(n_students, n_problems_per_skill, probs)
        
        #
        # generate problem->skill assignments
        #
        X, A = generate_clusters(n_skills, 
            n_problems_per_skill, 
            n_features, 
            target_nn_acc,
            std_range)
        np.save("data/datasets/sd_%d.embeddings.npy" % n_skills, X)

        #
        # generate dataframes
        #
        df_blocked = generate_df(skc, A, 'blocked')
        df_blocked.to_csv("data/datasets/sd_%d_blocked.csv" % (n_skills), index=False)

        df_interleaved = generate_df(skc, A, 'interleaved')
        df_interleaved.to_csv("data/datasets/sd_%d_interleaved.csv" % (n_skills), index=False)

        splits = split_dataset.main(df_blocked, 5, 5)

        #
        # both scheduled variants use the same split
        #
        np.save("data/splits/sd_%d_blocked.npy" % n_skills, splits)
        np.save("data/splits/sd_%d_interleaved.npy" % n_skills, splits)


def generate_skill_params(n_skills):
   
    # possible parameter values
    pIs = [0.1, 0.25, 0.5, 0.75, 0.9]
    pLs = [0.01, 0.05, 0.1, 0.2] 
    pFs = [0.01, 0.05, 0.1, 0.2]
    pGs = [0.1, 0.2, 0.3, 0.4]
    pSs = [0.1, 0.2, 0.3, 0.4]

    all_prob_combs = np.array(list(itertools.product(pIs, pLs, pFs, pGs, pSs)))

    print("Choosing from %d combinations with replacement" % all_prob_combs.shape[0])

    probs = all_prob_combs[rng.choice(all_prob_combs.shape[0], replace=True, size=n_skills), :]
    
    # n_skills x 5
    return probs 

def generate_clusters(n_clusters, 
    n_samples_per_cluster, 
    n_features, 
    target_nn_acc, 
    sorted_cluster_stds, 
    reps=100):
    """
        Generates clusters that are caliberated such that a nearest neighbor classifier would have a given accuracy
    """
    min_diff = np.inf
    best_std = 0
    last_diff = 0
    for cluster_std in sorted_cluster_stds:
        means = np.zeros(reps)
        for r in range(reps):
            X, y = sklearn.datasets.make_blobs(n_samples=n_clusters*n_samples_per_cluster, 
                                               centers=n_clusters, 
                                               n_features=n_features, 
                                               cluster_std=cluster_std, 
                                               center_box=[-1, 1])
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)
            point_label = y[indices[:,0]]
            label_of_nn = y[indices[:,1]]
            means[r] = np.mean(point_label == label_of_nn)
        acc_diff = np.mean(np.abs(target_nn_acc - np.mean(means)))
        
        print("%0.2f %0.2f %0.2f" % (cluster_std, np.mean(means), acc_diff))
        
        if acc_diff < min_diff:
            min_diff = acc_diff
            best_std = cluster_std 
        elif acc_diff - last_diff > 0.02:
            pass
            #break # diff is starting to increase
        
        last_diff = acc_diff
    
    print("Best Std: %0.2f (diff: %0.2f)" % (best_std, min_diff))
    X, y = sklearn.datasets.make_blobs(n_samples=n_clusters*n_samples_per_cluster, 
                                        centers=n_clusters, 
                                        n_features=n_features, 
                                        cluster_std=best_std, 
                                        center_box=[-1, 1])
    
    return X, y


def generate_seqs(n_students, n_problems_per_skill, probs):
    n_kcs = probs.shape[0]

    skc = np.zeros((n_students, n_kcs, n_problems_per_skill))

    for s in range(n_students):
        
        state = rng.binomial(1, probs[:,0])
        
        pL = probs[:,1]
        pF = probs[:,2]
        pG = probs[:,3]
        pS = probs[:,4]

        for t in range(n_problems_per_skill):
            
            pC = (1 - pS) * state + (1-state) * pG 
            
            ans = rng.binomial(1, pC)
            
            state = rng.binomial(1, (1 - pF) * state + (1-state) * pL)
            
            skc[s, :, t] = ans
        
    return skc 

def generate_df(skc, A, kcseq):
    n_kcs = skc.shape[1]
    n_problems_per_skill = skc.shape[2]

    cols = defaultdict(list)
    for s in range(skc.shape[0]):
        #
        # create random problem sequence for each KC
        #
        problem_seqs = []
        for kc in range(n_kcs):
            kc_problems = np.where(A == kc)[0]
            np.random.shuffle(kc_problems)
            problem_seqs.append(kc_problems)
        problem_seqs = np.array(problem_seqs)

        #
        # flatten trial sequence, either blocked or interleaved
        #
        order = 'C' if kcseq == 'blocked' else 'F'
        ans_seq = skc[s, :, :].flatten(order)
        problem_seq = problem_seqs.flatten(order)
        skill_seq = np.tile(np.arange(n_kcs), (n_problems_per_skill,1)).T.flatten(order)
        
        #
        # generate df
        #
        for i in range(ans_seq.shape[0]):
            cols["student"].append(s)
            cols["correct"].append(ans_seq[i])
            cols["skill"].append(skill_seq[i])
            cols["problem"].append(problem_seq[i])


    print(skill_seq)
    df = pd.DataFrame(cols)
    df['correct'] = df['correct'].astype(int)
    return df 

if __name__ == "__main__":
    main()
