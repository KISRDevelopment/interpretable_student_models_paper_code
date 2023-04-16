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


def main():

    n_students = int(sys.argv[1]) 
    n_problems_per_skill = int(sys.argv[2]) 
    n_skills = int(sys.argv[3])
    n_features = int(sys.argv[4])
    target_nn_acc = float(sys.argv[5])
    kcseq = sys.argv[6]
    dataset_name = sys.argv[7]
    
    df, probs, A, X = generate(n_students, 
                           n_problems_per_skill, 
                           n_skills, 
                           n_features=n_features, 
                           target_nn_acc=target_nn_acc,
                           kcseq=kcseq)

    df.to_csv("data/datasets/%s.csv" % dataset_name, index=False)
    np.save("data/datasets/%s.embeddings.npy" % dataset_name, X)

    splits = split_dataset.main(df, 5, 5)
    np.save("data/splits/%s.npy" % dataset_name, splits)


def generate(n_students, 
             n_problems_per_skill, 
             n_skills,
             n_features,
             target_nn_acc,
             kcseq,
             seed=None):
    
    if seed is not None:
        rng.seed(seed)

    #
    # Possible parameter combinations
    #
    pIs = [0.1, 0.25, 0.5, 0.75, 0.9]
    pLs = [0.01, 0.05, 0.1, 0.2] 
    pFs = [0.01, 0.05, 0.1, 0.2]
    pGs = [0.1, 0.2, 0.3, 0.4]
    pSs = [0.1, 0.2, 0.3, 0.4]
    all_prob_combs = np.array(list(itertools.product(pIs, pLs, pFs, pGs, pSs)))
    print("Choosing from %d combinations with replacement" % all_prob_combs.shape[0])
    probs = all_prob_combs[rng.choice(all_prob_combs.shape[0], replace=True, size=n_skills), :]
    
    #
    # generate assignments based on problem representation
    #
    X, A = generate_clusters(n_skills, 
        n_problems_per_skill, 
        n_features, 
        target_nn_acc, 
        np.linspace(0.75, 1.25, 25))
    #
    # generate trials
    #
    cols = defaultdict(list)
    kc_one = 0
    kc_zero = 0
    for s in range(n_students):

        # initialize state (n_skills,)
        state = rng.binomial(1, probs[:,0])
        
        # generate problem sequence
        if kcseq == 'blocked':
            problem_seq = blocked_generator(A)
        elif kcseq == 'interleaved':
            problem_seq = interleaved_generator(A)
        
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

    df = pd.DataFrame(cols)
        
    print("Mean correct: %0.2f" % np.mean(df['correct']))

    return df, probs, A, X 

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
            break # diff is starting to increase
        
        last_diff = acc_diff
    
    print("Best Std: %0.2f" % best_std)
    X, y = sklearn.datasets.make_blobs(n_samples=n_clusters*n_samples_per_cluster, 
                                        centers=n_clusters, 
                                        n_features=n_features, 
                                        cluster_std=best_std, 
                                        center_box=[-1, 1])
    
    return X, y

def blocked_generator(A):
    """ KCs are blocked, but problems within the KC are shuffled """
    n_kcs = np.max(A)+1
    
    problem_seq = []
    for kc in range(n_kcs):
        kc_problems = np.where(A == kc)[0]
        np.random.shuffle(kc_problems)
        problem_seq.extend(kc_problems)
    
    problem_seq = np.array(problem_seq)
    assert np.unique(problem_seq).shape == problem_seq.shape
    return problem_seq 

def interleaved_generator(A):
    """ KCs are interleaved but problems within KC are shuffled """
    n_kcs = np.max(A)+1
    
    problem_seq = np.zeros(A.shape[0]).astype(int)
    for kc in range(n_kcs):
        kc_problems = np.where(A == kc)[0]
        np.random.shuffle(kc_problems)
        problem_seq[kc:A.shape[0]:n_kcs] = kc_problems
    
    assert np.unique(problem_seq).shape == problem_seq.shape
    return problem_seq 

if __name__ == "__main__":
    main()
