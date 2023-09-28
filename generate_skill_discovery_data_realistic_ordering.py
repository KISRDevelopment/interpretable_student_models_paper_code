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
import model_brute_force_bkt

def to_student_sequences(df):
    seqs = defaultdict(lambda: {
        "obs" : [],
        "kc" : [],
        "problem" : []
    })
    for r in df.itertuples():
        seqs[r.student]["obs"].append(r.correct)
        seqs[r.student]["kc"].append(r.skill)
        seqs[r.student]["problem"].append(r.problem)
    return seqs

def main(dataset_path):
    
    df = pd.read_csv(dataset_path)
    
    seqs = to_student_sequences(df)

    n_skills = np.max(df['skill']) + 1
    n_problems = np.max(df['problem']) + 1

    n_features = 50
    target_nn_acc = 0.85
    std_range = np.linspace(0, 1, 50)
    
    print("Problems: %d, skills: %d" % (n_problems, n_skills))
    
    
    #
    # generate skill parameters
    #
    probs = generate_skill_params(df)
    
    #
    # generate answers
    #
    synth_df = generate_seqs(seqs, probs)
    
    #
    # split
    #
    splits = split_dataset.main(synth_df, 5, 5)

    #
    # create mapping from skills -> problems 
    #
    skills_to_problems = create_one_to_many_mapping(df['skill'], df['problem'])
    
    #
    # generate problem->skill assignments
    #
    X = generate_clusters(skills_to_problems, 
        n_features, 
        target_nn_acc,
        std_range)
    
    #
    # save everything
    #

    basename = os.path.basename(dataset_path).replace('.csv','')
    synth_df.to_csv("data/datasets/sd-realistic_%s.csv" % basename, index=False)
    np.save("data/datasets/sd-realistic_%s.embeddings.npy" % basename, X)
    np.save("data/splits/sd-realistic_%s.npy" % basename, splits)
    
    

def create_one_to_many_mapping(aa, bb):
    mapping = defaultdict(set)
    for a, b in zip(aa, bb):
        mapping[a].add(b)
    return { k: list(v) for k, v in mapping.items() }

def generate_skill_params(df):
    
    # 
    # extract KC parameters
    #
    seqs_by_skill = model_brute_force_bkt.prepare(df)
    params = fit_bkt(seqs_by_skill) # skill -> pL, pF, pG, pS, pL0
    
    probs = np.zeros((len(params), 5))
    for skill, p in params.items():
        probs[skill, :] = [p[4], p[0], p[1], p[2], p[3]]
    
    # n_skills x 5
    return probs 

def fit_bkt(seqs_by_skill):
    """ fit one BKT model per skill """

    pIs = [0.1, 0.25, 0.5, 0.75, 0.9]
    pLs = [0.01, 0.05, 0.1, 0.15, 0.2] 
    pFs = [0.01, 0.05, 0.1, 0.15, 0.2]
    pGs = [0.05, 0.1, 0.2, 0.3, 0.4]
    pSs = [0.05, 0.1, 0.2, 0.3, 0.4]
    search_space = np.array(list(itertools.product(pLs, pFs, pGs, pSs, pIs)))
    
    p_by_skill = {}
    for skill in sorted(seqs_by_skill.keys()):
        seqs = seqs_by_skill[skill]
        best_p = model_brute_force_bkt.fit_brute(seqs, search_space)
        p_by_skill[skill] = best_p
        print("Finished skill %d" % skill)
        
    return p_by_skill

def generate_clusters(skills_to_problems, 
    n_features, 
    target_nn_acc, 
    sorted_cluster_stds, 
    reps=10):
    """
        Generates clusters that are caliberated such that a nearest neighbor classifier would have a given accuracy
    """

    skills = sorted(skills_to_problems.keys())
    samples_per_cluster = [len(skills_to_problems[s]) for s in skills]
    
    min_diff = np.inf
    best_std = 0
    last_diff = 0
    for cluster_std in sorted_cluster_stds:
        means = np.zeros(reps)
        for r in range(reps):
            X, y = sklearn.datasets.make_blobs(n_samples=samples_per_cluster,
                                               n_features=n_features, 
                                               cluster_std=cluster_std, 
                                               center_box=(-1, 1))
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X)
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
    X, y = sklearn.datasets.make_blobs(n_samples=samples_per_cluster,
                                        n_features=n_features, 
                                        cluster_std=best_std, 
                                        center_box=(-1, 1))
    
    # reorganize according to problem assignments
    output_X = np.zeros_like(X)
    for skill in skills:
        ix = y == skill 
        problem_ids = skills_to_problems[skill]
        output_X[problem_ids, :] = X[ix, :]
    
    return output_X


def generate_seqs(seqs, probs):
    """
        seqs: the realistic sequence of kc-ordering to mimic
    """

    n_kcs = probs.shape[0]
    n_students = len(seqs)

    output_cols = defaultdict(list)

    for student, seq in seqs.items():
        
        state = rng.binomial(1, probs[:,0])
        
        pL = probs[:,1]
        pF = probs[:,2]
        pG = probs[:,3]
        pS = probs[:,4]

        for i in range(len(seq['kc'])):
            
            skill = seq['kc'][i]
            problem = seq['problem'][i]
            
            pC = (1 - pS[skill]) * state[skill] + (1-state[skill]) * pG[skill] 
            
            ans = rng.binomial(1, pC)
            
            state[skill] = rng.binomial(1, (1 - pF[skill]) * state[skill] + (1-state[skill]) * pL[skill])
            
            output_cols['student'].append(student)
            output_cols['skill'].append(skill)
            output_cols['problem'].append(problem)
            output_cols['correct'].append(int(ans==1))
        
    return pd.DataFrame(output_cols) 

if __name__ == "__main__":
    main(sys.argv[1])
