"""
    Standard Bayesian Knowledge Tracing
    Mohammad M Khajah <mmkhajah@kisr.edu.kw>
"""
import pandas as pd
import numpy as np 
import sys 
import sklearn.metrics 
import subprocess 
from numba import jit
import itertools 
from numba.typed import List
import time 

def main():
    input_path = sys.argv[1]
    split_path = sys.argv[2]
    output_path = sys.argv[3]

    master_df = pd.read_csv(sys.argv[1])

    splits = np.load(split_path)

    results = []
    param_dfs = []
    for split_id in range(splits.shape[0]):

        split = splits[split_id, :]

        train_df = master_df[(split == 2) | (split == 1)]
        test_df = master_df[split == 0]

        test_only_skills = set(np.unique(test_df['skill'])) - set(np.unique(train_df['skill'])) 
        
        print("Number of skills: %d" % len(set(master_df['skill'])))
        remove_ix = test_df['skill'].isin(test_only_skills)
        print("Removing %d trials from test due to lack of skill in training" % np.sum(remove_ix))

        test_df = test_df[~remove_ix]
        
        train_seqs_by_skill = prepare(train_df)
        test_seqs_by_skill = prepare(test_df)

        tic = time.perf_counter()

        params = fit_bkt(train_seqs_by_skill, True)
        
        all_obs, all_probs = test_bkt(params, test_seqs_by_skill)
        toc = time.perf_counter()

        loglik = all_obs * np.log(all_probs) + (1-all_obs) * np.log(1-all_probs)

        print("Test loglik: %0.4f" % np.sum(loglik))

        auc_roc = sklearn.metrics.roc_auc_score(all_obs, all_probs)
        auc_pr = sklearn.metrics.average_precision_score(all_obs, all_probs)
        bacc = sklearn.metrics.balanced_accuracy_score(all_obs, all_probs >= 0.5)
        rand_probs = all_probs.copy()
        np.random.shuffle(rand_probs)
        auc_pr_null = sklearn.metrics.average_precision_score(all_obs, rand_probs)
        print("Test AUC-ROC: %0.2f, AUC-PR: %0.2f (Null: %0.2f)" % (auc_roc, auc_pr, auc_pr_null))
        
        row = {
            "auc_roc" : auc_roc,
            "auc_pr" : auc_pr,
            "auc_pr_null" : auc_pr_null,
            "bacc" : bacc,
            "time_diff_sec" : toc - tic
        }
        results.append(row)
        print(pd.DataFrame(results))

        params_df = pd.DataFrame.from_dict(params, orient='index')
        params_df.columns = ['pT', 'pF', 'pG', 'pS', 'pL0']
        params_df['skill'] = params_df.index 
        params_df = params_df.reset_index(drop=True)
        params_df['split'] = split_id 
        param_dfs.append(params_df)
    

    params_path = output_path.replace('.csv','.params.csv')
    pd.concat(param_dfs, axis=0, ignore_index=True).to_csv(params_path, index=False)
        

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(output_path, index=False)

def prepare(df):
    """ prepares data to be fitted by multiple BKT models, one per skill """
    seqs_by_skill_student = {}

    skill_set = set(df['skill'])
    student_set = set(df['student'])

    skills = np.array(df['skill'])
    students = np.array(df['student'])
    corrects = np.array(df['correct'])

    for i in range(len(corrects)):
        skill = skills[i]
        student = students[i] 
        correct = corrects[i]

        if skill not in seqs_by_skill_student:
            seqs_by_skill_student[skill] = {}
        
        if student not in seqs_by_skill_student[skill]:
            seqs_by_skill_student[skill][student] = []
        
        seqs_by_skill_student[skill][student].append((i, correct))

    by_skill = {}
    for skill in seqs_by_skill_student:
        by_skill[skill] = List()
        for student, seq in seqs_by_skill_student[skill].items():
            by_skill[skill].append(np.array(seq))

    return by_skill

def fit_bkt(seqs_by_skill, with_forgetting):
    """ fit one BKT model per skill """

    
    points = np.linspace(0.01, 0.99, 5)

    if with_forgetting:
        search_space = np.array(list(itertools.product(points, points, points, points, points)))
    else:
        search_space = np.array(list(itertools.product(points, [0], points, points, points)))
    
    p_by_skill = {}
    for skill in sorted(seqs_by_skill.keys()):
        seqs = seqs_by_skill[skill]
        best_p = fit_brute(seqs, search_space)
        p_by_skill[skill] = best_p
        print("Finished skill %d" % skill)
        
    return p_by_skill

@jit(nopython=True)
def fit_brute(seqs, search_space):
    """ optimize BKT using a brute force strategy """

    best_p = np.zeros(search_space.shape[1])
    best_ll = -np.inf
    for i in range(search_space.shape[0]):
        loglik = 0.0
        for seq in seqs:
            probs = forward_bkt(seq, search_space[i,0], search_space[i,1], search_space[i,2], search_space[i,3], search_space[i,4])
            y = seq[:,1]
            ll = np.sum(y * np.log(probs) + (1-y) * np.log(1-probs))
            loglik += ll 
                        
        if loglik > best_ll:
            best_ll = loglik
            best_p = search_space[i,:]
    
    return best_p 

@jit(nopython=True)
def forward_bkt(seq, pT, pF, pG, pS, pL0):
    """ computes the likelihood of a sequence, given BKT parameters """
    
    probs = np.zeros(seq.shape[0])
    pL = pL0
    npL = 0.0
    for i in range(seq.shape[0]):
        
        prob_correct = pL * (1.0-pS) + (1.0-pL) * pG
        
        if seq[i,1] == 1:
            npL = (pL * (1.0 - pS)) / (pL * (1.0 - pS) + (1.0 - pL) * pG)
        else:
            npL = (pL * pS) / (pL * pS + (1.0 - pL) * (1.0 - pG))
        pL = npL * (1-pF) + (1.0-npL) * pT
        
        probs[i] = prob_correct
    
    probs = np.clip(probs, 0.01, 0.99)
    
    return probs

def test_bkt(params, seqs_by_skill):
    all_probs = []
    all_obs = []
    for skill in seqs_by_skill:
        p = params[skill]
        for seq in seqs_by_skill[skill]:
            probs = forward_bkt(seq, p[0], p[1], p[2], p[3], p[4])
            all_probs.extend(probs)
            all_obs.extend(seq[:,1])


    return np.array(all_obs), np.array(all_probs)

if __name__ == "__main__":
    main()

    # probs = forward_bkt(np.array([[0, 1], [1, 1], [2, 0], [3, 1]]), 0.2, 0.2, 0.2, 0.2, 0.2)
    # print(probs)
    # probs = forward_bkt_python([[0, 1], [1,1], [2, 0], [3, 1]], 0.2, 0.2, 0.2, 0.2, 0.2)
    # print(probs)
    #print(search_space)