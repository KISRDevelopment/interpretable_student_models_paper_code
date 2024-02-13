import numpy as np 
import pandas as pd 
import glob 
import os 
import json 
from collections import defaultdict

def main(path, output_path):

    df, all_assignments = load_results(path)
    
    df.to_csv(output_path, index=False)
    print(df)

    with open(output_path.replace('.csv', '.assignments.json'), 'w') as f:
        json.dump(all_assignments, f, indent=4)
    
def load_results(path):

    with open('data/datasets/equations.problem_text_to_id.json', 'r') as f:
        problem_text_to_id = json.load(f)
    with open('data/datasets/equations.problem_text_to_orig.json', 'r') as f:
        problem_text_to_orig = json.load(f)
    
    files = glob.glob(path + "/*.csv")

    dfs = []
    
    all_assignments = {}
    for file in files:
        
        parts = os.path.basename(file).replace('.csv','').split('_')
        model, dataset = parts[0], '_'.join(parts[1:])

        df = pd.read_csv(file)
        df['model'] = model 
        df['dataset'] = dataset
        
        with open(file.replace('.csv','.json'), 'r') as f:
            cfg = json.load(f)

        if cfg['script'] == 'model_sd_old.py':
            df['n_kcs'] = cfg['initial_kcs']
            
            all_assignments[model] = extract_assignments(file, problem_text_to_id, problem_text_to_orig)
            
        dfs.append(df)


    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.rename(columns={
        'Unnamed: 0' : 'split'
    }, inplace=True)

    return df, all_assignments

def extract_assignments(path, problem_text_to_id, problem_text_to_orig, thres=50):
    params_file = path.replace('.csv', '.params.npy.npz')
    d = np.load(params_file)
    Aprior = d['Aprior'] # Splits x Problems x KCs
    
    id_to_problem_text = { v:k for k,v in problem_text_to_id.items() }

    assignments_by_split = []

    for i in range(Aprior.shape[0]):
        Q = Aprior[i, :, :]

        # compute frequency of each skill
        problem_assignment = np.argmax(Q, axis=1) # P
        skills, counts = np.unique(problem_assignment, return_counts=True)
        skill_freq = np.zeros(Q.shape[1])
        skill_freq[skills] = counts
        sorted_skills = np.argsort(-skill_freq)
        
        split_assignments = np.argsort(-Q, axis=0) # PxK
        split_assignments = split_assignments[:thres, :] # 10xK
        readable_assignments = [ {
                "top_problems" : [id_to_problem_text[i] for i in split_assignments[:, k]],
                "freq" : int(skill_freq[k])
            } for k in sorted_skills ]

        assignments_by_split.append(readable_assignments)

    return assignments_by_split

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])


