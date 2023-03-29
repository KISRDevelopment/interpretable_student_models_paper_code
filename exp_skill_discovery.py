import numpy as np 
import pandas as pd
import generate_skill_discovery_data
import split_dataset
import os 
import subprocess 
import glob 

def main():
    
    ns_skills = [1, 2, 5, 25, 50]
    n_students = 100
    n_problems_per_skill = 10
    same_order = False
    regenerate = False 

    #
    # generate KC data
    #
    for n_skills in ns_skills:
        if not regenerate and os.path.exists("data/datasets/skill_discovery_%d.csv" % n_skills):
            continue
        
        df, probs, actual_labels = generate_skill_discovery_data.main(n_problems_per_skill=n_problems_per_skill, 
            n_students=n_students, 
            n_skills=n_skills,
            seed=41,
            same_order=same_order)
        splits = split_dataset.main(df, 5, 5)
        
        df.to_csv("data/datasets/skill_discovery_%d.csv" % n_skills, index=False)
        np.save("data/splits/skill_discovery_%d.npy" % n_skills, splits)
    
    print("Generated datasets.")
    
    #
    # Cheating BKT model with access to true Q-matrix
    #
    for n_skills in ns_skills:
        print("Number of skills = %d, model = Cheating BKT" % n_skills)
        dataset_name = "skill_discovery_%d" % n_skills
        output_path = "data/results-sd/bkt-cheating_%d.csv" % n_skills
        cfg_path = "cfgs/bkt.json"
        if os.path.exists(output_path):
            continue
        subprocess.call(['python', "torch_bkt.py", 
            cfg_path,
            dataset_name, 
            output_path])
    
    #
    # BKT model that does not perform skill discovery (assumes problems are skills)
    #
    for n_skills in ns_skills:
        print("Number of skills = %d, model = BKT" % n_skills)
        dataset_name = "skill_discovery_%d" % n_skills
        output_path = "data/results-sd/bkt_%d.csv" % n_skills
        cfg_path = "cfgs/exp_sd_bkt.json"
        if os.path.exists(output_path):
            continue
        subprocess.call(['python', "torch_bkt.py", 
            cfg_path,
            dataset_name, 
            output_path])
    

    #
    # BKT model with skill discovery
    #
    for n_skills in ns_skills:
        print("Number of skills = %d, model = Skill Discovery BKT" % n_skills)
        dataset_name = "skill_discovery_%d" % n_skills
        output_path = "data/results-sd/bkt-sd_%d.csv" % n_skills
        cfg_path = "cfgs/exp_sd_bkt-sd.json"
        if os.path.exists(output_path):
            continue
        subprocess.call(['python', "torch_bkt_skill_discovery.py", 
            cfg_path,
            dataset_name, 
            output_path])

    
    results = generate_results()
    results.to_csv('tmp/results_exp_skill_discovery.csv', index=False)

    gdf = results.groupby(['model', 'n_skills'])['auc_roc'].mean()
    print(gdf)
def generate_results():

    files = glob.glob("data/results-sd/*.csv")

    dfs = []
    for file in files:

        model, n_skills = os.path.basename(file).replace('.csv','').split('_')
        n_skills = int(n_skills)

        df = pd.read_csv(file).rename(columns={ 'Unnamed: 0' : 'split' })
        df['model'] = model 
        df['n_skills'] = n_skills
        dfs.append(df)
    
    df = pd.concat(dfs, axis=0).sort_values(['model', 'n_skills'])
    
    return df


if __name__ == "__main__":
    main()