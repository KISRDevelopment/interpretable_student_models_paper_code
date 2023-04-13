import numpy as np 
import pandas as pd
import generate_skill_discovery_data
import split_dataset
import os 
import subprocess 
import glob 

def main():
    regenerate = False 

    ns_skills = [1, 2, 5, 25, 50]
    n_students = 100
    n_problems_per_skill = 10
    block_kcs = True
    has_forgetting = True
    final_results_path = 'tmp/results_exp_skill_discovery.csv'
    results_dir = "data/results-sd"
    dataset_name_tmpl = "skill_discovery_%d"

    #
    # Random KC ordering
    #
    block_kcs = False
    final_results_path = 'tmp/results_exp_skill_discovery_random.csv'
    results_dir = "data/results-sd-random"
    dataset_name_tmpl = "skill_discovery_random_%d"


    #
    # No forgetting, Random KC ordering
    #
    has_forgetting = False
    block_kcs = False 
    final_results_path = 'tmp/results_exp_skill_discovery_random_no_forgetting.csv'
    results_dir = "data/results-sd-random-no-forgetting"
    dataset_name_tmpl = "skill_discovery_random_no_forgetting_%d"

    os.makedirs(results_dir, exist_ok=True)

    #
    # generate KC data
    #
    for n_skills in ns_skills:
        dataset_file = dataset_name_tmpl % n_skills
        if not regenerate and os.path.exists("data/datasets/%s.csv" % dataset_file):
            continue
        
        df, probs, actual_labels = generate_skill_discovery_data.main(n_problems_per_skill=n_problems_per_skill, 
            n_students=n_students, 
            n_skills=n_skills,
            seed=456,
            block_kcs=block_kcs,
            has_forgetting=has_forgetting)
        splits = split_dataset.main(df, 5, 5)
        
        df.to_csv("data/datasets/%s.csv" % dataset_file, index=False)
        np.save("data/splits/%s.npy" % dataset_file, splits)
    
    print("Generated datasets.")
    
    #
    # Cheating BKT model with access to true Q-matrix
    #
    for n_skills in ns_skills:
        print("Number of skills = %d, model = Cheating BKT" % n_skills)
        dataset_file = dataset_name_tmpl % n_skills
        output_path = "%s/bkt-cheating_%d.csv" % (results_dir, n_skills)
        cfg_path = "cfgs/bkt.json"
        if os.path.exists(output_path):
            continue
        subprocess.call(['python', "torch_bkt.py", 
            cfg_path,
            dataset_file, 
            output_path])
    

    #
    # BKT model that does not perform skill discovery (assumes problems are skills)
    #
    for n_skills in ns_skills:
        print("Number of skills = %d, model = BKT" % n_skills)
        dataset_file = dataset_name_tmpl % n_skills
        output_path = "%s/bkt_%d.csv" % (results_dir, n_skills)
        cfg_path = "cfgs/exp_sd_bkt.json"
        if os.path.exists(output_path):
            continue
        subprocess.call(['python', "torch_bkt.py", 
            cfg_path,
            dataset_file, 
            output_path])
    
    

    #
    # BKT model with one KC
    #
    for n_skills in ns_skills:
        print("Number of skills = %d, model = Single KC BKT" % n_skills)
        dataset_file = dataset_name_tmpl % n_skills
        output_path = "%s/bkt-single-kc_%d.csv" % (results_dir, n_skills)
        cfg_path = "cfgs/bkt_single_kc.json"
        if os.path.exists(output_path):
            continue
        subprocess.call(['python', "torch_bkt.py", 
            cfg_path,
            dataset_file, 
            output_path])

    
    #
    # BKT model with skill discovery
    #
    for n_skills in ns_skills:
        print("Number of skills = %d, model = Skill Discovery BKT" % n_skills)
        dataset_file = dataset_name_tmpl % n_skills
        output_path = "%s/bkt-sd_%d.csv" % (results_dir, n_skills)
        cfg_path = "cfgs/exp_sd_bkt-sd.json"
        if os.path.exists(output_path):
            continue
        subprocess.call(['python', "torch_bkt_skill_discovery.py", 
            cfg_path,
            dataset_file, 
            output_path])

    #
    # BKT model with skill discovery (bootstrapped via EMA)
    #
    for n_skills in ns_skills:
        print("Number of skills = %d, model = Skill Discovery BKT with Bootstrapping" % n_skills)
        dataset_file = dataset_name_tmpl % n_skills
        output_path = "%s/bkt-sd-bootstrapped_%d.csv" % (results_dir, n_skills)
        cfg_path = "cfgs/exp_sd_bkt-sd.json"
        if os.path.exists(output_path):
            continue
        subprocess.call(['python', "torch_bkt_skill_discovery_representation.py", 
            cfg_path,
            dataset_file, 
            output_path])

    results = generate_results(results_dir)
    results.to_csv(final_results_path, index=False)

    gdf_mean = results.groupby(['model', 'n_skills'])['auc_roc'].mean()
    gdf_stderr = results.groupby(['model', 'n_skills'])['auc_roc'].std(ddof=1) / np.sqrt(5)
    results = pd.concat((gdf_mean, gdf_stderr), axis=1)
    results.columns = ['mean', 'stderr']
    print(results)

def generate_results(results_dir):

    files = glob.glob("%s/*.csv" % results_dir)

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