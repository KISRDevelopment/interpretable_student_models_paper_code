import numpy as np 
import pandas as pd
import generate_skill_discovery_data
import split_dataset
import os 
import subprocess 

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

    
    exit()


if __name__ == "__main__":
    main()