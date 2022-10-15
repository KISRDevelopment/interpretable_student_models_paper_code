import numpy as np 
import pandas as pd
import torch_bkt_skill_discovery
import generate_skill_discovery_data
import split_dataset
import os 

def main():
    os.makedirs("data/results-skill-discovery", exist_ok=True)

    n_students = 500
    n_skills = 20
    n_problems_per_skill = 10

    df, probs, actual_labels = generate_skill_discovery_data.main(n_problems_per_skill=n_problems_per_skill, 
        n_students=n_students, 
        n_skills=n_skills,
        no_bkt=False)
    splits = split_dataset.main(df, 5, 5)

    ns_latent_kcs = [20]
    for n_latent_kcs in ns_latent_kcs:
        cfg = {
            "learning_rate" : 0.2, 
            "epochs" : 1000, 
            "patience" : 10,
            "n_test_batch_seqs" : n_students,
            "n_batch_seqs" : n_students // 10,
            "tau" : 1.5,
            "n_latent_kcs" : n_latent_kcs,
            "n_valid_samples" : 10,
            "n_train_samples" : 50,
            "n_test_samples" : 50,
            "use_problems" : True,
            "lambda" : 0.000,
            "ref_labels" : actual_labels
        }
        results_df, all_params = torch_bkt_skill_discovery.main(cfg, df, splits[[0],:])
        Aprior = all_params['Aprior'][0]
        print(results_df)
        

if __name__ == "__main__":
    main()