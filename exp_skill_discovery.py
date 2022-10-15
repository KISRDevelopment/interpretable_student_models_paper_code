import numpy as np 
import pandas as pd
import torch_bkt_skill_discovery
import generate_skill_discovery_data
import split_dataset
import os 
import torch_bkt_one_hot_kcs 

def main():
    
    ns_latent_kcs = [50, 20, 15, 10, 5]

    n_epochs = 1000
    n_patience = 10

    n_students = 500
    n_skills = 10
    n_trials_per_student = 200
    n_problems_per_skill = n_trials_per_student // n_skills

    df, probs, actual_labels = generate_skill_discovery_data.main(n_problems_per_skill=n_problems_per_skill, 
        n_students=n_students, 
        n_skills=n_skills,
        no_bkt=False)
    splits = split_dataset.main(df, 5, 5)

    result_dfs = []

    baseline_results_df, _ = torch_bkt_one_hot_kcs.main({
        "learning_rate" : 0.5, 
        "epochs" : n_epochs, 
        "patience" : n_patience,
        "n_batch_seqs" : n_students // 10
    }, df, splits[[0],:])
    baseline_results_df['model'] = 'baseline'
    baseline_results_df['n_latent_kcs'] = n_skills
    baseline_results_df['adj_rand_index'] = 1

    result_dfs.append(baseline_results_df)
    print(pd.concat(result_dfs, axis=0, ignore_index=True))

    
    for n_latent_kcs in ns_latent_kcs:
        cfg = {
            "learning_rate" : 0.2, 
            "epochs" : n_epochs, 
            "patience" : n_patience,
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
        results_df, _ = torch_bkt_skill_discovery.main(cfg, df, splits[[0],:])
        results_df['n_latent_kcs'] = n_latent_kcs
        results_df['model'] = 'sd'
        
        result_dfs.append(results_df)
        print(pd.concat(result_dfs, axis=0, ignore_index=True))

    result_df = pd.concat(result_dfs, axis=0, ignore_index=True)
    result_df.to_csv("tmp/result-exp-skill-discovery.csv", index=False)
        

if __name__ == "__main__":
    main()