import numpy as np 
import pandas as pd
import torch_bkt_skill_discovery
import generate_skill_discovery_data
import os 

def main():
    os.makedirs("data/results-skill-discovery", exist_ok=True)

    n_students = 500
    n_skills = 20

    #generate_skill_discovery_data.main(n_problems_per_skill=10, n_students=n_students, n_skills=n_skills)
    
    ns_latent_kcs = [20]
    for n_latent_kcs in ns_latent_kcs:
        cfg = {
            "learning_rate" : 0.2, 
            "epochs" : 100, 
            "patience" : 10,
            "n_test_batch_seqs" : n_students,
            "n_batch_seqs" : n_students // 10,
            "tau" : 1.5,
            "n_latent_kcs" : n_latent_kcs,
            "n_valid_samples" : 10,
            "n_train_samples" : 50,
            "n_test_samples" : 50,
            "use_problems" : True,
            "lambda" : 0.000
        }
        torch_bkt_skill_discovery.main(cfg, "sd_%d" % n_students, "data/results-skill-discovery/n_latent_kcs_%d.csv"%n_latent_kcs)

if __name__ == "__main__":
    main()