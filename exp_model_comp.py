import subprocess
import glob 
import os 
import json 
import pandas as pd 

def main():
    os.makedirs('data/results-model-comp', exist_ok=True)
    
    datasets = ['gervetetal_assistments12', 
        'gervetetal_bridge_algebra06', 
        'gervetetal_assistments17', 
        'gervetetal_assistments15', 
        'gervetetal_algebra05', 
        'gervetetal_spanish', 
        'gervetetal_assistments09', 
        'gervetetal_statics']

    datasets.reverse()

    model_script = 'torch_bkt.py'
    cfg = {
        "learning_rate" : 0.5, 
        "epochs" : 100, 
        "patience" : 10,
        "n_batch_seqs" : 500,
        "n_test_batch_seqs" : 500,
        "cfg_name" : "bkt"
    }
    run_model(cfg, model_script, datasets, 'data/results-model-comp')

    model_script = 'torch_bkt_problems.py'
    cfg = {
        "learning_rate" : 0.5, 
        "epochs" : 100, 
        "patience" : 10,
        "n_batch_seqs" : 500,
        "n_test_batch_seqs" : 500,
        "cfg_name" : "bkt-problems"
    }
    run_model(cfg, model_script, datasets, 'data/results-model-comp')

    model_script = 'torch_bkt_abilities.py'
    cfg = {
        "learning_rate" : 0.5, 
        "epochs" : 100, 
        "patience" : 10,
        "n_batch_seqs" : 500,
        "n_abilities" : 5,
        "min_ability" : -3,
        "max_ability" : 3,
        "cfg_name" : "bkt-abilities"
    }
    run_model(cfg, model_script, datasets, 'data/results-model-comp')

    model_script = 'torch_bkt_irt.py'
    cfg = {
        "learning_rate" : 0.5, 
        "epochs" : 100, 
        "patience" : 10,
        "n_batch_seqs" : 500,
        "n_abilities" : 5,
        "min_ability" : -3,
        "max_ability" : 3,
        "cfg_name" : "bkt-irt"
    }
    run_model(cfg, model_script, datasets, 'data/results-model-comp')

    model_script = 'torch_bkt_skill_discovery.py'
    cfg = {
        "learning_rate" : 0.1, 
        "epochs" : 20, 
        "patience" : 5,
        "tau" : 1.5,
        "n_latent_kcs" : 20,
        "lambda" : 0.00,
        "n_batch_seqs" : 20,
        "n_test_batch_seqs" : 500,
        "hard_samples" : False,
        "ref_labels" : None,
        "use_problems" : True,
        "n_initial_kcs" : 5,
        "n_valid_samples" : 50,
        "n_test_samples" : 50,
        "n_train_samples" : 1,
        "cfg_name" : "bkt-sd"
    }
    run_model(cfg, model_script, datasets, 'data/results-model-comp')

def run_model(cfg, model_script, datasets, base_path):
    with open('tmp/model_cfg.json', 'w') as f:
        json.dump(cfg, f, indent=4)
    
    cfg_name = cfg['cfg_name']
    for dataset in datasets:
        results_path = os.path.join(base_path, "%s_%s.csv" % (cfg_name, dataset))

        if os.path.exists(results_path):
            print("%s exists ... ignoring" % (results_path))
            continue
    
        print(dataset)

        subprocess.call(['python', model_script, 
            "tmp/model_cfg.json", 
            dataset, 
            results_path])

if __name__ == "__main__":
    main()
