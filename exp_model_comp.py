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
        "learning_rate" : 0.1, 
        "epochs" : 50, 
        "patience" : 5,
        "train_batch_seqs" : [0.1, 50, 500],
        "test_batch_seqs" : [0.5, 200, 1000],
        "cfg_name" : "bkt"
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
