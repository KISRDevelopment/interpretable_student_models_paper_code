import subprocess
import glob 
import os 
import json 
import pandas as pd 
import sys 

datasets = sorted([
    'gervetetal_assistments12', 
    'gervetetal_bridge_algebra06', 
    'gervetetal_assistments17', 
    'gervetetal_assistments15', 
    'gervetetal_algebra05', 
    'gervetetal_spanish', 
    'gervetetal_assistments09', 
    'gervetetal_statics'
])

def main():
    cfg_path = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)

    cfg_name = os.path.basename(cfg_path).replace('.json', '')
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    for dataset in datasets:
        if os.path.exists("%s/%s_%s.csv"%(output_dir, cfg_name, dataset)):
            print("Ignoring %s because results already exist" % dataset)
            continue
        
        print(cfg_name, dataset)
        output_path = "%s/%s_%s.csv" % (output_dir, cfg_name, dataset)

        subprocess.call(['python', cfg['script'], cfg_path, dataset, output_path])

        exp_cfg_path = output_path.replace('.csv', '.json')
        with open(exp_cfg_path, 'w') as f:
           json.dump(cfg, f, indent=4)
        
if __name__ == "__main__":
    main()
