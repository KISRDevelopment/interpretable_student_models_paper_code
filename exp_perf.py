#
#   Run performance measurement experiments
#
#   Runs the performance experiment on all datasets in data/datasets/perf_*.
#
#   Usage:
#       python exp_perf.py cfgs/exp_perf/bkt.json data/results-perf
#
import subprocess
import glob 
import os 
import sys 
import json 

def main():
    cfg_path = sys.argv[1]
    output_dir = sys.argv[2]

    datasets = [os.path.basename(p).replace('.csv','') for p in 
        glob.glob("data/datasets/perf_*")]
    datasets = sorted(datasets, key=lambda d: int(d.split('_')[1]))

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
