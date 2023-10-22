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
    dataset_name = sys.argv[1]
    cfg_path = sys.argv[2]
    output_dir = sys.argv[3]
    use_embeddings = sys.argv[4] == '1'

    datasets = [dataset_name]
    
    os.makedirs(output_dir, exist_ok=True)

    cfg_name = os.path.basename(cfg_path).replace('.json', '')
    if use_embeddings:
        cfg_name  = cfg_name + '-rep'
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    for dataset in datasets:
        if os.path.exists("%s/%s_%s.csv"%(output_dir, cfg_name, dataset)):
            print("Ignoring %s because results already exist" % dataset)
            continue
        
        embedding_path = "./data/datasets/%s.embeddings.npy" % dataset
        print(cfg_name, dataset, embedding_path)
        output_path = "%s/%s_%s.csv" % (output_dir, cfg_name, dataset)

        command = ['python', cfg['script'], cfg_path, dataset, output_path]
        if use_embeddings:
            command += [embedding_path]
        subprocess.call(command)

        exp_cfg_path = output_path.replace('.csv', '.json')
        if use_embeddings:
            cfg['problem_feature_mat_path'] = embedding_path
        
        with open(exp_cfg_path, 'w') as f:
           json.dump(cfg, f, indent=4)
        
if __name__ == "__main__":
    main()
