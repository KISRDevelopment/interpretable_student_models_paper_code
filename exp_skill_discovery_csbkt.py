import numpy as np 
import pandas as pd
import os 
import subprocess 
import glob 
import json 

def write_json(path, v):
    with open(path, 'w') as f:
        json.dump(v, f)
    
def main():
    
    ns_skills = [1, 5, 25, 50]
    kcseq = 'blocked'
    results_dir = './data/results-csbkt-%s' % kcseq

    os.makedirs(results_dir, exist_ok=True)

    base_cfg = {
        "n_skills" : 7,
        "pred_layer" : "nido",
        "lr" : 0.5,
        "epochs" : 100,
        "patience" : 10,
        "n_batch_seqs" : 50,
        "n_test_batch_seqs" : 50,
        "aux_loss_coeff" : 0,
        "n_hidden" : 10
    }

    for n_skills in ns_skills:
        print("Number of skills = %d" % n_skills)
        dataset_file = "sd_%d_%s" % (n_skills, kcseq)
        
        #
        # CSBKT + True Representations + Sequential Aux Loss
        #
        cfg = { **base_cfg, "pred_layer" : "featurized_nido", 'aux_loss_coeff' : 1.0 }
        run_configuration(cfg, dataset_file, 
            output_path="%s/csbkt-true_reps-aux_loss-%d.csv" % (results_dir, n_skills),
            embd_path="./data/datasets/sd_%d.embeddings.npy" % n_skills)

        #
        # CSBKT + True Representations
        #
        cfg = { **base_cfg, "pred_layer" : "featurized_nido" }
        run_configuration(cfg, dataset_file, 
            output_path="%s/csbkt-true_reps-%d.csv" % (results_dir, n_skills),
            embd_path="./data/datasets/sd_%d.embeddings.npy" % n_skills)
        
        #
        # CSBKT + Sequential Aux Loss
        #
        cfg = { **base_cfg, 'aux_loss_coeff' : 1.0 }
        run_configuration(cfg, dataset_file, 
            output_path="%s/csbkt-aux_loss-%d.csv" % (results_dir, n_skills))

        #
        # CSBKT
        #
        cfg = { **base_cfg }
        run_configuration(cfg, dataset_file, 
            output_path="%s/csbkt-%d.csv" % (results_dir, n_skills))

        #
        # Cheating BKT model with access to true Q-matrix
        #
        output_path = "%s/bkt-cheating-%d.csv" % (results_dir, n_skills)
        cfg_path = "cfgs/bkt.json"
        if os.path.exists(output_path):
            print("Ignoring %s" % output_path)
        else:
            subprocess.call(['python', "torch_bkt.py", 
                cfg_path,
                dataset_file, 
                output_path])
        
        #
        # One KC Model
        #
        output_path = "%s/bkt-single_kc-%d.csv" % (results_dir, n_skills)
        cfg_path = "cfgs/bkt_single_kc.json"
        if os.path.exists(output_path):
            print("Ignoring %s" % output_path)
        else:
            subprocess.call(['python', "torch_bkt.py", 
                cfg_path,
                dataset_file, 
                output_path])

        #
        # BKT model that just uses problem IDs as skills
        #
        output_path = "%s/bkt-no_sd-%d.csv" % (results_dir, n_skills)
        cfg_path = "cfgs/bkt_no_sd.json"
        if os.path.exists(output_path):
            print("Ignoring %s" % output_path)
        else:
            subprocess.call(['python', "torch_bkt.py", 
                cfg_path,
                dataset_file, 
                output_path])

    results = generate_results(results_dir)
    
    gdf_mean = results.groupby(['model', 'n_skills'])['auc_roc'].mean()
    gdf_stderr = results.groupby(['model', 'n_skills'])['auc_roc'].std(ddof=1) / np.sqrt(5)
    gdf2_mean = results.groupby(['model', 'n_skills'])['rand_index'].mean()
    gdf2_stderr = results.groupby(['model', 'n_skills'])['rand_index'].std(ddof=1) / np.sqrt(5)
    
    results = pd.concat((gdf_mean, gdf_stderr, gdf2_mean, gdf2_stderr), axis=1)
    results.columns = ['auc_mean', 'auc_stderr', 'rand_index_mean', 'rand_index_stderr']
    print(results)

    
def run_configuration(cfg, dataset_file, output_path, embd_path=None):
    #output_path = "%s/%s-%d.csv" % (results_dir, cfg_name, n_skills)
    if os.path.exists(output_path):
        print("Ignoring %s" % output_path)
    else:
        write_json("tmp/csbkt.json", cfg)
        args = ['python', 
                "model_csbkt.py", 
                "tmp/csbkt.json",
                dataset_file, 
                output_path]
        if embd_path is not None:
            args += [embd_path]
        subprocess.call(args)

def generate_results(results_dir):

    files = glob.glob("%s/*.csv" % results_dir)

    dfs = []
    for file in files:

        parts = os.path.basename(file).replace('.csv','').split('-')
        n_skills = int(parts[-1])

        df = pd.read_csv(file).rename(columns={ 'Unnamed: 0' : 'split' })
        df['model'] = '-'.join(parts[:-1])
        df['n_skills'] = n_skills
        dfs.append(df)
    
    df = pd.concat(dfs, axis=0).sort_values(['model', 'n_skills'])
    
    return df


if __name__ == "__main__":
    main()