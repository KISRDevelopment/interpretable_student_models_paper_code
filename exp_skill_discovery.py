import numpy as np 
import pandas as pd
import generate_skill_discovery_data
import split_dataset
import os 
import subprocess 
import glob 

def main():
    
    ns_skills = [1, 5, 25, 50]
    kcseq = 'blocked'
    results_dir = './data/results-sd_%s' % kcseq

    os.makedirs(results_dir, exist_ok=True)

    for n_skills in ns_skills:
        print("Number of skills = %d" % n_skills)
        dataset_file = "sd_%d_%s" % (n_skills, kcseq)
        
        #
        # BKT model that just uses problem IDs as skills
        #
        output_path = "%s/bkt-no-sd_%d.csv" % (results_dir, n_skills)
        cfg_path = "cfgs/bkt_no_sd.json"
        if os.path.exists(output_path):
            print("Ignoring %s" % output_path)
        else:
            subprocess.call(['python', "torch_bkt.py", 
                cfg_path,
                dataset_file, 
                output_path])

        #
        # Cheating BKT model with access to true Q-matrix
        #
        output_path = "%s/bkt-cheating_%d.csv" % (results_dir, n_skills)
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
        output_path = "%s/bkt-single-kc_%d.csv" % (results_dir, n_skills)
        cfg_path = "cfgs/bkt_single_kc.json"
        if os.path.exists(output_path):
            print("Ignoring %s" % output_path)
        else:
            subprocess.call(['python', "torch_bkt.py", 
                cfg_path,
                dataset_file, 
                output_path])

        #
        # KC Discovery with true problem representations
        #
        output_path = "%s/bkt-sd-rep_%d.csv" % (results_dir, n_skills)
        cfg_path = "cfgs/exp_sd_bkt-sd.json"
        if os.path.exists(output_path):
            print("Ignoring %s" % output_path)
        else:
            subprocess.call(['python', "torch_bkt_skill_discovery_representation.py", 
                cfg_path,
                dataset_file, 
                "./data/datasets/sd_%d.embeddings.npy" % n_skills,
                output_path])
        
        #
        # KC Discovery without problem representations
        #
        embds = np.load("./data/datasets/sd_%d.embeddings.npy" % n_skills)
        one_hot = np.eye(embds.shape[0])
        np.save("./tmp/sd_%d.onehot.embeddings.npy" % n_skills, one_hot)

        output_path = "%s/bkt-sd-norep_%d.csv" % (results_dir, n_skills)
        cfg_path = "cfgs/exp_sd_bkt-sd.json"
        if os.path.exists(output_path):
            print("Ignoring %s" % output_path)
        else:
            subprocess.call(['python', "torch_bkt_skill_discovery_representation.py", 
                cfg_path,
                dataset_file, 
                "./tmp/sd_%d.onehot.embeddings.npy" % n_skills,
                output_path])

    results = generate_results(results_dir)
    #results.to_csv(final_results_path, index=False)

    gdf_mean = results.groupby(['model', 'n_skills'])['auc_roc'].mean()
    gdf_stderr = results.groupby(['model', 'n_skills'])['auc_roc'].std(ddof=1) / np.sqrt(5)
    results = pd.concat((gdf_mean, gdf_stderr), axis=1)
    results.columns = ['mean', 'stderr']
    print(results)

    

def generate_results(results_dir):

    files = glob.glob("%s/*.csv" % results_dir)

    dfs = []
    for file in files:

        model, n_skills = os.path.basename(file).replace('.csv','').split('_')
        n_skills = int(n_skills)

        df = pd.read_csv(file).rename(columns={ 'Unnamed: 0' : 'split' })
        df['model'] = model 
        df['n_skills'] = n_skills
        dfs.append(df)
    
    df = pd.concat(dfs, axis=0).sort_values(['model', 'n_skills'])
    
    return df


if __name__ == "__main__":
    main()