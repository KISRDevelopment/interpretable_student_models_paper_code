import numpy as np 
import pandas as pd 
import glob 
import os 
import json 

def main(path, output_path):

    df = load_results(path)
    df.to_csv(output_path, index=False)
    #print(df)

    #gdf = df.groupby(['model', 'dataset'])['auc_roc'].mean()
    #print(gdf)
    #gdf = df.groupby(['model', 'dataset'])['auc_roc'].std()
    #print(gdf)

    gdf = df.groupby(['model', 'dataset'])[['n_train_batch_seqs','n_test_batch_seqs','n_test_samples','learning_rate','es_thres']].agg('mean').reset_index()
    print(gdf)
def load_results(path):

    files = glob.glob(path + "/*.csv")

    dfs = []
    for file in files:
        if '.params' in file:
            continue
        parts = os.path.basename(file).replace('.csv','').split('_')
        model, dataset = parts[0], '_'.join(parts[1:])
        df = pd.read_csv(file)
        df['model'] = model 
        df['dataset'] = dataset.replace('gervetetal_','')

        cfg_file = file.replace('.csv', '.json')
        if not os.path.exists(cfg_file):
            print("Couldn't find %s" % cfg_file)
            continue
        with open(cfg_file, 'r') as f:
            cfg = json.load(f)
        
        df['n_train_batch_seqs'] = cfg.get('n_train_batch_seqs', None)
        df['n_test_batch_seqs'] = cfg.get('n_test_batch_seqs', None)
        df['n_test_samples'] = cfg.get('n_test_samples', None)
        df['learning_rate'] = cfg.get('learning_rate', None)
        df['es_thres'] = cfg.get('es_thres', None)


        dfs.append(df)
    
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.rename(columns={
        'Unnamed: 0' : 'split'
    }, inplace=True)

    return df

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])


