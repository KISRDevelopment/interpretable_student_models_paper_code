import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import glob 
import os 

MODEL_NAMES = {
    'bkt-brute-force' : 'Brute Force',
    'torch-bkt' : 'torch-bkt',
    'torch-bkt-fast' : 'torch-bkt-fast',
    'ref-bkt': 'hmm-scalable'
}

def main(path, output_path):

    df = load_results(path)
    df = df[~pd.isna(df['time_diff_sec'])]
    df.to_csv(output_path, index=False)
    print(df)

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

        n_students = dataset.split('_')[-1]
        n_students = int(n_students)
        df['n_students'] = n_students
        
        dfs.append(df)
    
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.rename(columns={
        'Unnamed: 0' : 'split'
    }, inplace=True)

    return df

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])


