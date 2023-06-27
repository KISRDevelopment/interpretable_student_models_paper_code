import numpy as np 
import pandas as pd 
import glob 
import os 

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

        n_students, n_trials_per_skill = dataset.split('_')[-2:]
        
        df['n_students'] = int(n_students)
        df['n_trials_per_skill'] = int(n_trials_per_skill)

        dfs.append(df)
    
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.rename(columns={
        'Unnamed: 0' : 'split'
    }, inplace=True)

    df.sort_values(['n_students', 'n_trials_per_skill'], inplace=True)
    return df

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])


