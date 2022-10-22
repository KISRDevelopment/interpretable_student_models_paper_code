import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import glob 
import os 

def main(path):

    reported_df = load_reported_results()
    df = load_results(path)
    df = pd.concat((df, reported_df), axis=0, ignore_index=True)
    df.to_csv("tmp/results-model-comp.csv", index=False)
    print(df)

def load_reported_results(n_splits=5):
    df = pd.read_csv("reported_results_gervetetal.csv")
    df = pd.melt(df, id_vars='model', value_vars=df.columns[1:], var_name='dataset', value_name='auc_roc')
    dfs = []
    for s in range(n_splits):
        sdf = df.copy()
        sdf['split'] = 'Split %d' % s
        dfs.append(sdf)
    
    df = pd.concat(dfs, axis=0, ignore_index=True)
    
    return df
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import glob 
import os 

def main(path):

    reported_df = load_reported_results()
    df = load_results(path)
    df = df[df['split'] != 'Overall']
    df = pd.concat((df, reported_df), axis=0, ignore_index=True)
    df.to_csv("tmp/results-model-comp.csv", index=False)
    print(df)
    print(set(df['dataset']))
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

        dfs.append(df)
    
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.rename(columns={
        'Unnamed: 0' : 'split'
    }, inplace=True)

    return df

def load_reported_results(n_splits=5):
    df = pd.read_csv("reported_results_gervetetal.csv")
    df = pd.melt(df, id_vars='model', value_vars=df.columns[1:], var_name='dataset', value_name='auc_roc')
    dfs = []
    for s in range(n_splits):
        sdf = df.copy()
        sdf['split'] = 'Split %d' % s
        dfs.append(sdf)
    
    df = pd.concat(dfs, axis=0, ignore_index=True)
    
    return df

if __name__ == "__main__":
    import sys
    main(sys.argv[1])


