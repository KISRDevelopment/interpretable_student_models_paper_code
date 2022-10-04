import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import glob 
import os 


def main(path):

    df = load_results(path)
    df = df[df['split']!='Overall']
    df.to_csv("tmp/results-replearning.csv", index=False)

def load_results(path):

    files = glob.glob(path + "/*.csv")

    dfs = []
    for file in files:
        parts = os.path.basename(file).replace('.csv','').split('_')
        model, dataset = parts[0], '_'.join(parts[1:])
        dataset = dataset.replace('gervetetal_','')
        df = pd.read_csv(file)
        df['model'] = model 
        df['dataset'] = 'replearning' 
        
        dfs.append(df)
    
    df = pd.concat(dfs, axis=0)
    df.rename(columns={
        'Unnamed: 0' : 'split'
    }, inplace=True)

    return df

if __name__ == "__main__":
    import sys
    main(sys.argv[1])


