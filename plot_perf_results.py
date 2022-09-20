import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import glob 
import os 

MODEL_NAMES = {
    'bkt-brute-force' : 'Brute Force',
    'torch-bkt' : 'Pytorch'
}
def main(path):

    df = load_results(path)
    df.sort_values('model', inplace=True)
    df = df[~pd.isna(df['time_diff_sec'])]
    f, ax = plt.subplots(1, 1, figsize=(10, 5))

    models = set(df['model'])
    for model in models:
        ix = df['model'] == model 
        sdf = df[ix]
        
        mean_sec = sdf.groupby('n_students')['time_diff_sec'].agg('mean')
        std_sec = sdf.groupby('n_students')['time_diff_sec'].agg(lambda vals: np.std(vals,ddof=1)/np.sqrt(len(vals)))
        
        ax.errorbar(mean_sec.index, mean_sec, 
            label=MODEL_NAMES[model],
            yerr=std_sec, markersize=5, marker='o', elinewidth=2, linewidth=2, capsize=2.5)
    
    ax.grid(True, linestyle='--')
    ax.set_xlabel('# of Students', fontsize=22)
    ax.set_ylabel('Mean Time (sec)', fontsize=22)
    ax.xaxis.set_tick_params(labelsize=22, rotation=90)
    ax.yaxis.set_tick_params(labelsize=22)
    #ax.set_xticks(mean_sec.index)
    #ax.set_xticklabels(mean_sec.index)
    ax.legend(fontsize=22, frameon=False)
    ax.set_title('BKT Implementations Performance Comparison', fontsize=32)
    plt.savefig("tmp/results_perf.png", bbox_inches='tight', dpi=120)

def load_results(path):

    files = glob.glob(path + "/*.csv")

    dfs = []
    for file in files:
        parts = os.path.basename(file).replace('.csv','').split('_')
        model, dataset = parts[0], '_'.join(parts[1:])
        df = pd.read_csv(file)
        df['model'] = model 

        _, n_students = dataset.split('_')
        n_students = int(n_students)
        df['n_students'] = n_students
        
        dfs.append(df)
    
    df = pd.concat(dfs, axis=0)
    df.rename(columns={
        'Unnamed: 0' : 'split'
    }, inplace=True)

    return df

if __name__ == "__main__":
    import sys
    main(sys.argv[1])


