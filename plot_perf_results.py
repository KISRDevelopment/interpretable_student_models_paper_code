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
def main(path):

    df = load_results(path)
    df.sort_values('model', inplace=True)
    df = df[~pd.isna(df['time_diff_sec'])]
    f, axes = plt.subplots(1, 2, figsize=(20, 5))

    plot_curves(axes[0], df, 'time_diff_sec', 'Mean Time (sec)', 'Exectuion Time (Train+Predict)', True, yticks=np.arange(0, 351, 50))
    plot_curves(axes[1], df, 'auc_roc', 'Mean AUC-ROC', 'Test AUC-ROC', ylim=[0.7, 0.8])
    
    plt.savefig("tmp/results_perf.png", bbox_inches='tight', dpi=120)

def plot_curves(ax, df, col, ylabel, title, show_legend=False, ylim=None, yticks=None):
    models = list(MODEL_NAMES.keys())
    for model in models:
        ix = df['model'] == model 
        sdf = df[ix]
        
        mean_sec = sdf.groupby('n_students')[col].agg('mean')
        std_sec = sdf.groupby('n_students')[col].agg(lambda vals: np.std(vals,ddof=1)/np.sqrt(len(vals)))
        
        ax.errorbar(mean_sec.index, mean_sec, 
            label=MODEL_NAMES[model],
            yerr=std_sec, markersize=5, marker='o', elinewidth=2, linewidth=2, capsize=2.5)
    
    ax.grid(True, linestyle='--')
    ax.set_xlabel('# of Students', fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.xaxis.set_tick_params(labelsize=22, rotation=90)
    ax.yaxis.set_tick_params(labelsize=22)

    if show_legend:
        ax.legend(fontsize=22, frameon=False)
    
    if ylim:
        ax.set_ylim(ylim)
    
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_title(title, fontsize=32)
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
    
    df = pd.concat(dfs, axis=0)
    df.rename(columns={
        'Unnamed: 0' : 'split'
    }, inplace=True)

    return df

if __name__ == "__main__":
    import sys
    main(sys.argv[1])


