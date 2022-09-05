import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import glob 
import os 

def main(path):

    reported_df = load_reported_results(['IRT', 'Best-LR'])

    df = load_results(path)
    df.sort_values('model', inplace=True)

    datasets = sorted(set(df['dataset']))
    
    n_cols = int(np.ceil(len(datasets)/2))
    f, axes = plt.subplots(2, n_cols, figsize=(20, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        dataset = datasets[i]
        ix = (df['dataset'] == dataset) & (df['split'] != 'Overall')
        sdf = df[ix]
        
        sdf = pd.concat((sdf, reported_df[reported_df['dataset']==dataset]), axis=0, ignore_index=True)
        plot_results(ax, sdf, 'auc_roc', '', dataset, show_xtick_labels=(i//n_cols) > 0)
    f.subplots_adjust(wspace=0.3, hspace=0.2)
    plt.savefig("tmp/results.png", bbox_inches='tight', dpi=120)

def load_reported_results(selected_models, n_splits=5):
    df = pd.read_csv("reported_results_gervetetal.csv")
    df = df[df['model'].isin(selected_models)]
    df = pd.melt(df, id_vars='model', value_vars=df.columns[1:], var_name='dataset', value_name='auc_roc')
    dfs = []
    for s in range(n_splits):
        sdf = df.copy()
        sdf['split'] = 'Split %d' % s
        dfs.append(sdf)
    
    df = pd.concat(dfs, axis=0, ignore_index=True)
    
    return df

def plot_results(ax, df, col, ylabel, title, ylim=None, show_xtick_labels=False):

    gdf = df.groupby('model', sort=False)

    error_df = gdf[col].agg('std') / np.sqrt(gdf[col].count())
    mean_df = gdf[col].agg('mean')

    #ax.bar(mean_df.index, mean_df)
    sns.barplot(x="model", y=col, data=df, ax=ax, ci=None, palette=sns.color_palette("pastel"),
        linewidth=1,
        edgecolor='black'
    )
    ax.errorbar(error_df.index, mean_df.loc[error_df.index], error_df, 
        capsize=5, capthick=2, linestyle='', fmt='o', 
        linewidth=2,
        markerfacecolor='black', markeredgecolor='black', markersize=5, ecolor='black')
    ax.set_xlabel('', fontsize=32)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.xaxis.set_tick_params(labelsize=22, rotation=90)
    ax.yaxis.set_tick_params(labelsize=22)
    ax.grid(True, linestyle='--', axis='y')
    if ylim is None:
        ylim = [np.min(mean_df)*0.95, np.max(mean_df)*1.05]
    if not show_xtick_labels:
        ax.set_xticks([])
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=28)

def load_results(path):

    files = glob.glob(path + "/*.csv")

    dfs = []
    for file in files:
        parts = os.path.basename(file).replace('.csv','').split('_')
        model, dataset = parts[0], '_'.join(parts[1:])
        dataset = dataset.replace('gervetetal_','')
        df = pd.read_csv(file)
        df['model'] = model 
        df['dataset'] = dataset 
        
        dfs.append(df)
    
    df = pd.concat(dfs, axis=0)
    df.rename(columns={
        'Unnamed: 0' : 'split'
    }, inplace=True)

    return df

if __name__ == "__main__":
    import sys
    main(sys.argv[1])


