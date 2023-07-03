import numpy as np 
import pandas as pd 
import glob 
import os 

def main(path, output_path):

    df = load_results(path)
    
    df.to_csv(output_path, index=False)
    print(df)

def load_results(path):

    files = glob.glob(path + "/*.csv")

    dfs = []
    for file in files:
        
        parts = os.path.basename(file).replace('.csv','').split('_')
        model, dataset = parts[0], '_'.join(parts[1:])

        df = pd.read_csv(file)
        df['model'] = model 
        df['dataset'] = dataset 

        if model.startswith('sd'):
            params_file = file.replace('.csv', '.params.npy.npz')
            
            d = np.load(params_file)
            Aprior = d['Aprior']
            pred_assignments = np.argmax(Aprior, axis=2)
            ns_unique_kcs = []
            for i in range(pred_assignments.shape[0]):
                unique_kcs = np.unique(pred_assignments[i, :])
                ns_unique_kcs.append(len(unique_kcs))
            
            df['mean_n_unique_kcs'] = np.mean(ns_unique_kcs)

        dfs.append(df)
    
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.rename(columns={
        'Unnamed: 0' : 'split'
    }, inplace=True)

    
    return df

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])


