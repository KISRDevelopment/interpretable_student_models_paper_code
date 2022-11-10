import numpy as np 
import pandas as pd 
from collections import defaultdict

def main(input_dir, output_path):

    actual_params = "data/perf_data_probs.npy"
    actual_params = np.load(actual_params)
    n_kcs = actual_params.shape[0]

    ns_students = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    col_labels = ['pI0', 'pL', 'pF', 'pG', 'pS']

    dfs = []
    for n_students in ns_students:
        diffs_by_kc = get_torch_bkt_probs("%s/torch-bkt_perf_%d.params.npy.npz" % (input_dir, n_students), actual_params)
        df = reshape_df(diffs_by_kc)
        df['model'] = 'torch-bkt'
        df['n_students'] = n_students
        dfs.append(df)

        diffs_by_kc = get_brute_force_bkt_probs("%s/bkt-brute-force_perf_%d.params.csv" % (input_dir, n_students), actual_params)
        df = reshape_df(diffs_by_kc)
        df['model'] = 'brute-force-bkt'
        df['n_students'] = n_students
        dfs.append(df)
    
    df = pd.concat(dfs, axis=0, ignore_index=True)
    
    df.to_csv(output_path, index=False)
def reshape_df(diffs_by_kc):
    
    n_kcs, n_folds, _ = diffs_by_kc.shape

    diffs_by_kc = np.reshape(diffs_by_kc, (diffs_by_kc.shape[0]*diffs_by_kc.shape[1], diffs_by_kc.shape[2]))

    df = pd.DataFrame(data=diffs_by_kc, columns=['pL0', 'pT', 'pF', 'pG', 'pS'])
    df['kc'] = np.repeat(np.arange(n_kcs), n_folds)
    df['fold'] = np.tile(np.arange(n_folds), n_kcs)

    return df 

def get_brute_force_bkt_probs(params_path, actual_params):
    learned_params = pd.read_csv(params_path)
    
    n_folds = np.max(learned_params['split'])+1
    n_kcs = actual_params.shape[0]
    diffs_by_kc = np.zeros((n_kcs, n_folds, 5))
    for r in learned_params.itertuples():
        skill = r.skill 
        diff = np.abs(np.array([1-r.pL0, r.pL0]) - actual_params[skill, 0])
        knowing_state = np.argmin(diff)
        if knowing_state == 1:
            print("knowing state", 1)
            diffs_by_kc[r.skill, r.split, :] = np.abs(np.array([r.pL0, r.pT, r.pF, r.pG, r.pS]) - actual_params[skill,:])
        else:
            print("knowing state", 0)
            diffs_by_kc[r.skill, r.split, :] = np.abs(np.array([1-r.pL0, r.pF, r.pT, 1-r.pS, 1-r.pG]) - actual_params[skill,:])
    return diffs_by_kc

def get_torch_bkt_probs(params_path, actual_params):
    learned_params = np.load(params_path)
    alpha, obs, t = learned_params['alpha'], learned_params['obs'], learned_params['t']

    n_folds = alpha.shape[0]
    n_kcs = actual_params.shape[0]
    diffs_by_kc = np.zeros((n_kcs, n_folds, 5))
    for k in range(n_kcs):

        probs = np.zeros(5)
        for f in range(n_folds):
            diff = np.abs(alpha[f, k, :] - actual_params[k, 0])
            knowing_state = np.argmin(diff)
            probs[0] = alpha[f, k, knowing_state]

            probs[1] = t[f, k, knowing_state, 1-knowing_state]
            probs[2] = t[f, k, 1-knowing_state, knowing_state]
            probs[3] = obs[f, k, 1-knowing_state, 1]
            probs[4] = obs[f, k, knowing_state, 0]
        
            diff = np.abs(probs - actual_params[k,:])
            diffs_by_kc[k, f, :] = diff
    
    
    return diffs_by_kc
    
if __name__ == "__main__":
    import sys 
    main(sys.argv[1], sys.argv[2])
