import numpy as np 
import pandas as pd 
from collections import defaultdict
import matplotlib.pyplot as plt 

def main():

    actual_params = "data/perf_data_probs.npy"
    actual_params = np.load(actual_params)
    n_kcs = actual_params.shape[0]

    ns_students = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    col_labels = ['pI0', 'pL', 'pF', 'pG', 'pS']

    all_diffs = []
    for n_students in ns_students:
        #diffs_by_kc = get_torch_bkt_probs("data/results-perf/torch-bkt_perf_%d.params.npy.npz" % n_students, actual_params)
        diffs_by_kc = get_brute_force_bkt_probs("data/results-perf/bkt-brute-force_perf_%d.params.csv" % n_students, actual_params)
        
        all_diffs.append(diffs_by_kc)
    
    all_diffs = np.array(all_diffs)
    
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    for c in np.arange(all_diffs.shape[1]):
        col = all_diffs[:,c]
        ax.plot(ns_students, col, label=col_labels[c])
    
    ax.legend(fontsize=22, frameon=False)
    f.savefig('tmp/figure_param_recovery.png', bbox_inches='tight', dpi=120)

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
    return np.mean(np.mean(diffs_by_kc, axis=1), axis=0)

def get_torch_bkt_probs(params_path, actual_params):
    learned_params = np.load(params_path)
    alpha, obs, t = learned_params['alpha'], learned_params['obs'], learned_params['t']

    n_folds = alpha.shape[0]
    n_kcs = actual_params.shape[0]
    diffs_by_kc = []
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
        diffs_by_kc.append(diff)
        
    diffs_by_kc = np.mean(diffs_by_kc, axis=0)
    
    return diffs_by_kc
    
if __name__ == "__main__":
    main()
