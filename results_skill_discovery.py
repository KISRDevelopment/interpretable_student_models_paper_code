import numpy as np 
import pandas as pd 
import sklearn.metrics 
from collections import defaultdict

N_SAMPLES = 100

def main():

    d = np.load("data/skill_discovery_data_params.npz")
    actual_labels = d['A']
    print(actual_labels)

    d = np.load("data/results-skill-discovery/n_latent_kcs_27.params.npy.npz")
    Aprior = d['Aprior']
    
    cols = defaultdict(list)
    for s in range(Aprior.shape[0]):
        A = Aprior[s,:,:]
        
        for r in range(N_SAMPLES):

            pred_labels = np.zeros_like(A)
            for i in range(A.shape[0]):
                pred_labels[i, :] = np.random.multinomial(1, A[i,:])
            pred_labels = np.argmax(pred_labels, axis=1)
            print(pred_labels)
            cols['split'].append(s)
            cols['sample'].append(r)
            cols['rand'].append(sklearn.metrics.rand_score(actual_labels, pred_labels))
            cols['adj_rand'].append(sklearn.metrics.adjusted_rand_score(actual_labels, pred_labels))
        
    df = pd.DataFrame(cols)
    print(np.mean(df, axis=0))
if __name__ == "__main__":
    main()
