import numpy as np 
import pandas as pd 
from collections import defaultdict
import utils 

def main(path):

    df = pd.read_csv(path)
    seqs = utils.to_seqs(df)

    print("# sequences: %d" % len(seqs))

    n_kcs = np.max(df['skill']) + 1

    cols = defaultdict(list)
    for _, seq in seqs.items():
        
        # proportion of unique problems
        n_unique = np.unique(seq['problem']).shape[0]
        prop_unique = n_unique / seq['problem'].shape[0]
        cols['unique_problem_prop'].append(prop_unique)

        # kc sequencing
        cols['same_kc_rate_1'].append(np.mean(seq['kc'][1:] == seq['kc'][:-1]))
        cols['same_kc_rate_2'].append(np.mean(seq['kc'][2:] == seq['kc'][:-2]))
        cols['same_kc_rate_3'].append(np.mean(seq['kc'][3:] == seq['kc'][:-3]))

        # number of kcs exercises in sequence
        cols['prop_exercised_kcs'].append(np.unique(seq['kc']).shape[0] / n_kcs)

    print((pd.DataFrame(cols).median(0) * 100).round(1))

if __name__ == "__main__":
    import sys 
    main(sys.argv[1])
