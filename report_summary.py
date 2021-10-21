import json 
import numpy as np 
import sys 
from collections import defaultdict
import pandas as pd 
def main(models, datasets, metric):

    rows = []
    for model in models:
        row = {
            "model" : model
        }
        for dataset in datasets:
            results_file = "results/%s__%s.json" % (model, dataset)
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            baccs = [r['bacc'] for r in results]
            rocs = [r['auc-roc'] for r in results]

            val = np.mean(baccs) if metric == 'bacc' else np.mean(rocs)

            row[dataset] = val
        rows.append(row)

    df = pd.DataFrame(rows).set_index('model')
    
    print(df)
if __name__ == "__main__":
    main( ['bkt', 'bkt-sd', 'ldkt-sd', 'dash-sd', 'dash', 'dkt'],
          ['algebra2010', 'assistment', 'junyi', 'kdd2010', 'statics2011', 'synthetic'],
          'roc' )