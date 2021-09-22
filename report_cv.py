import json 
import numpy as np 
import sys 
from collections import defaultdict

def main(path):

    with open(path, 'r') as f:
        output = json.load(f)
    
    results = output
    print("Number of splits: %d" % len(results))
    baccs = []
    rocs = []
    cms = []

    for result in results:
        baccs.append(result['bacc'])
        rocs.append(result['auc-roc'])

        cm = np.array(result['cm'])
        cm = cm / np.sum(cm, axis=1, keepdims=True)

        cms.append(cm)
    
    mean_bacc = np.mean(baccs)
    mean_rocs = np.mean(rocs, axis=0)
    mean_cms = np.mean(cms, axis=0)

    print("Mean Balanced Accuracy: %.2f" % mean_bacc)
    print("Mean AUC-ROC:")
    print("  ", end='')
    print(mean_rocs)
    print("Mean Confusion Matrix:")
    print_matrix(mean_cms)

def print_matrix(a):

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            print("%8.2f" % a[i,j], end='')
        print()

if __name__ == "__main__":
    main(sys.argv[1])