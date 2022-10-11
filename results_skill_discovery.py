import numpy as np 
import pandas as pd 
import sklearn.metrics 

def main():

    d = np.load("data/skill_discovery_data_params.npz")
    actual_labels = d['A']
    print(actual_labels)

    d = np.load("tmp/test.npz")
    Aprior = d['Aprior']

    Aprior = Aprior[0,:,:]
    
    pred_labels = np.argmax(Aprior, axis=1)
    print (pred_labels)

    print("Rand: %0.2f, Adjusted Rand: %0.2f" % (
        sklearn.metrics.rand_score(actual_labels, pred_labels),
        sklearn.metrics.adjusted_rand_score(actual_labels, pred_labels)
    ))
if __name__ == "__main__":
    main()
