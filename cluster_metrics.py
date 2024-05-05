import numpy as np 
import sklearn.metrics

def make_contingency_table(clustering_a, clustering_b):

    r = np.unique(clustering_a)
    s = np.unique(clustering_b)

    C = np.zeros((r.shape[0], s.shape[0]))

    for i in range(C.shape[0]):
        r_assigned = clustering_a == r[i]
        for j in range(C.shape[1]):
            s_assigned = clustering_b == s[j]
            intersect = np.sum(r_assigned & s_assigned)
            C[i, j] = intersect
    
    return C 

def fmeasure(clustering_a, clustering_b):
    C = make_contingency_table(clustering_a, clustering_b)
    return _fmeasure(C)
def _fmeasure(C):
    """
        Given contingency table of reference x predicted (RxP),
        Computes f-measure per reference cluster.
    """

    O = np.zeros_like(C)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pres = C[i, j] / np.sum(C[:, j])
            recall = C[i, j] / np.sum(C[i, :])
            f = (2 * pres * recall) / (pres + recall)
            O[i, j] = f 
    
    
    N = np.sum(C, axis=1) # R

    f = np.sum(N * np.nanmax(O, axis=1)) / np.sum(N)

    return f


def main():
    C = make_contingency_table([0, 0, 1, 2, 3, 3], [1, 1, 2, 0, 5, 10])
    print(C)
    f = fmeasure(C)
    print(f)
   
if __name__ == "__main__":
    main()