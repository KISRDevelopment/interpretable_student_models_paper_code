import numpy as np 
import sklearn.metrics

def recovered(ref_clustering, pred_clustering, thres=0.9):

    # ref_M = make_membership_profile(ref_clustering) # C_ref x P
    # pred_M = make_membership_profile(pred_clustering) # C_pred x P

    r_clusters = np.unique(ref_clustering)
    p_clusters = np.unique(pred_clustering)


    R = np.zeros((r_clusters.shape[0], p_clusters.shape[0]))
    
    print("Processing ", R.shape)
    counts = np.zeros(r_clusters.shape[0])
    for i in range(R.shape[0]):
        u = ref_clustering == r_clusters[i]
        counts[i] = np.sum(u)
        for j in range(R.shape[1]):
            dst = fast_bacc(u, pred_clustering == p_clusters[j])
            R[i, j] = dst 
    print("Done ", R.shape)
    
    R = np.sum(counts * (np.max(R, axis=1) >= thres)) / ref_clustering.shape[0]
    
    return R 

def fast_bacc(ytrue, ypred):

    recall00 = np.sum((ytrue == 0) & (ypred == 0)) / np.sum(ytrue == 0)
    recall11 = np.sum((ytrue == 1) & (ypred == 1)) / np.sum(ytrue == 1)

    return (recall00 + recall11) / 2

def make_membership_profile(clustering):

    r = np.unique(clustering)
    
    M = np.zeros((r.shape[0], len(clustering)))
    for i in range(M.shape[0]):
        M[i, :] = clustering == r[i]
    

    return M 

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
    f = fmeasure([0, 0, 1, 2, 3, 3], [1, 1, 2, 0, 5, 10])
    print(f)
    
    R = recovered(np.array([0, 0, 1, 2, 3, 3]), np.array([1, 1, 2, 0, 5, 10]), thres=0.9)
    print(R)

    a, b = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1]), np.array([0, 1, 0,  0, 0, 0, 0, 0, 1])
    x = sklearn.metrics.balanced_accuracy_score(a, b)
    y = fast_bacc(a, b)

    print(x, y)

    x = sklearn.metrics.balanced_accuracy_score(a, np.zeros_like(a))
    y = fast_bacc(a, np.zeros_like(a))

    print(x, y)
if __name__ == "__main__":
    main()