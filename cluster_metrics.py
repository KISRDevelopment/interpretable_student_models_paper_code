import numpy as np 
import sklearn.metrics

def recovered(ref_clustering, pred_clustering, recall_thres, precision_thres):

    r_clusters = np.unique(ref_clustering)
    p_clusters = np.unique(pred_clustering)


    R = np.zeros((r_clusters.shape[0], p_clusters.shape[0]))
    
    print("Processing ", R.shape)
    counts = np.zeros(r_clusters.shape[0])
    for i in range(R.shape[0]):
        u = ref_clustering == r_clusters[i]
        counts[i] = np.sum(u)
        for j in range(R.shape[1]):
            R[i, j] = is_match(u, pred_clustering == p_clusters[j], recall_thres=recall_thres, precision_thres=precision_thres) 
    print("Done ", R.shape)
    
    R = np.sum(counts * (np.sum(R, axis=1) > 0)) / ref_clustering.shape[0]
    
    return R 

def is_match(ytrue, ypred, recall_thres, precision_thres):
    count11 = np.sum((ytrue == 1) & (ypred == 1))
    recall11 =    count11 / np.sum(ytrue == 1)
    precision11 = count11 / (1e-6+np.sum(ypred == 1))
    #print("Recall: %0.2f, precision: %0.2f" % (recall11, precision11))
    return (recall11 >= recall_thres) and (precision11 >= precision_thres)

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
    # C = make_contingency_table([0, 0, 1, 2, 3, 3], [1, 1, 2, 0, 5, 10])
    # print(C)
    # f = fmeasure([0, 0, 1, 2, 3, 3], [1, 1, 2, 0, 5, 10])
    # print(f)
    
    # R = recovered(np.array([0, 0, 1, 2, 3, 3]), np.array([1, 1, 2, 0, 5, 10]), thres=0.9)
    # print(R)

    # a, b = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1]), np.array([0, 1, 0,  0, 0, 0, 0, 0, 1])
    # x = sklearn.metrics.balanced_accuracy_score(a, b)
    # y = fast_bacc(a, b)

    # print(x, y)

    # x = sklearn.metrics.balanced_accuracy_score(a, np.zeros_like(a))
    # y = fast_bacc(a, np.zeros_like(a))

    # print(x, y)

    #R = recovered(np.array([0, 0, 1, 2, 3, 3]), np.array([1, 1, 2, 0, 5, 10]))
    #print(R)

    ytrue = np.zeros(500)
    ytrue[:10] = 1
    ypred = np.zeros(500)
    ypred[:5] = 1
    print(is_match(ytrue, ypred, recall_thres=0.9, precision_thres=0.9))

if __name__ == "__main__":
    main()