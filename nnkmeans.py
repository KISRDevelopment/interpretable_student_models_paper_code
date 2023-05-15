import numpy as np 
import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 
import sklearn.cluster
import pandas as pd 

def main():

    cfg = {
        'n_clusters' : 5,
        'device' : 'cuda:0',
        'max_iterations' : 1000,
        'epsilon': 1e-6,
        'tau' : 0.1
    }

    X = np.load("data/datasets/sd_5.embeddings.npy")
    df = pd.read_csv("data/datasets/sd_5_blocked.csv")

    problems_to_skills = dict(zip(df['problem'], df['skill']))
    n_problems = np.max(df['problem']) + 1
    ref_labels = np.array([problems_to_skills[p] for p in range(n_problems)])

    predA = train(cfg, X)

    pred_labels = np.argmax(predA, axis=1)
    rand_index = sklearn.metrics.adjusted_rand_score(ref_labels, pred_labels)
    print("NN Kmeans Rand: ", rand_index)       

    kmeans = sklearn.cluster.KMeans(n_clusters=cfg['n_clusters'], init='k-means++', n_init='auto').fit(X)
    pred_labels = kmeans.labels_
    rand_index = sklearn.metrics.adjusted_rand_score(ref_labels, pred_labels)
    print("Sklearn Kmeans Rand: ", rand_index)       

def train(cfg, X):
    
    model = NNKmeans(cfg['epsilon'], cfg['max_iterations'])
    model = model.to(cfg['device'])
    
    #
    # run K-means++ to initialize cluster centroids
    #
    centers, indices = sklearn.cluster.kmeans_plusplus(X, n_clusters=cfg['n_clusters'])
    
    #
    # run NNKmeans
    #
    Cinit_th = th.tensor(centers).float().to(cfg['device'])
    X_th = th.tensor(X).float().to(cfg['device'])
    A_th, C_th = model(X_th, Cinit_th, tau=cfg['tau'])

    
    return A_th.cpu().numpy()

class NNKmeans(nn.Module):

    def __init__(self, epsilon, max_iterations):
        super().__init__()

        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def forward(self, X, Cinit, tau):
        """
            X: observations         n_obs x d
            Cinit: centroids        n_clusters x d
            tau: temperature
        """

        C = Cinit
        for i in range(self.max_iterations):

            #
            # compute pairwise distances    n_obs x n_clusters
            #
            pairwise_dst = -th.cdist(X, C, p=2) / tau
            
            #
            # compute attention             n_obs x n_clusters
            #
            A = th.softmax(pairwise_dst, dim=1)
            
            #
            # new centroid matrix 
            #
            attention_sums = A.sum(0) # n_clusters

            # n_obs x n_clusters x 1 * n_obs x 1 x d = n_obs x n_clusters x d
            # final: n_clusters x d
            
            Ctilde = (A[:,:,None] * X[:,None,:]).sum(0) / attention_sums[:,None]

            diff = (C - Ctilde).square().sum(1).mean()

            C = Ctilde 

            if diff < self.epsilon:
               break
            
            
        
        #print("Terminated at ", i)
        return A, C



if __name__ == "__main__":
    main()