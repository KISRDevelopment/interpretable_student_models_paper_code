import numpy as np 
import torch as th 
import torch.nn.functional as F 
import itertools
import numpy.random as rng 
from numba import jit
import time 
import torch_bkt 

def sigmoid(x):
    return 1/(1+np.exp(-x))
def main():

    
    # BKT parameters probabilities
    logit_pI0 = 0.5
    logit_pG = -0.5
    logit_pS = 1.0
    logit_pL = -2
    logit_pF = -2 

    probs = sigmoid(np.array([logit_pL, logit_pF, logit_pG, logit_pS, logit_pI0]))
    
    # test sequence
    seqs = generate_seqs(500, 50, probs)
    
    # limit on time steps
    N = 10

    corr = th.tensor(seqs).float()

    model = FastBkt(N, 'cuda:0')


    tic = time.perf_counter()
    test_bkt_logprob_correct = model.forward(corr.to('cuda:0'), logit_pL, logit_pF, logit_pG, logit_pS, logit_pI0)
    test_bkt_prob_corr = test_bkt_logprob_correct.exp().cpu().numpy()
    toc = time.perf_counter()

    test_bkt_time_sec = toc - tic 

    #
    # Baseline
    # 
    baseline_model = torch_bkt.BktModel(1)
    tic = time.perf_counter()
    output = baseline_model(corr.long().to('cuda:0'), th.zeros_like(corr).long().to('cuda:0'))
    toc = time.perf_counter()

    baseline_time_sec = toc - tic 

    #
    # Reference
    #
    import model_brute_force_bkt 

    tic = time.perf_counter()
    ref_bkt_prob_corr = forward_bkt(seqs, *probs)
    toc = time.perf_counter()

    ref_bkt_time_sec = toc - tic 

    print("All close? ", np.allclose(test_bkt_prob_corr, ref_bkt_prob_corr))
    

    print("Test BKT time(s): ", test_bkt_time_sec)
    print("Baseline BKT time(s): ", baseline_time_sec)
    print("Ref BKT time(s):", ref_bkt_time_sec)

    print("Test BKT mean BCE: %8.4f" % bce_loss(seqs, test_bkt_prob_corr))
    print("Ref BKT mean BCE: %8.4f" % bce_loss(seqs, ref_bkt_prob_corr))
    
    #logliks = np.array(seq) * np.log(probs) + (1-np.array(seq)) * np.log(1-probs)
    #print(np.sum(logliks))

def bce_loss(ytrue, prob):
    return np.mean(ytrue * np.log(prob) + (1-ytrue)*np.log(1-prob))

class FastBkt:

    def __init__(self, n, device):

        self.n = n 

        # all possible trajectories of length n
        trajectories = make_trajectories(n)

        # transition indecies to compute the probaiblity of a trajectory
        # 0: -pL, 1: pL, 2: pF, 3: 1-pF, 4: 1-pI0, 5: pI0 
        trans_ind = make_transition_indices(trajectories)

        # transition indecies to compute the posterior predictive over
        # the hidden state
        pred_ind = make_predictive_indices(trajectories)

        # move to torch
        self._trajectories = th.tensor(trajectories).float().to(device)
        self._trans_ind = th.tensor(trans_ind).long().to(device)
        self._pred_ind = th.tensor(pred_ind).long().to(device)
        self._device = device 

    def forward(self, corr, logit_pL, logit_pF, logit_pG, logit_pS, logit_pI0):
        """
            Input: 
                corr: trial sequence BxT
                logit_*: BKT parameter logits (pL, pF, pG, pS, pI0)
            Output:
                output_logprobs BxTx2
        """

        #
        # transition probabilities that will be indexed into
        #
        trans_logits = th.tensor([-logit_pL, 
                                  logit_pL, 
                                  logit_pF, 
                                  -logit_pF, 
                                  -logit_pI0, 
                                  logit_pI0]).to(self._device)
        trans_logprobs = F.logsigmoid(trans_logits)

        # probability of answering correctly (2**N, N)
        obs_logits_correct = self._trajectories * (-logit_pS) + (1-self._trajectories) * logit_pG
        obs_logprob_correct = F.logsigmoid(obs_logits_correct)

        # probability of answering incorrectly (2**N, N)
        obs_logits_incorrect = self._trajectories * (logit_pS) + (1-self._trajectories) * (-logit_pG)
        obs_logprob_incorrect = F.logsigmoid(obs_logits_incorrect)

        #
        # probability of each trajectory
        #

        # 2**Nx1
        initial_logprobs = self._trajectories[:, [0]] * trans_logprobs[5] + (1-self._trajectories[:, [0]]) * trans_logprobs[4]
        # 2**Nx1
        logprob_h = trans_logprobs[self._trans_ind].sum(1, keepdims=True) + initial_logprobs
        # Bx2**Nx1
        logprob_h = th.tile(logprob_h[None,:,:], (corr.shape[0], 1, 1))

        # iterate over time 
        pred_corrects = []
        pred_incorrects = []
        for i in range(0, corr.shape[1], self.n):
            from_index = i
            to_index = from_index + self.n 

            # grab slice BxN
            corr_slice = corr[:, from_index:to_index]

            # likelikelihood of observing each observation under each trajectory Bx2**NxN
            # corr_slice[:,None,:]          Bx1xN
            # obs_logprob_correct[None,:,:] 1x2**NxN
            
            obs_loglik_given_h = corr_slice[:,None,:] * obs_logprob_correct[None,:,:] + \
                (1-corr_slice[:,None,:]) * obs_logprob_incorrect[None,:,:]

            # running likelihood of prior observations Bx2**NxN
            past_loglik_given_h = obs_loglik_given_h.cumsum(2).roll(dims=2, shifts=1)
            past_loglik_given_h[:,:,0] = 0.0

            # probability of correct/incorrect for each trajectory (weighted by trajectory weight)
            # Bx2**NxN + 1x2**NxN + Bx2**Nx1 = Bx2**NxN
            pred_correct_given_h = past_loglik_given_h + obs_logprob_correct[None,:,:] + logprob_h
            pred_incorrect_given_h = past_loglik_given_h + obs_logprob_incorrect[None,:,:] + logprob_h

            # unnormed probabilities of correctness and incorrectness BxN
            pred_correct = th.logsumexp(pred_correct_given_h, dim=1)
            pred_incorrect = th.logsumexp(pred_incorrect_given_h, dim=1)
            pred_corrects.append(pred_correct)
            pred_incorrects.append(pred_incorrect)

            #
            # new state prior for next iteration
            #
            seq_loglik_given_h = obs_loglik_given_h.sum(2) # Bx2**N
            seq_loglik = seq_loglik_given_h + logprob_h[:,:,0] # Bx2**N
            next_h_one_logprob = F.logsigmoid(trans_logits[self._pred_ind])[None,:] + seq_loglik # Bx2**N
            next_h_zero_logprob = F.logsigmoid(-trans_logits[self._pred_ind])[None,:] + seq_loglik  # Bx2**N
            next_h_one_logprob = th.logsumexp(next_h_one_logprob, dim=1, keepdims=True) # Bx1
            next_h_zero_logprob = th.logsumexp(next_h_zero_logprob, dim=1, keepdims=True) # Bx1
            
            # 1x2**N * Bx1 = Bx2**N
            initial_logprobs = self._trajectories[None,:,0] * next_h_one_logprob + (1-self._trajectories[None,:,0]) * next_h_zero_logprob
            
            # 1x2**Nx1 + Bx2**Nx1 = Bx2**Nx1
            logprob_h = trans_logprobs[self._trans_ind].sum(1, keepdims=True)[None,:,:] + initial_logprobs[:,:,None]

        # BxT
        pred_corrects = th.concat(pred_corrects, dim=1)
        pred_incorrects = th.concat(pred_incorrects, dim=1)

        # BxTx2

        preds = th.concat((pred_incorrects[:,:,None], pred_corrects[:,:,None]), dim=2)
        
        logprob_correct = pred_corrects - th.logsumexp(preds, dim=2)

        # BxT
        return logprob_correct

def make_trajectories(n):
    """
        constructs all possible trajectories of binary state of
        a sequence of length n.
        Returns a matrix of size 2^n x n
    """
    trajectories = np.array(list(itertools.product(*[[0, 1]]*n)))
    
    return trajectories

def make_transition_indices(trajectories):
    """
        computes the transition indices
        Returns a matrix of size 2^n x n-1
        because it excludes the first trial.
    """

    convolved = np.zeros_like(trajectories)
    for i in range(trajectories.shape[0]):
        indices = np.convolve(trajectories[i,:], [1, 2], mode='same')
        convolved[i, :] = indices
    convolved = convolved[:,1:]

    return convolved

def make_predictive_indices(trajectories):
    """
        computes the indices to predict the transition from
        the last state.
        
    """
    target_state = np.ones(trajectories.shape[0])
    indices = trajectories[:,-1] * 2 + target_state
    return indices

@jit(nopython=True)
def forward_bkt(seqs, pT, pF, pG, pS, pL0):
    """ computes the likelihood of a sequence, given BKT parameters """
    probs = np.zeros_like(seqs)
    
    for s in range(seqs.shape[0]):
        pL = pL0
        npL = 0.0
        for i in range(seqs.shape[1]):
            
            prob_correct = pL * (1.0-pS) + (1.0-pL) * pG
            
            if seqs[s, i] == 1:
                npL = (pL * (1.0 - pS)) / (pL * (1.0 - pS) + (1.0 - pL) * pG)
            else:
                npL = (pL * pS) / (pL * pS + (1.0 - pL) * (1.0 - pG))
            pL = npL * (1-pF) + (1.0-npL) * pT
            
            probs[s, i] = prob_correct
    
    #probs = np.clip(probs, 0.01, 0.99)
    
    return probs

def generate_seqs(n_students, n_trials, probs):
    
    skc = np.zeros((n_students, n_trials))

    for s in range(n_students):
        
        state = rng.binomial(1, probs[4])
        
        pL = probs[0]
        pF = probs[1]
        pG = probs[2]
        pS = probs[3]

        for t in range(n_trials):
            
            pC = (1 - pS) * state + (1-state) * pG 
            
            ans = rng.binomial(1, pC)
            
            state = rng.binomial(1, (1 - pF) * state + (1-state) * pL)
            
            skc[s, t] = ans
        
    return skc 

if __name__ == "__main__":
    main()
