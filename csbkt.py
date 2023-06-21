
import numpy as np 
import itertools 
import torch as th 
import torch.nn.functional as F 
from torch import Tensor
from typing import List, Tuple
import torch.nn as nn 
from numba import jit
import joint_pmf 
import layer_seq_bayesian
import sklearn.cluster

def main():
    # n_skills = 10

    # decoder = make_state_decoder_matrix(n_skills)
    # #print(decoder)
    # assert decoder.shape[0] == 2**n_skills

    # im = make_transition_indicator_matrix(decoder)
    # im = th.tensor(im).float()

    # logit_pL = th.logit(th.tensor([0.2, 0.5, 0.05]).float()[:,None])
    # logit_pF = th.logit(th.tensor([0.1, 0.3, 0.05]).float()[:,None])

    # A = make_transition_log_prob_matrix(im.numpy(), logit_pL.numpy(), logit_pF.numpy())
    # print(A.exp())
    # assert th.allclose(A.exp().sum(1), th.ones(A.shape[0]))

    # logit_pi = th.logit(th.tensor([0.2, 0.5, 0.3])[:, None])
    # initial_log_prob_vec = make_initial_log_prob_vector(decoder, logit_pi)
    # print(initial_log_prob_vec.exp().sum())
    
    # n_skills = 10
    # model = HmmCell(32, 2)
    # n_batch = 4 
    # timesteps = 7 

    # trans_log_probs = A # [n_states, n_states]
    # init_log_probs = th.tile(initial_log_prob_vec, (1, n_batch)).T # [n_batch, n_states]
    
    # obs_probs = th.softmax(th.randn(2**n_skills, 2), axis=1) # [n_states, 2]
    # obs_probs = th.tile(obs_probs[None, :, :], (timesteps, 1, 1))
    # obs_probs = th.tile(obs_probs[None, :, :, :], (n_batch, 1, 1, 1))
    # obs_log_probs = obs_probs.log()
    
    n_skills = 5
    n_batch = 4 
    timesteps = 7 
    n_problems = 20

    obs = np.random.binomial(1, 0.5, (n_batch, timesteps))
    obs = th.tensor(obs).long()

    decoder = make_state_decoder_matrix(n_skills)

    nido_layer = NIDOLayer(n_problems, th.tensor(decoder).float())
    
    problem_seq = np.tile(np.random.permutation(n_problems)[:timesteps][None,:], (n_batch, 1))
    problem_seq = th.tensor(problem_seq).long()

    model = CsbktModel({ 'n_skills' : n_skills, 'n_problems' : n_problems, 'device' : 'cpu:0', 'pred_layer' : 'nido' })
    log_prob, log_alpha = model(obs, problem_seq)
    print(log_prob.exp())
    
    ml = model.get_membership_logits()

    #print(ml)

    #print(get_effective_assignment(ml))

    ml = th.ones((n_problems, n_skills)) * (-10)
    ml[:, 0] = 10
    
    loss_layer = SequentialLossLayer(n_skills)
    output = loss_layer(problem_seq, ml)
    print(output.exp())
def make_state_decoder_matrix(k):
    """
        creates a lookup table that maps integer states into
        the binary states of each core skill.
        Arguments:
            k: the number of core skills
        
        Returns:
            decoder: (2**K, K) matrix (Pytorch)
    """
    r = list(itertools.product([0, 1], repeat=k))
    return np.array(r)

@jit(nopython=False)
def make_transition_indicator_matrix(decoder):
    """
        Returns a matrix that specifies how each skill transitions 
        skill transitions (0 -> 0, 0 -> 1, 1 -> 0, 1 -> 1)
        
        Inputs:
            decoder: the integer->state combination mapping [n_total, n_skills]
        
        Output:
            transition indictor_mat: for each source and target states of the
            HMM, specifies how each of the core n_skills transitioned.
                [n_total, n_total, n_skills, 4]
                 Source,  Target
                 (Pytorch)
    """

    n_total = decoder.shape[0]
    n_skills = decoder.shape[1]

    im = np.zeros((n_total, n_total, n_skills, 4))
    
    for i in range(n_total):
        from_h = decoder[i, :]
        for j in range(n_total):
            to_h = decoder[j, :]
            diff = to_h - from_h
            for l in range(n_skills):
                im[i, j, l, 0] = (diff[l] == 0) & (from_h[l] == 0)
                im[i, j, l, 1] = diff[l] == 1
                im[i, j, l, 2] = diff[l] == -1
                im[i, j, l, 3] = (diff[l] == 0) & (from_h[l] == 1)
    
    return im

@th.jit.script
def make_initial_log_prob_vector(decoder, logit_pi):
    """
        (Pytorch function)
        Creates the initial probability distribution over states of the full HMM.

        Inputs:
            decoder     [n_total, n_skills]
            logit_pi    [n_skills, 1]
        Output:
            log_prob of each state
    """

    logit_pi = logit_pi.T # [1, n_skills]

    logit_probs = decoder * logit_pi + (1-decoder) * (-logit_pi) # [n_total, n_skills]

    log_pinitial = F.logsigmoid(logit_probs)
    
    return log_pinitial.sum(1)[:,None]

@th.jit.script
def make_transition_log_prob_matrix(im, logit_pL, logit_pF):
    """
        (Pytorch function)

        Given the individual KC's learning and forgetting probabilities,
        this constructs the transition matrix for the full HMM.

        Inputs:
            im: transition_indicator_matrix  [n_total, n_total, n_skills, 4]
            logit_pL: logit of probability of learning    [n_skills,1]
            logit_pF: logit of probability of forgetting  [n_skills,1]
        
        Output:
            T: the log transition matrix of the full HMM [n_total, n_total]
    """
    
    # [n_skills, 4]
    log_prob_matrix = F.logsigmoid(th.concat((-logit_pL, logit_pL, logit_pF, -logit_pF), dim=1))

    # compute log probability for each skill for each state transition
    # [n_total, n_total, n_skills]
    U = (im * log_prob_matrix[None, None, :, :]).sum(3)

    # finally, compute summation to get the log transition matrix
    return U.sum(2) # [n_total, n_total]

def get_effective_assignment(membership_logits):

    skills_range = np.arange(membership_logits.shape[1])

    has_skill = membership_logits > 0

    return np.sum(has_skill * np.power(2, skills_range[None,:]), axis=1)

class CsbktModel(nn.Module):
    def __init__(self, cfg):
        """
            n_skills: # of latent skills (max around 10)
            n_problems: # of problems
        """
        super(CsbktModel, self).__init__()
        
        decoder = make_state_decoder_matrix(cfg['n_skills'])
        self.decoder = th.tensor(decoder).float().to(cfg['device'])

        self.tim = th.tensor(make_transition_indicator_matrix(decoder)).float().to(cfg['device'])
        
        n_states = self.decoder.shape[0]

        #
        # generates output predictions given state
        #
        self.pred_layer = NIDOLayer(cfg['n_problems'], self.decoder, cfg.get('problem_feature_mat', None))
        
        # 
        # BKT HMM
        #
        self.hmm = HmmCell(n_states, 2)
        
        #
        # KC parameters
        #
        self.kc_logit_pL = nn.Parameter(th.randn(cfg['n_skills']))
        self.kc_logit_pF = nn.Parameter(th.randn(cfg['n_skills']))
        self.kc_logit_pi = nn.Parameter(th.randn(cfg['n_skills']))

        self.ability_levels = th.tensor(cfg['ability_levels']).to(cfg['device']) # A
        
        self.cfg = cfg 

    def forward(self, corr, problem_seq, test=False):
        ytrue = corr
        n_batch, n_steps = corr.shape
        n_ability_levels = self.ability_levels.shape[0]

        ability_level = th.repeat_interleave(self.ability_levels, corr.shape[0]) # B'
        
        corr = th.tile(corr, (n_ability_levels, 1)) # B'xT
        problem_seq = th.tile(problem_seq, (n_ability_levels, 1)) # B'xT

        logpred, _ = self.forward_(corr, problem_seq, ability_level, test) # B'xTx2

        # reshape so that we can marginalize over abilities
        logpred = th.reshape(logpred, (n_ability_levels, n_batch, n_steps, 2))
        logpred = th.permute(logpred, (1, 0, 2, 3)) # BxAxTx2

        logpred, _ = layer_seq_bayesian.seq_bayesian(logpred, ytrue) # BxTx2

        return logpred
    def forward_(self, corr, problem_seq, ability_level, test):
        # [n_states, 1]
        initial_log_prob_vec = make_initial_log_prob_vector(self.decoder, self.kc_logit_pi[:,None])

        # [n_batch, n_states]
        initial_log_prob_vec = th.tile(initial_log_prob_vec.T, (corr.shape[0], 1)) 

        # [n_batch, t, n_states, 2]
        obs_log_probs = self.pred_layer(problem_seq, ability_level, test)

        # [n_states, n_states] (Target x Source)
        trans_log_probs = make_transition_log_prob_matrix(self.tim, self.kc_logit_pL[:,None], self.kc_logit_pF[:,None])
        
        # [n_states, 1]
        initial_log_prob_vec = make_initial_log_prob_vector(self.decoder, self.kc_logit_pi[:,None])
        
        # [n_batch, n_states]
        initial_log_prob_vec = th.tile(initial_log_prob_vec.T, (corr.shape[0], 1)) 

        # [n_batch, t, 2]
        result, log_alpha  = self.hmm(corr, trans_log_probs, obs_log_probs, initial_log_prob_vec)
        
        return result, log_alpha

    def get_membership_logits(self):
        with th.no_grad():
            return self.pred_layer.get_membership_logits().cpu().numpy()
    
class HmmCell(th.jit.ScriptModule):
    
    def __init__(self, n_states, n_outputs):
        super(HmmCell, self).__init__()
        
        self.n_states = n_states
        self.n_outputs = n_outputs
        
    @th.jit.script_method
    def forward(self, obs: Tensor,
                      trans_log_probs: Tensor, 
                      obs_log_probs: Tensor, 
                      init_log_probs: Tensor) -> Tuple[Tensor, Tensor]:
        """
            input:
                obs: [n_batch, t]
                trans_log_probs: [n_states, n_states] (Target, Source)
                obs_log_probs:   [n_batch, t, n_states, n_outputs]
                init_log_probs:  [n_batch, n_states]
            output:
                log_output_prob: [n_batch, t, n_outputs]
                log_alpha: [n_batch, n_states]
        """
        outputs = th.jit.annotate(List[Tensor], [])
        batch_idx = th.arange(init_log_probs.shape[0])

        # [n_batch, n_states]
        log_alpha = init_log_probs
        for i in range(0, obs.shape[1]):
            
            # predict
            # B X S X O + B X S X 1
            log_py = th.logsumexp(obs_log_probs[:,i,:,:] + log_alpha[:, :, None], dim=1)  #[n_batch, n_obs]
            log_py = log_py - th.logsumexp(log_py, dim=1)[:,None]
            outputs += [log_py]

            # update
            curr_y = obs[:,i]
            log_py = obs_log_probs[batch_idx, i, :, curr_y] # [n_batch, n_states]
            
            # B x 1 X S + B x 1 x S + 1 x S x S
            log_alpha = th.logsumexp(log_py[:,None,:] + log_alpha[:,None,:] + trans_log_probs[None,:,:], dim=2)
        
        outputs = th.stack(outputs)
        outputs = th.transpose(outputs, 0, 1)
        
        return outputs, log_alpha

class NIDOLayer(th.jit.ScriptModule):

    def __init__(self, n_problems, decoder, problem_feature_mat):
        super().__init__()

        n_states, n_skills = decoder.shape 

        self.skill_offset = nn.Parameter(th.randn(n_skills))
        self.skill_slope = nn.Parameter(th.randn(n_skills))

        if problem_feature_mat is None:
            self.membership_layer = ShallowProblemMembershipLayer(n_problems, n_skills)
        else:
            self.membership_layer = FeaturizedProblemMembershipLayer(problem_feature_mat, n_skills)
        
        self.decoder = decoder # [n_states, n_skills]

    @th.jit.script_method
    def get_membership_logits(self):
        return self.membership_layer.get_membership_logits()
        
    @th.jit.script_method
    def forward(self, problem_seq, ability_level, test):
        """
            Input:
                problem_seq: [n_batch, t]
                ability_level: n_batch
                test: whether this is testing or not, if testing the membership
                will be thresholded at 0.5
            Output:
                obs_log_probs: [n_batch, t, n_states, 2]
        """

        #
        # compute state logits
        # 1xK + (1xK) * (SxK) = SxK 
        # [n_states, n_skills]
        #
        state_logits = self.skill_offset[None,:] + self.skill_slope[None,:] * self.decoder

        #
        # compute problem logits
        #
        membership_probs = th.sigmoid(self.get_membership_logits()) # [n_problems, n_skills]
        if test:
            membership_probs = (membership_probs > 0.5) * 1.0
            
        problem_logits = membership_probs @ state_logits.T # [n_problems, n_states]

        obs_correct_logits = problem_logits[problem_seq] # [n_batch, t, n_states]

        # adjust observation probabilities to account for student ability
        obs_correct_logits[:, :, 0] = ability_level[:, None] + obs_correct_logits[:, :, 0] # greater ability -> greater prob of guessing
        obs_correct_logits[:, :, 1] = obs_correct_logits[:, :, 1] - ability_level[:, None] # greater ability -> smaller prob of slipping

        # [n_batch, t, n_states, 2]
        obs_log_probs = F.log_softmax(th.concat((-obs_correct_logits[:,:,:,None], obs_correct_logits[:,:,:,None]), dim=3), dim=3)

        return obs_log_probs

class ShallowProblemMembershipLayer(th.jit.ScriptModule):

    def __init__(self, n_problems, n_skills):
        super().__init__()
        self.membership_logits = nn.Parameter(th.randn(n_problems, n_skills))
    
    @th.jit.script_method
    def get_membership_logits(self):
        return self.membership_logits

class FeaturizedProblemMembershipLayer(th.jit.ScriptModule):

    def __init__(self, problem_feature_mat, n_skills):
        super().__init__()

        # perform clustering
        kmeans = sklearn.cluster.KMeans(n_clusters=n_skills, n_init='auto').fit(problem_feature_mat.cpu().numpy())

        # get initial assignment
        A = kmeans.labels_
        A = F.one_hot(th.tensor(A).long(), num_classes=n_skills).float().to('cuda:0')
        
        # initialize
        #self.logits = nn.Parameter(A * 10 - 10*(1-A))
        self.logits = A*10 - 10*(1-A)
        self.problem_feature_mat = problem_feature_mat

    @th.jit.script_method
    def get_membership_logits(self):
        return self.logits
    
if __name__ == "__main__":
    main()