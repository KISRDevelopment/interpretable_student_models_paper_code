#
# Bayesian Knowledge Tracing (BKT) RNN Cell
#
import tensorflow as tf
import numpy as np

class ModuleBktCell(object):
    
    def __init__(self):
        pass 

    def on_train(self, df, train_students, valid_students):
        
        self.n_kcs = len(set(df['skill']))
        self._states = None 

        initializer = tf.keras.initializers.GlorotNormal()

        self._logit_pI = tf.Variable(initializer((self.n_kcs, 1)), name="module_bktcell_logit_pI")
        self._trainables = [self._logit_pI]

    
    def get_trainables(self, new_seqs):
        if new_seqs:
            return self._trainables
        
        return []
    

    def __call__(self, prev_skill, prev_corr, curr_skill, probs_curr, new_seqs):
        """
            prev_skill [n_batch, n_steps, n_skills]     Skill encountered at previous time step (one-hot)
            prev_corr  [n_batch, n_steps]               Whether answer at previous time step is correct or not
            curr_skill [n_batch, n_steps, n_skills]     Skill at current time step (one-hot)
            probs_curr: [n_batch, n_steps, 4]           The  pL, pF, pC0, pC1 at curr time step
            new_seqs: boolean                           Is this a batch of new sequences?
        """

        if new_seqs:
            self._states = None 

        # this is used to clip the states
        # to efficiently handle variable length sequences
        # it assumes that sequences within a batch are sorted from shortest to longest
        if self._states is not None:
            n_batch = curr_skill.shape[0]
            n_diff = self._states.shape[0] - n_batch
            if n_diff > 0:
                self._states = self._states[n_diff:,:]
        
        n_batch, n_steps, n_vars = probs_curr.shape 

        probs_prev = tf.concat((tf.zeros((n_batch, 1, n_vars)), probs_curr[:,1:,:]), axis=1)

        ypred, next_states = bkt_hmm(new_seqs,
            prev_skill, 
            prev_corr, 
            curr_skill,
            self._logit_pI,
            probs_prev,
            probs_curr,
            None if self._states is None else tf.convert_to_tensor(self._states, dtype=tf.float32))

        self._states = next_states.numpy()
        
        return ypred 
        
@tf.function(experimental_relax_shapes=True)
def bkt_hmm(new_seqs,
    prev_skill, 
    prev_corr,
    curr_skill,
    logit_pI,
    probs_prev,
    probs_curr,
    states):
    
    def _fn(curr_state, curr_input):
        """
            curr_state: p(h_(t-1) = 1| y_1, ..., y_(t-2)) [n_batch, n_skills]
            curr_input: 
                KC at previous time step [n_batch, n_skills]
                Correctness at previous time step [n_batch, 1]
                Prev pL, pF, pC0, pC1 [n_batch, 4]
        """
        prev_skill, prev_corr, prev_probs = curr_input

        pC0_batch = prev_probs[:, 2, None]
        pC1_batch = prev_probs[:, 3, None]

        # compute probability of previous steps' output
        # [n_batch, 1]
        prob_output_h0 = tf.pow(pC0_batch, prev_corr) * tf.pow(1-pC0_batch, 1-prev_corr)
        prob_output_h1 = tf.pow(pC1_batch, prev_corr) * tf.pow(1-pC1_batch, 1-prev_corr)

        # compute filtering distribution p(h_(t-1) = 1 | y_1 ... y_(t-1))
        # [n_batch, 1]
        skill_state = tf.reduce_sum(curr_state * prev_skill, axis=1, keepdims=True)
        filtering = (prob_output_h1 * skill_state) / (prob_output_h0 * (1-skill_state) + prob_output_h1 * skill_state)

        # compute prediction distribution
        # [n_batch, 1]
        # p(h_t = 1 | y_1 .. y_(t-1))
        pL_batch = prev_probs[:, 0, None]
        pF_batch = prev_probs[:, 1, None]
        prediction = pL_batch * (1 - filtering) + (1-pF_batch) * filtering

        # finally, update relevant entry in the state
        next_state = (1-prev_skill) * curr_state + prev_skill * prediction
        
        return next_state
    
    # change orientation to [n_steps, n_batch, ...]
    prev_skill = tf.transpose(prev_skill, [1,0,2])
    prev_corr = tf.transpose(prev_corr)
    curr_skill = tf.transpose(curr_skill, [1,0,2])

    # [n_steps, n_batch, 4]
    probs_prev = tf.transpose(probs_prev, [1, 0, 2])
    probs_curr = tf.transpose(probs_curr, [1, 0, 2])

    # initialize state to initial learning probability if we're starting a new batch
    # [n_batch, n_skills]

    # get the current batch size which is <= self.n_batch
    input_shape = tf.shape(prev_skill)
    n_batch = input_shape[1]
    
    if new_seqs:
        pi = tf.sigmoid(logit_pI)
        states = tf.tile(tf.transpose(pi), (n_batch, 1))
    
    
    # compute prediction states
    pred_states = tf.scan(_fn, [prev_skill[1:,:,:], prev_corr[1:,:,None], probs_prev[1:,:,:]], 
        initializer=states)
    # attach initial state to the beginning [n_steps, n_batch, n_skills]
    pred_states = tf.concat((tf.expand_dims(states, 0), pred_states), axis=0)

    # update state
    next_states = pred_states[-1, :, :]
        
    # compute the probability of the ouptut = 1. 
    # [n_steps, n_batch, 1]
    prob_h1 = tf.reduce_sum(pred_states * curr_skill, axis=2, keepdims=True)
    prob_correct_h0 = probs_curr[:, :, 2, None]
    prob_correct_h1 = probs_curr[:, :, 3, None]
    prob_correct = prob_correct_h0 * (1 - prob_h1) + prob_correct_h1 * prob_h1
    
    # go back  to [n_batch, n_steps]
    ypred = tf.squeeze(tf.transpose(prob_correct, [1,0,2]), axis=2)
    ypred = tf.clip_by_value(ypred, 0.01, 0.99)

    # [n_batch, n_steps], [n_batch, n_kcs]
    return ypred, next_states

