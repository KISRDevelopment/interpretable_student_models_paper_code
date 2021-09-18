# 
# Linear Dynamical Knowledge Tracing (LDKT) Cell
#
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

class LdktCell(object):
    def __init__(self, n_kcs, pos_cov, n_input, **kwargs):
        self.n_kcs = n_kcs
        self.pos_cov = pos_cov
        self.n_input = n_input 

        # decay parameters 
        self.pre_G = self._create_var((n_kcs,), "G")

        # drift parameters 
        self.b = self._create_var((n_kcs, 1), "b")

        # noise covariance matrix
        self.pre_W = self._create_var((n_kcs, n_kcs), "W")
        
        # initial state mean
        self.mu0 = self._create_var((n_kcs,), 'mu0')

        # initial state covariance 
        self.pre_C0 = self._create_var((n_kcs, n_kcs), 'C0')

        # weight matrix for input variable
        self.O = self._create_var((n_input, 1), 'O')

        self.trainables_new_seqs = [
            ('pre_G', self.pre_G),
            ('b',  self.b),
            ('pre_W', self.pre_W),
            ('O', self.O),
            ('mu0', self.mu0), 
            ('pre_C0', self.pre_C0)
        ]
        self.trainables_same_seqs = self.trainables_new_seqs[:-2]
        self.trainables = self.trainables_new_seqs 

        self.states = None 

    @property
    def G(self):
        return tf.linalg.diag(tf.sigmoid(self.pre_G))
    
    @property
    def W(self):
        R = tf.matmul(self.pre_W, self.pre_W, transpose_a=True)
        if self.pos_cov:
            R = tf.abs(R)
        return R 

    @property
    def C0(self):
        R = tf.matmul(self.pre_C0, self.pre_C0, transpose_a=True)
        if self.pos_cov:
            R = tf.abs(R)
        return R 
        
    def get_trainables(self, new_seqs):
        if new_seqs:
            return self.trainables_new_seqs
        return self.trainables_same_seqs

    def _create_var(self, shape, name):
        var = tf.Variable(tf.random.truncated_normal(dtype=tf.float32, shape=shape, mean=0, stddev=0.01), name=name)
        return var

    def __call__(self, prev_skill, prev_corr, curr_skill, prev_input, curr_input, new_seqs):
        """
            prev_skill [n_batch, n_steps, n_skills]
            prev_corr  [n_batch, n_steps]
            curr_skill [n_batch, n_steps, n_skills]
            prev_input [n_batch, n_steps, n_input]
            curr_input [n_batch, n_steps, n_input]
        """

        if self.states is not None:
            n_batch = curr_skill.shape[0]
            n_diff = self.states[0].shape[0] - n_batch
            if n_diff > 0:
                self.states = [self.states[0][n_diff:,:], self.states[1][n_diff:,:]]
        
        ypred, next_states = ldkt(new_seqs, 
            tf.convert_to_tensor(prev_skill, dtype=tf.float32), 
            tf.convert_to_tensor(prev_corr, dtype=tf.float32), 
            tf.convert_to_tensor(curr_skill, dtype=tf.float32),
            tf.convert_to_tensor(prev_input, dtype=tf.float32),
            tf.convert_to_tensor(curr_input, dtype=tf.float32),
            self.G, self.b, self.W, self.O, self.mu0, self.C0,
            None if self.states is None else [tf.convert_to_tensor(self.states[0], dtype=tf.float32), 
                                              tf.convert_to_tensor(self.states[1], dtype=tf.float32)])

        self.states = [next_states[0].numpy(), next_states[1].numpy()]
        
        return ypred 
        
@tf.function(experimental_relax_shapes=True)
def ldkt(new_seqs, 
    prev_skill, 
    prev_corr,
    curr_skill,
    prev_obs_input,
    curr_obs_input,
    G, b, W, O, mu0, C0,
    states):
    
    def _fn(curr_state, curr_input):
        """
            curr_state: 
                a: [n_batch, n_kcs]
                R: [n_batch, n_kcs, n_kcs]
            
            curr_input:
                prev_skill: [n_batch, n_kcs]
                prev_corr: [n_batch, 1]
                prev_obs_input: [n_batch, n_input]

            B: n_batch
            K: n_kcs
        """

        a, R = curr_state
        prev_skill, prev_corr, prev_obs_input = curr_input

        x = prev_skill

        # Original: (1 x K) * (K x 1), Batched: (B x K) * (B x K) = (B x 1)
        
        f = tf.reduce_sum(x * a, axis=1, keepdims=True)
        h = tf.sigmoid(f + tf.matmul(prev_obs_input, O))
        h_expanded = tf.expand_dims(h,2)

        # Original: (K x K) * (K x 1) = (K x 1)
        # Batched: (B x K x K) * (B x K) = (B x K)
        Rx = tf.expand_dims(K.batch_dot(R, x, axes=[2,1]), 2, name='Rx')
            
        # Original: (1 x K) * (K x K) = (1 x K)
        # Batched: (BxK) * (BxKxK) = (BxK)
        xTR = K.batch_dot(x, R, axes=1)
        xTR_expanded = tf.expand_dims(xTR,1, name='XTR') # (Bx1xK)

        # Original: (K x 1) * (1 x K) = (KxK)
        # Batched: (B x K x 1) * (B x 1 x K) = (B x K x K)
        RxxTR = K.batch_dot(Rx, xTR_expanded, axes=[2, 1])

        # Original: (1xK) * (KxK) * (Kx1) = (1x1)
        # Batched: (BxK) * (BxK) = (Bx1)
        xTRx = tf.expand_dims(K.batch_dot(xTR, x, axes=1), 2, name='XTR')
            
        # Original: (KxK)
        # Batched: (BxKxK)
        C = R - tf.identity(h_expanded * (1-h_expanded) * RxxTR) / (1+h_expanded*(1-h_expanded)*xTRx)

        # Original: (KxK) * (Kx1) = (Kx1)
        # Batched: (BxKxK) * (BxK) = (BxK)
        Cx = K.batch_dot(C, x, axes=[2,1])

        # (BxK)
        m = a + Cx * (prev_corr - h)
            
        # Original: (KxK) * (Kx1) = (Kx1)
        # Batched: (KxK) * (BxK).T = (KxB) - transpose -> (BxK)
        Gm = tf.transpose(tf.matmul(G, m, transpose_b=True))
        
        next_a = Gm + tf.ones_like(Gm) * b[:,0]

        # (BxKxK)
        U = tf.ones_like(R) * W 
            
        #  Original: (KxK) * (KxK) = (KxK) 
        #  Batched: (KxK) * (BxKxK) = (BxKxK)
        GC = tf.tensordot(C, G, axes=[1,1])

        # Original: (KxK) * (KxK).T
        # Batched: (BxKxK) * (KxK).T = (BxKxK)
        GCGT = tf.tensordot(GC, G, axes=[2,1])

        # (BxKxK)
        next_R = GCGT + U

        return [next_a, next_R]
        
    
    # change orientation to [n_steps, n_batch, ...]
    prev_skill = tf.transpose(prev_skill, [1,0,2])
    prev_corr = tf.transpose(prev_corr)
    curr_skill = tf.transpose(curr_skill, [1,0,2])
    prev_obs_input = tf.transpose(prev_obs_input, [1, 0, 2])
    curr_obs_input = tf.transpose(curr_obs_input, [1, 0, 2])

    # get the current batch size which is <= self.n_batch
    input_shape = tf.shape(prev_skill)
    n_batch = input_shape[1]
    n_kcs = input_shape[2]

    if new_seqs:
        states = [
            tf.ones((n_batch, n_kcs)) * mu0,
            tf.ones((n_batch, n_kcs, n_kcs)) * C0
        ]
    
    # compute prediction states
    # [n_trials, n_batch, n_kcs], [n_trials, n_batch, n_kcs, n_kcs]
    pred_a, pred_R = tf.scan(_fn, [prev_skill[1:,:,:], prev_corr[1:,:,None], prev_obs_input[1:,:,:]], 
        initializer=states)
    pred_a = tf.concat((tf.expand_dims(states[0], 0), pred_a), axis=0)
    pred_R = tf.concat((tf.expand_dims(states[1], 0), pred_R), axis=0)

    # update state
    next_states = [pred_a[-1, :, :], pred_R[-1, :, :, :]] 
    
    f = tf.reduce_sum(pred_a * curr_skill, axis=2, keepdims=True)
    h = tf.sigmoid(f + tf.matmul(curr_obs_input, O))
    h = tf.clip_by_value(h, 0.001, 0.999)
    
    # [n_batch, n_steps], [n_batch, n_kcs]
    return tf.transpose(tf.squeeze(h, axis=2)), next_states

