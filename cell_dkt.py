import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class DktCell(object):
    
    def __init__(self, n_kcs, n_hidden):
        self.n_kcs = n_kcs
        self.states = None 

        self._gru = keras.layers.GRU(n_hidden, return_sequences=True, return_state=True, name='gru')
        self._output = keras.layers.Dense(n_kcs, activation='sigmoid', name='kc')

        

    def get_trainables(self, new_seqs):
        
        trainables = []
        trainables.extend([(t.name,t) for t in self._gru.trainable_variables])
        trainables.extend([(t.name,t) for t in self._output.trainable_variables])
        self.trainables = trainables

        return trainables 
        
    def __call__(self, prev_skill, prev_corr, curr_skill, new_seqs, extra_features):
        """
            prev_skill [n_batch, n_steps, n_skills]     Skill encountered at previous time step (one-hot)
            prev_corr  [n_batch, n_steps]               Whether answer at previous time step is correct or not
            curr_skill [n_batch, n_steps, n_skills]     Skill at current time step (one-hot)
            new_seqs: boolean                           Is this a batch of new sequences?
            extra_features: [n_batch, n_steps, m]       Extra input features at each time step
        """

        # this is used to clip the states
        # to efficiently handle variable length sequences
        # it assumes that sequences within a batch are sorted from shortest to longest
        if self.states is not None:
            n_batch = curr_skill.shape[0]
            n_diff = self.states.shape[0] - n_batch
            if n_diff > 0:
                self.states = self.states[n_diff:,:]
        
        if new_seqs:
            self.states = None 
        
        # form the inputs [n_batch, n_steps, 2*n_skills + m]
        incorrect_answer_to_prev_skill = prev_skill * (1 - prev_corr[:,:,None])
        correct_answer_to_prev_skill = prev_skill * prev_corr[:,:,None]
        xt = tf.concat((incorrect_answer_to_prev_skill, correct_answer_to_prev_skill, extra_features), axis=2)
        xt = tf.convert_to_tensor(xt, dtype=tf.float32)

        # run RNN [n_batch, n_steps, n_hidden]
        seq_output, next_states = self._gru(xt, 
            initial_state=None if self.states is None else tf.convert_to_tensor(self.states, dtype=tf.float32))

        # calculate KC probs [n_batch, n_steps, n_kcs]
        kc_probs = self._output(seq_output)
        
        ypred = tf.reduce_sum(kc_probs * curr_skill, axis=2)
        ypred = tf.clip_by_value(ypred, 0.001, 0.999)
    
        self.states = next_states.numpy()
        
        return ypred 
        