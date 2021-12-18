#
# Converts a time delta into a probability of knowing
#
# Uses a simple powerlaw: p(L) = A * (1+x)^(-beta)
#
import tensorflow as tf 
import numpy as np 

class ModuleTransitionProbs:

    def __init__(self, name, x0, x1, scale, **kwargs):
        self.name = name 
        self.x0 = x0 
        self.x1 = x1 
        self.scale = scale 

    def get_trainables(self, newseqs):

        return self._trainables

    def on_train(self, df, train_students, valid_students):
        
        max_kc_id = np.max(df['skill']) 
        self.n_kcs = max_kc_id + 1
 
        initializer = tf.keras.initializers.GlorotNormal()

        self._logit_y0 = tf.Variable(initializer((self.n_kcs, 1)), name="%s_logit_y0" % self.name)
        self._logit_y1 = tf.Variable(initializer((self.n_kcs, 1)), name="%s_logit_y1" % self.name)

        self._trainables = [self._logit_y0, self._logit_y1]

        self._states = None 

    def on_gradient_update(self):
        pass

    def __call__(self, prev_kc, curr_delta, newseqs, testing=False):
        
        y0 = tf.sigmoid(self._logit_y0) 
        y1 = tf.sigmoid(self._logit_y1)
        curr_delta = tf.expand_dims(curr_delta, 2)
        
        # [n_kcs, 1]
        beta = -tf.math.log(y0 / y1) / tf.math.log((1 + self.x0) / (1+self.x1))
        A = y0 / tf.pow(1 + self.x0, -beta)

        # deltas: [n_batch, n_steps, n_kcs]
        deltas = self._calculate_kc_deltas(prev_kc, curr_delta, newseqs)

        return A[:,0] * tf.pow(1 + deltas, -beta[:,0])

    def _calculate_kc_deltas(self, prev_kc, curr_delta, newseqs):
        curr_delta = curr_delta / self.scale 
        
        if newseqs:
            self._states = None 
        
        # this is used to clip the states
        # to efficiently handle variable length sequences
        # it assumes that sequences within a batch are sorted from shortest to longest
        if self._states is not None:
            n_batch = prev_kc.shape[0]
            n_diff = self._states.shape[0] - n_batch
            if n_diff > 0:
                self._states = self._states[n_diff:,:]
        
        if self._states is None:
            initial_state = tf.zeros((prev_kc.shape[0], prev_kc.shape[-1]))
        else:
            initial_state = tf.convert_to_tensor(self._states, dtype=tf.float32)

        prev_kc = tf.transpose(prev_kc, [1, 0, 2])
        curr_delta = tf.transpose(curr_delta, [1, 0, 2])

        result = tf.scan(fn, [prev_kc, curr_delta], initial_state)

        self._states = result[-1, :, :].numpy()

        result = tf.transpose(result, [1, 0, 2])

        return result 
def fn(curr_state, curr_input):
    prev_kappa, delta = curr_input
    next_state = prev_kappa * delta + (1-prev_kappa) * (curr_state + delta)
    return next_state

if __name__ == "__main__":
    import module_kc_timedeltas

    prev_kc = np.array([
        [
            [0, 0],
            [1, 0],
            [1, 0],
            [0, 1]
        ],
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [0, 1]
        ],
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [0, 1]
        ],
    ])
    deltas = np.array([
        [
            0,
            60,
            40,
            5
        ],
        [
            0,
            30,
            70,
            100
        ],
        [
            0,
            30,
            70,
            100
        ]
    ])

    m = module_kc_timedeltas.ModuleKCTimeDeltas(60)

    kc_deltas = m.compute_deltas(tf.convert_to_tensor(prev_kc, dtype=tf.float32), tf.convert_to_tensor(deltas[:,:,np.newaxis], dtype=tf.float32))   

    print(kc_deltas)

    mtp = ModuleTransitionProbs(2, "pL", 0.25, 3)

    r = mtp(None, kc_deltas)

    print(r)
    