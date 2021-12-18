#
# Groups N knowledge components into K groups
#
# Requires the number of clusters to group the KCs into and
# a temperature parameter to control the sharpness of the 
# softmax function.
#
# Outputs a KC assignment matrix of shape [n_kcs, n_groups]
#
import tensorflow as tf 
import tensorflow.keras as keras 
import tensorflow_probability as tfp 
import numpy as np 

class ModuleKCDiscovery:

    def __init__(self, n_groups, temperature, quantized=True, **kwargs):

        self.n_groups = n_groups 
        self.temperature = temperature
        self.quantized = quantized

    def get_trainables(self, newseqs):
        return self._trainables
    
    def on_train(self, df, train_students, valid_students):

        max_kc_id = np.max(df['skill']) 
        self.n_kcs = max_kc_id + 1

        initializer = tf.keras.initializers.GlorotNormal()

        self._logit_probs_kc_assignment = tf.Variable(initializer((self.n_kcs, self.n_groups)), name="module_kc_discovery_logit_probs_kc_assignment")

        self._trainables = [self._logit_probs_kc_assignment]
    
    def on_gradient_update(self):
        pass

    def __call__(self, testing=False):
        
        if testing:
            temperature = 1e-6
        else:
            temperature = self.temperature
        
        dist = tfp.distributions.RelaxedOneHotCategorical(temperature, 
            logits=self._logit_probs_kc_assignment)
        
        # sample an assignment [n_kcs, n_groups]
        S = dist.sample()
        
        # quantize it
        if self.quantized:
            S = quantize(S)

        return S 

@tf.custom_gradient
def quantize(x):
    """
        x: [n_kcs, n_groups]
    """
    
    n_groups = x.shape[1]

    # quantize it
    x = tf.one_hot(tf.argmax(x, axis=1), depth = n_groups)
    
    def grad(dy):
        """
            Pass the gradient straight through ...
        """
        return dy
    
    return x, grad

if __name__ == "__main__":

    m = ModuleKCDiscovery(10, 2, 0.5)
    print(m.logit_probs_kc_assignment)

    r = m(True)
    print(r)