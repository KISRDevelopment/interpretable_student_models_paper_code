import tensorflow as tf 
import numpy as np 

class ModuleContext:

    def __init__(self, **kwargs):
        pass 

    def on_train(self, df, train_students, valid_students):
        
        kc_item_matrix = calculate_kc_item_matrix(df)

        ix = df['student'].isin(train_students)
        train_df = df[ix]

        train_items_ix = list(set(train_df['problem']))

        self._n_kcs = kc_item_matrix.shape[0]
        self._n_items = kc_item_matrix.shape[1]
        self._kc_item_matrix = kc_item_matrix

        self._training_items = np.zeros(self._n_items, dtype=bool)
        self._training_items[train_items_ix] = True
        self._kc_train_item_matrix = self._kc_item_matrix * self._training_items

        initializer = tf.keras.initializers.GlorotNormal()

        self._logit_pC = tf.Variable(initializer((self._n_items, 2)), name="module_context_logit_pC")

        self._trainables = [self._logit_pC]

    def get_trainables(self, newseqs):
        return self._trainables

    def on_gradient_update(self):
        """
            For items never seen during training, we replace pG and pS
            with the average per KC on the training set.
        """

        if np.sum(self._training_items) == self.n_items:
            # nothing to do ... all items were seen during training
            return  

        # move to probability space
        item_probs = tf.sigmoid(self._logit_pC).numpy()
        
        # calculate per-kc mean probs
        kc_probs = np.dot(self._kc_train_item_matrix, item_probs)
        kc_probs = kc_probs / (1e-6 + np.sum(self.kc_train_item_matrix, axis=1,keepdims=True))
        
        # expand back to items
        item_kc_probs = np.dot(self._kc_item_matrix.T, kc_probs)

        # only update the entries for new test items
        item_probs[~self._training_items,:] = item_kc_probs[~self._training_items,:]

        # back to logits
        item_probs = np.clip(item_probs, 0.01, 0.99)
        item_logits = np.log( item_probs / (1 - item_probs) )

        # to tensorflow
        self._logit_pC.assign(item_logits) 

    def compute_context(self, item):

        pC = tf.sigmoid(self._logit_pC)

        trial_pC = tf.matmul(item, pC)

        return trial_pC 

    
def calculate_kc_item_matrix(df):
    
    max_kc_id = np.max(df['skill'])
    max_item_id = np.max(df['problem'])

    n_kcs = max_kc_id + 1
    n_items = max_item_id + 1

    m = np.zeros((n_kcs, n_items))
    for skill, problem in zip(df['skill'], df['problem']):
        m[skill, problem] = 1
    
    assert np.all(np.sum(m, axis=0) == 1)
    
    return m
