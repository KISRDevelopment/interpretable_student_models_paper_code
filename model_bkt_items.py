import tensorflow as tf
import sequence_funcs as sf 
import utils 
import tensorflow.keras as keras 
import numpy as np
import cell_bkt
import student_model
from collections import defaultdict, namedtuple
def create_model(cfg, df):

    # skill_problem = list(zip(df['skill'], df['problem']))
    # unique_skill_problem = set(skill_problem)

    # new_problem_ids = dict(zip(unique_skill_problem, range(len(unique_skill_problem))))
    # df['problem'] = [new_problem_ids[ skill_problem[i] ] for i in range(df.shape[0])]

    result = calculate_kc_item_matrix(df)
    
    return BktItemsModel(cfg, *result)
    
def calculate_kc_item_matrix(df):
    
    max_kc_id = np.max(df['skill'])
    max_item_id = np.max(df['problem'])

    placeholder_kc_id = max_kc_id + 1
    placeholder_item_id = max_item_id + 1
    n_kcs = placeholder_kc_id + 1
    n_items = placeholder_item_id + 1
    print("Items: %d" % n_items)
    m = np.zeros((n_kcs, n_items))
    for skill, problem in zip(df['skill'], df['problem']):
        m[skill, problem] = 1
    
    m[placeholder_kc_id, placeholder_item_id] = 1

    assert np.all(np.sum(m, axis=0) == 1)
    
    return m, placeholder_kc_id, n_kcs, placeholder_item_id, n_items  

class BktItemsModel(student_model.StudentModel):

    def __init__(self, cfg, kc_item_matrix, placeholder_kc_id, n_kcs, placeholder_item_id, n_items):

        self.kc_item_matrix = kc_item_matrix
        self.placeholder_kc_id = placeholder_kc_id
        self.n_kcs = n_kcs
        self.placeholder_item_id = placeholder_item_id
        self.n_items = n_items

        self.with_forgetting = cfg['with_forgetting']

        super().__init__(cfg)

    def _init_model_components(self, components):
        self._probs_module = ContextualizedBktProbs(self.kc_item_matrix)
        self._rnn_module = cell_bkt.BktCell(self.n_kcs)

        components.extend([self._probs_module, self._rnn_module])
    
    
    def _make_seqs(self, df):
        """
            Creates and featurizes the sequences from the dataframe
        """
        seqs = sf.make_sequences(df, 'student')
        seqs = sf.pad_seqs(seqs, self.n_batch_trials)
        seqs = sf.featurize_seqs(seqs, {
            "correct" : 0,
            "skill" : self.placeholder_kc_id,
            "problem" : self.placeholder_item_id
        })

        return seqs 

    def _run_model(self, features, new_seqs, testing=False):
        """
            Executes the model
        """
        # acquire BKT's transition and emission parameters
        probs_prev = self._probs_module(features.prev_skill, features.prev_item)
        probs_curr = self._probs_module(features.curr_skill, features.curr_item)

        # run BKT
        ypred = self._rnn_module(features.prev_skill, features.prev_corr, features.curr_skill, 
            new_seqs, probs_prev, probs_curr)

        return ypred 

    def _create_feature_transformer(self):
        """
            Transforms features into numeric arrays
        """
        return create_kt_item_transformer(self.n_kcs, self.n_items)

    def _on_before_training(self, train_df, valid_df):
        ix = list(set(train_df['problem']))
        training_items = np.zeros(self.n_items, dtype=bool)
        training_items[ix] = True 
        training_items[-1] = 1

        self._probs_module.set_training_items(training_items)

    def _post_update(self):
        self._probs_module.post_update()

def create_kt_item_transformer(n_kcs, n_items):

    KTItemFeatures = namedtuple('KTFeatures', 'prev_corr prev_skill prev_item curr_corr curr_skill curr_item curr_mask trial_index')

    def transformer(subseqs):
        n_batch = len(subseqs)
        n_trials = len(subseqs[0])

        prev_corr = np.zeros((n_batch, n_trials), dtype=np.float32)
        prev_skill = np.zeros((n_batch, n_trials, n_kcs), dtype=np.float32)
        prev_item = np.zeros((n_batch, n_trials, n_items), dtype=np.float32)

        curr_corr = np.zeros((n_batch, n_trials), dtype=np.float32)
        curr_skill = np.zeros((n_batch, n_trials, n_kcs), dtype=np.float32)
        curr_item = np.zeros((n_batch, n_trials, n_items), dtype=np.float32)

        curr_mask = np.zeros((n_batch, n_trials), dtype=np.float32)
        trial_index = np.zeros((n_batch, n_trials), dtype=np.int)

        for s, seq in enumerate(subseqs):
            for t, elm in enumerate(seq):
                prev_corr[s, t] = elm['prev_correct']
                curr_corr[s, t] = elm['curr_correct']
            
                curr_mask[s, t] = not elm['__padding__']

                prev_skill[s, t, elm['prev_skill']] = 1
                curr_skill[s, t, elm['curr_skill']] = 1
                
                prev_item[s, t, elm['prev_problem']] = 1
                curr_item[s, t, elm['curr_problem']] = 1

                trial_index[s, t] = elm['__index__']
        return KTItemFeatures(prev_corr, prev_skill, prev_item, curr_corr, curr_skill, curr_item, curr_mask, trial_index)
    
    return transformer

class ContextualizedBktProbs(object):

    def __init__(self, kc_item_matrix):
        n_kcs, n_items = kc_item_matrix.shape 

        self.n_kcs = n_kcs
        self.n_items = n_items 
        self.kc_item_matrix = kc_item_matrix
        
        # [n_skills, 4]
        self.logit_output_probs = tf.Variable(tf.random.normal((self.n_items,2), mean=0, stddev=0.1), name="item_output_probs")
        self.logit_state_probs = tf.Variable(tf.random.normal((self.n_kcs,2), mean=0, stddev=0.1), name="bktprobs")
        
        self.trainables = [
            ('logit_state_probs', self.logit_state_probs),
            ('logit_output_probs', self.logit_output_probs)
        ]
    
    def set_training_items(self, training_items):
        self.kc_train_item_matrix = self.kc_item_matrix * training_items
        self.training_items = training_items

    def post_update(self):
        """
            For items never seen during training, we replace pG and pS
            with the average per KC on the training set.
        """

        # move to probability space
        item_probs = tf.sigmoid(self.logit_output_probs).numpy()
        
        # calculate per-kc mean probs
        kc_probs = np.dot(self.kc_train_item_matrix, item_probs)
        kc_probs = kc_probs / np.sum(self.kc_train_item_matrix, axis=1,keepdims=True)
        #print(kc_probs)
        # expand back to items
        item_kc_probs = np.dot(self.kc_item_matrix.T, kc_probs)

        # only update the entries for new test items
        item_probs[~self.training_items,:] = item_kc_probs[~self.training_items,:]

        # back to logits
        item_probs = np.clip(item_probs, 0.01, 0.99)
        item_logits = np.log( item_probs / (1 - item_probs) )

        self.logit_output_probs.assign(item_logits)


    def get_trainables(self, new_seqs):
        return self.trainables
    
    def __call__(self, skill, item):
        """ 
            skill: [n_batch, n_steps, n_skills]
            item: [n_batch, n_steps, n_items]

            Returns:
                BKT Probabilities per skill (pL, pF, pC0, pC1) [n_batch, n_steps, 4]
        """

        output_probs = tf.matmul(item, tf.sigmoid(self.logit_output_probs))
        state_probs = tf.matmul(skill, tf.sigmoid(self.logit_state_probs))

        return tf.concat((state_probs, output_probs), axis=2)
        
    