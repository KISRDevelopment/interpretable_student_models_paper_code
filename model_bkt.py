import tensorflow as tf
import sequence_funcs as sf 
import utils 
import tensorflow.keras as keras 
import numpy as np
import cell_bkt
import student_model

def create_model(cfg, df):

    max_kc_id = np.max(df['skill'])
    return BktModel(max_kc_id, cfg)
    
class BktModel(student_model.StudentModel):

    def __init__(self, max_kc_id, cfg):

        self.max_kc_id = max_kc_id
        self.placeholder_kc_id = self.max_kc_id + 1
        self.n_kcs = self.placeholder_kc_id + 1
        self.with_forgetting = cfg['with_forgetting']

        super().__init__(cfg)

    def _init_model_components(self, components):
        self._probs_module = StandardBktProbs(self.n_kcs, self.with_forgetting)
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
            "skill" : self.placeholder_kc_id
        })

        return seqs 

    def _run_model(self, features, new_seqs, testing=False):
        """
            Executes the model
        """
        # acquire BKT's transition and emission parameters
        probs_prev = self._probs_module(features.prev_skill)
        probs_curr = self._probs_module(features.curr_skill)

        # run BKT
        ypred = self._rnn_module(features.prev_skill, features.prev_corr, features.curr_skill, 
            new_seqs, probs_prev, probs_curr)

        return ypred 

    def _create_feature_transformer(self):
        """
            Transforms features into numeric arrays
        """
        return sf.create_kt_transformer(self.n_kcs)


class StandardBktProbs(object):

    def __init__(self, n_kcs, with_forgetting):
        self.n_kcs = n_kcs
        self.with_forgetting = with_forgetting 

        # [n_skills, 4]
        self.logit_probs = tf.Variable(tf.random.normal((self.n_kcs,4), mean=0, stddev=0.1), name="bktprobs")
        
        forgetting_mask = np.ones((self.n_kcs, 4))
        if not self.with_forgetting:
            forgetting_mask[:,1] = 0
        self.forgetting_mask = tf.convert_to_tensor(forgetting_mask.astype(np.float32))

        self.trainables = [
            ('logit_probs', self.logit_probs)
        ]
    
    def get_trainables(self, new_seqs):
        return self.trainables
    
    def __call__(self, skill):
        """ 
            skill: [n_batch, n_steps, n_skills]
            Returns:
                BKT Probabilities per skill (pL, pF, pC0, pC1) [n_batch, n_steps, 4]
        """
        return tf.matmul(skill, tf.sigmoid(self.logit_probs) * self.forgetting_mask)
        
    