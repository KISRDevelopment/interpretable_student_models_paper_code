import tensorflow as tf
import sequence_funcs as sf 
import utils 
import tensorflow.keras as keras 
import numpy as np
import cell_ldkt
import student_model
import tensorflow_probability as tfp

def create_model(cfg, df):

    max_kc_id = np.max(df['skill'])
    return LdktSkillDiscoveryModel(max_kc_id, cfg)
    
class LdktSkillDiscoveryModel(student_model.StudentModel):

    def __init__(self, max_kc_id, cfg):

        self.max_kc_id = max_kc_id
        self.placeholder_kc_id = self.max_kc_id + 1
        self.n_kcs = self.placeholder_kc_id + 1
        self.n_groups = cfg['n_groups']
        self.temperature = cfg['temperature']
        self.n_samples = cfg['n_samples']

        super().__init__(cfg)

    def _init_model_components(self, components):
        self._kca_module = KCAssignmentModule(self.n_kcs, self.n_groups)
        
        self._rnn_module = cell_ldkt.LdktCell(self.n_groups, True, 1)

        components.extend([self._kca_module, self._rnn_module])
    
    
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
        
        # get KC assignments [n_kcs, n_groups]
        if testing:
            S = self._S
        else:
            S = self._kca_module(self.temperature)
        
        # apply assignments 
        prev_skill = tf.matmul(features.prev_skill, S)
        curr_skill = tf.matmul(features.curr_skill, S)

        ypred = self._rnn_module(
            prev_skill, 
            features.prev_corr, 
            curr_skill, 
            np.zeros((features.prev_corr.shape[0], features.prev_corr.shape[1], 1)),
            np.zeros((features.prev_corr.shape[0], features.prev_corr.shape[1], 1)),
            new_seqs)

        return ypred 


    def evaluate(self, seqs):

        losses = []
        for i in range(self.n_samples):
            self._S = self._kca_module(1e-6)
            loss = super().evaluate(seqs)
            losses.append(loss)
        
        return np.mean(losses)
    
    def predict_seqs(self, n, seqs):

        all_preds = np.zeros((self.n_samples, n))
        for i in range(self.n_samples):
            self._S = self._kca_module(1e-6)
            all_preds[i,:] = super().predict_seqs(n, seqs)
        
        return np.mean(all_preds, axis=0)


    def _create_feature_transformer(self):
        """
            Transforms features into numeric arrays
        """
        return sf.create_kt_transformer(self.n_kcs)

class KCAssignmentModule:


    def __init__(self, n_kcs, n_groups):
        self.n_kcs = n_kcs
        self.n_groups = n_groups 
        
        self.logit_probs_kc_assignment = tf.Variable(tf.random.normal((n_kcs, n_groups), mean=0, stddev=0.1), name="logit_probs_kc_assignment")

        self.trainables = [
            ('logit_probs_kc_assignment', self.logit_probs_kc_assignment)
        ]
    
    def get_trainables(self, new_seqs):
        return self.trainables
    
    def __call__(self, temperature):
        dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=self.logit_probs_kc_assignment)
        
        # sample an assignment [n_kcs, n_groups]
        S = dist.sample()

        return S 

    