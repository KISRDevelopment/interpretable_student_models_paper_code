import tensorflow as tf
import sequence_funcs as sf 
import utils 
import tensorflow.keras as keras 
import numpy as np
import cell_ldkt
import student_model

def create_model(cfg, df):

    max_kc_id = np.max(df['skill'])
    return LdktModel(max_kc_id, cfg)
    
class LdktModel(student_model.StudentModel):

    def __init__(self, max_kc_id, cfg):

        self.max_kc_id = max_kc_id
        self.placeholder_kc_id = self.max_kc_id + 1
        self.n_kcs = self.placeholder_kc_id + 1
        
        super().__init__(cfg)

    def _init_model_components(self, components):
        
        self._rnn_module = cell_ldkt.LdktCell(self.n_kcs, True, 1)

        components.extend([self._rnn_module])
    
    
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
        
        ypred = self._rnn_module(
            features.prev_skill, 
            features.prev_corr, 
            features.curr_skill, 
            np.zeros((features.prev_corr.shape[0], features.prev_corr.shape[1], 1)),
            np.zeros((features.prev_corr.shape[0], features.prev_corr.shape[1], 1)),
            new_seqs)

        return ypred 

    def _create_feature_transformer(self):
        """
            Transforms features into numeric arrays
        """
        return sf.create_kt_transformer(self.n_kcs)
