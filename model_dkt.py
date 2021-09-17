import tensorflow as tf
import sequence_funcs as sf 
import utils 
import tensorflow.keras as keras 
import numpy as np
import cell_dkt
import student_model

def create_model(cfg, df):

    max_kc_id = np.max(df['skill'])
    return DktModel(max_kc_id, cfg)
    
class DktModel(student_model.StudentModel):

    def __init__(self, max_kc_id, cfg):

        self.max_kc_id = max_kc_id
        self.placeholder_kc_id = self.max_kc_id + 1
        self.n_kcs = self.placeholder_kc_id + 1
        self.n_hidden = cfg['n_hidden']
        super().__init__(cfg)

    def _init_model_components(self, components):
        self._rnn_module = cell_dkt.DktCell(self.n_kcs, self.n_hidden)
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

    def _run_model(self, features, new_seqs):
        """
            Executes the model
        """
        ypred = self._rnn_module(features.prev_skill, 
            features.prev_corr, features.curr_skill, 
            new_seqs, np.zeros_like(features.prev_corr)[:,:,np.newaxis])

        return ypred 

    def _create_feature_transformer(self):
        """
            Transforms features into numeric arrays
        """
        return sf.create_kt_transformer(self.n_kcs)
