import tensorflow as tf
import sequence_funcs as sf 
import utils 
import tensorflow.keras as keras 
import numpy as np
import uuid 
import pandas as pd 
import numpy as np
from collections import defaultdict, namedtuple
import numpy.random as rng
from model_bkt_skill_discovery import KCAssignmentModule

import tensorflow_probability as tfp

def create_model(cfg, df):
    max_kc_id = np.max(df['skill'])
    return DashModel(max_kc_id, cfg)

class DashModel:

    def __init__(self, max_kc_id, cfg):
        self.n_batch_trials = cfg['n_batch_trials']
        self.epochs = cfg['epochs']
        self.lr = cfg['lr']
        self.patience = cfg['patience']
        self.model_params_path = "tmp/%s.npz" % (str(uuid.uuid4()))
        self.n_batch_trials_testing = cfg['n_batch_trials_testing']
        self.n_windows = cfg['n_windows']
        self.n_groups = cfg['n_groups']
        self.n_samples = cfg['n_samples']
        self.temperature = cfg['temperature']

        self.max_kc_id = max_kc_id
        self.n_kcs = self.max_kc_id + 1

        self._coeff = tf.Variable(tf.random.normal((self.n_groups,2*self.n_windows+1), mean=0, stddev=0.1), name="dash_coeff")
        self._kca_module = KCAssignmentModule(self.n_kcs, self.n_groups, quantized=False)
        
    def get_trainables(self):
        trainables = []
        trainables.extend([(self._coeff.name,self._coeff)])
        trainables.extend(self._kca_module.get_trainables(True))
        
        return trainables

    def train(self, train_df, valid_df):
        
        optimizer = keras.optimizers.Nadam(learning_rate=self.lr)

        self.lossf = keras.losses.BinaryCrossentropy(from_logits=True)

        # train
        n_batch_trials = self.n_batch_trials
        min_loss = float("inf")
        best_params = None
        waited = 0

        valid_loader = Loader(valid_df, self.n_kcs, self.n_windows, self.n_batch_trials_testing, shuffle=False)
        train_loader = Loader(train_df, self.n_kcs, self.n_windows, n_batch_trials, shuffle=True)

        for e in range(self.epochs):
            batch_losses = []

            train_loader.reset_state()

            while True: 
                
                features = train_loader.next_batch()
                if features == None:
                    break 
                
                with tf.GradientTape() as t:
                    ypred = self._run_model(features.F, features.curr_skill, testing=False)
                    current_loss = self.lossf(features.curr_correct, ypred)

                trainables = [t[1] for t in self.get_trainables()]
                
                grads = t.gradient(current_loss, trainables)
                optimizer.apply_gradients(zip(grads, trainables))

                batch_losses.append(current_loss.numpy())
                print("Epoch %d, Train loss = %8.4f" % (e, np.mean(batch_losses)))

            valid_loss = self.evaluate(valid_loader)
            
            if valid_loss < min_loss:
                min_loss = valid_loss
                best_params = self.get_params()
                waited = 0
            else:
                waited += 1
            
            print("Epoch %d, Train loss = %8.4f, Validation loss = %8.4f" % (e, np.mean(batch_losses), valid_loss))

            if waited >= self.patience:
                break
        
        # restore 
        self.load_params(best_params)

        return min_loss 

    def evaluate(self, loader):

        losses = []
        for i in range(self.n_samples):
            self._S = self._kca_module(1e-6)
            loss = self._evaluate(loader)
            losses.append(loss)
        
        return np.mean(losses)
    
    def _evaluate(self, valid_loader):
        valid_loader.reset_state()

        # compute validation loss
        valid_batch_losses = []
        while True:
            features = valid_loader.next_batch()
            if features == None:
                break
        
            ypred = self._run_model(features.F, features.curr_skill, testing=True)
            current_loss = self.lossf(features.curr_correct, ypred)
            valid_batch_losses.append(current_loss.numpy())

        valid_loss = np.mean(valid_batch_losses)
        
        return valid_loss

    def predict(self, df):
        n = df.shape[0]

        test_loader = Loader(df, self.n_kcs, self.n_windows, self.n_batch_trials_testing, shuffle=False)

        all_preds = np.zeros((self.n_samples, n))
        for i in range(self.n_samples):
            
            self._S = self._kca_module(1e-6)
            all_preds[i,:] = self._predict(test_loader, n)
        
        return np.mean(all_preds, axis=0)

    def _predict(self, test_loader, n):
        test_loader.reset_state()
        preds = np.zeros(n)

        while True:
            features = test_loader.next_batch()
            if features == None:
                break
        
            ypred = self._run_model(features.F, features.curr_skill)
            preds[features.trial_index] = ypred[:,0] 
        
        preds = 1/(1+np.exp(-preds))

        return preds


    def _run_model(self, F, curr_skill, testing=False):
        """
            F: [n_batch, n_kcs, 1+2W]
            curr_skill: [n_batch, n_kcs]
        """
        
        F = tf.convert_to_tensor(F, dtype=tf.float32)
        curr_skill = tf.convert_to_tensor(curr_skill, dtype=tf.float32)

        # [n_kcs, n_groups]

        # get KC assignments [n_kcs, n_groups]
        if testing:
            S = self._S
        else:
            S = self._kca_module(self.temperature)
        
        # [n_batch, n_features, n_kcs]
        Ft = tf.transpose(F, [0, 2, 1])
        
        # [n_batch, n_features, n_groups]
        Ft = tf.matmul(Ft, S)

        # [n_batch, n_groups, n_features]
        Ft = tf.transpose(Ft, [0, 2, 1])

        # [n_batch, n_groups]
        logit_p = tf.reduce_sum(Ft * self._coeff, axis=2)

        curr_group = tf.matmul(curr_skill, S)

        logit_p = tf.reduce_sum(logit_p * tf.convert_to_tensor(curr_group, dtype=tf.float32), axis=1, keepdims=True)
        
        # [n_batch,1]
        return logit_p
            

    def get_params(self):
        public_params = { k: v for k,v in self.__dict__.items() if not k.startswith('_') }
        comp_params = { t: v.numpy() for t, v in self.get_trainables() }
        p = { **public_params, **comp_params }
        return p 

    def save(self, path=None):
        p = self.get_params()
        if path is None:
            path = self.model_params_path
        np.savez(path, **p)

    def load_params(self, p):
        trainables = self.get_trainables()
        for t, v in trainables:
            v.assign(p[t])
    
    def load(self, path=None):
        print("Loading weights ...")

        if path is None:
            path = self.model_params_path

        d = np.load(path)
        self.load_params(d)

def make_sequences(df, seq_identifier):
    """
        Generates list of sequences from data frame
    """
    seqs = defaultdict(list)

    for i, row in enumerate(df.to_dict('records')):

        key = row[seq_identifier]
        row['__index__'] = i
        seqs[key].append(row)
    
    return [s for _, s in seqs.items()] 


def to_matrix_format(seq, n_kcs):
    n_trials = len(seq)

    KC_seq = np.zeros((n_trials, n_kcs))
    corr_seq = np.zeros((n_trials, 1))
    for i, r in enumerate(seq):
        KC_seq[i, r['skill']] = 1
        corr_seq[i, 0] = r['correct']
    
    index_seq = [r['__index__'] for r in seq]

    return KC_seq, corr_seq, index_seq


def make_dash_features(KC_seq, corr_seq, n_windows):
    """
        KC_seq: n_trials x n_kcs
        corr_seq: n_trials x 1
        n_windows: int
    
        Output:
            n_trials x n_kcs x (2 * n_windows)
    """
    n_trials, n_kcs = KC_seq.shape

    F_opp = np.zeros((n_trials, n_kcs, n_windows), dtype=int)
    F_corr = np.zeros_like(F_opp)
    for w in range(n_windows):
        window_size = 2 ** w 

        # cant start at first trial
        for t in range(1, n_trials):
            from_t = max(0, t - window_size)
            to_t = min(t, from_t + window_size)

            KC_seq_window = KC_seq[from_t:to_t,:]
            corr_seq_window = corr_seq[from_t:to_t,:]
            
            # n_kcs
            opp = np.sum(KC_seq_window, axis=0)
            corr = np.sum(KC_seq_window *  corr_seq_window, axis=0)
            
            F_opp[t, :, w] = opp
            F_corr[t, :, w] = corr 

    return F_opp, F_corr 


# def vectorize(seqs, n_kcs, n_windows):
    
#     n_seqs = len(seqs)

#     list_F, list_curr_skill, list_curr_correct, list_curr_index = [], [], [], []
#     for seq in seqs:
#         curr_skill, curr_correct, curr_index = to_matrix_format(seq, n_kcs)
#         F_opp, F_corr = make_dash_features(curr_skill, curr_correct, n_windows)
#         F = np.concatenate((F_opp, F_corr), axis=2)

#         list_F.append(F)
#         list_curr_skill.append(curr_skill)
#         list_curr_correct.append(curr_correct)
#         list_curr_index.extend(curr_index)

#     F = np.vstack(list_F)
#     curr_skill = np.vstack(list_curr_skill)
#     curr_correct = np.vstack(list_curr_correct)
#     curr_index = np.array(list_curr_index)

#     F = np.concatenate((np.ones((F.shape[0], F.shape[1], 1)), np.log(1+F)), axis=2)
#     #F = np.ones((F.shape[0], F.shape[1], 1))

#     return F, curr_skill, curr_correct, curr_index 

# def create_loader(F, curr_skill, curr_correct, curr_index, n_batch_trials, shuffle=True):

#     DashFeatures = namedtuple('DashFeatures', 'F curr_skill curr_correct trial_index')

#     if shuffle:
#         ix = rng.permutation(F.shape[0])
#         F = F[ix,:,:]
#         curr_skill = curr_skill[ix,:]
#         curr_correct = curr_correct[ix,:]
#         curr_index = curr_index[ix]
    

#     for start in range(0, F.shape[0], n_batch_trials):
#         endoffset = start + n_batch_trials

#         batch_F = F[start:endoffset,:,:]
#         batch_curr_skill = curr_skill[start:endoffset,:]
#         batch_curr_correct = curr_correct[start:endoffset,:]
#         batch_curr_index = curr_index[start:endoffset]

#         yield DashFeatures(batch_F, batch_curr_skill, batch_curr_correct, batch_curr_index)

DashFeatures = namedtuple('DashFeatures', 'F curr_skill curr_correct trial_index')

class Loader:

    def __init__(self, df, n_kcs, n_windows, n_batch_trials, seqs_to_buffer = 20, shuffle=True):
        self._df = df

        self._seqs = make_sequences(df, 'student')
        
        self._n_kcs = n_kcs 
        self._n_windows = n_windows
        self._n_batch_trials = n_batch_trials
        self._shuffle = shuffle 
        self._seqs_to_buffer = seqs_to_buffer 

        self.reset_state()

    def reset_state(self):
        self._next_seq_id = 0
        self._F = None
        self._curr_skill = None
        self._curr_correct = None
        self._curr_index = []
        self._finished = False 
    
    def next_batch(self):
        
        if self._next_seq_id == 0 and self._shuffle:
            rng.shuffle(self._seqs)

        if self._finished and self._F.shape[0] == 0:
            return None 
        
        while not self._finished and (self._F is None or (self._F.shape[0] < self._n_batch_trials)):
            self._add_seq()
        
        batch_F = self._F[:self._n_batch_trials, :, :]
        batch_curr_skill = self._curr_skill[:self._n_batch_trials,:]
        batch_curr_correct = self._curr_correct[:self._n_batch_trials,:]
        batch_curr_index = self._curr_index[:self._n_batch_trials]

        if batch_F.shape[0] == 0:
            return None 
        
        self._F = self._F[self._n_batch_trials:,:,:]
        self._curr_skill = self._curr_skill[self._n_batch_trials:,:]
        self._curr_correct = self._curr_correct[self._n_batch_trials:,:]
        self._curr_index = self._curr_index[self._n_batch_trials:]

        if self._shuffle:
            #print("Shuffling batch")
            ix = rng.permutation(batch_F.shape[0])
            batch_F = batch_F[ix,:,:]
            batch_curr_skill = batch_curr_skill[ix,:]
            batch_curr_correct = batch_curr_correct[ix,:]
            batch_curr_index = np.array(batch_curr_index)[ix]

        return DashFeatures(batch_F, batch_curr_skill, batch_curr_correct, batch_curr_index)
        
    def _add_seq(self):
        
        list_F = []
        list_curr_skill = []
        list_curr_correct = []
        list_curr_index = []

        for i in range(self._seqs_to_buffer):
            if self._next_seq_id == len(self._seqs):
                self._finished = True
                break
            
            seq = self._seqs[self._next_seq_id]
            self._next_seq_id += 1

            curr_skill, curr_correct, curr_index = to_matrix_format(seq, self._n_kcs)
            F_opp, F_corr = make_dash_features(curr_skill, curr_correct, self._n_windows)
            F = np.concatenate((F_opp, F_corr), axis=2)
            F = np.concatenate((np.ones((F.shape[0], F.shape[1], 1)), np.log(1+F)), axis=2)

            list_F.append(F)
            list_curr_skill.append(curr_skill)
            list_curr_correct.append(curr_correct)
            list_curr_index.extend(curr_index)
        
        if self._F is not None:
            self._F = np.vstack([self._F] + list_F)
            self._curr_skill = np.vstack([self._curr_skill] + list_curr_skill)
            self._curr_correct = np.vstack([self._curr_correct] + list_curr_correct)
            self._curr_index.extend(list_curr_index)
        else:
            self._F = np.vstack(list_F)
            self._curr_skill = np.vstack(list_curr_skill)
            self._curr_correct = np.vstack(list_curr_correct)
            self._curr_index = list_curr_index
         
    


if __name__ == "__main__":

    df = pd.read_csv("datasets/assistment.csv")
    loader = Loader(df, np.max(df['skill']) + 1, 5, 500, True)

    covered_indecies = []
    while True:
        features = loader.next_batch()
        if features == None:
            break

        covered_indecies.extend(features.trial_index)

        print(features.F.shape, " ", len(features.trial_index))
    assert df.iloc[covered_indecies].shape[0] == df.shape[0]
