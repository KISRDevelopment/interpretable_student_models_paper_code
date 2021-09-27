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
        self.temperature = cfg['temperature']
        self.n_samples = cfg['n_samples']

        self.max_kc_id = max_kc_id
        self.n_kcs = self.max_kc_id + 1

        self._model = keras.layers.Dense(1, activation='linear')
        self._kca_module = KCAssignmentModule(self.n_kcs, self.n_groups)

    def get_trainables(self):
        trainables = []
        trainables.extend([(t.name,t) for t in self._model.trainable_variables])
        trainables.extend(self._kca_module.trainables)

        return trainables

    def train(self, train_df, valid_df):
        
        # prepare data for training and validation
        train_features = self._make_seqs(train_df)
        valid_features = self._make_seqs(valid_df)

        optimizer = keras.optimizers.Nadam(learning_rate=self.lr)

        self.lossf = keras.losses.BinaryCrossentropy(from_logits=True)

        # train
        n_batch_trials = int(train_df.shape[0] * 0.1)
        print("Batch trials: %d" % n_batch_trials)
        min_loss = float("inf")
        best_params = None
        waited = 0
        for e in range(self.epochs):
            batch_losses = []
            for features in create_loader(*train_features, 
                n_batch_trials=n_batch_trials, 
                shuffle=True):
        
                with tf.GradientTape() as t:
                    ypred = self._run_model(features.F, features.curr_skill, testing=False)
                    current_loss = self.lossf(features.curr_correct, ypred)

                trainables = [t[1] for t in self.get_trainables()]
                
                grads = t.gradient(current_loss, trainables)
                optimizer.apply_gradients(zip(grads, trainables))

                batch_losses.append(current_loss.numpy())
                print("Epoch %d, Train loss = %8.4f" % (e, np.mean(batch_losses)))

            valid_loss = self.evaluate(valid_features)
            
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

    def evaluate(self, valid_features):

        # compute validation loss
        sample_losses = []
        for i in range(self.n_samples):
            valid_batch_losses = []
            self._S = self._kca_module(self.temperature)
            for features in create_loader(*valid_features, 
                    n_batch_trials=self.n_batch_trials_testing, 
                    shuffle=False):
            
                ypred = self._run_model(features.F, features.curr_skill, testing=True)
                current_loss = self.lossf(features.curr_correct, ypred)

                valid_batch_losses.append(current_loss.numpy())

            valid_loss = np.mean(valid_batch_losses)
            sample_losses.append(valid_loss)
        
        return np.mean(sample_losses) 

    def predict(self, df):
        n = df.shape[0]

        preds = np.zeros(n)

        test_features = self._make_seqs(df)


        preds = np.zeros((n, self.n_samples))
        for i in range(self.n_samples):
            self._S = self._kca_module(self.temperature)
            for features in create_loader(*test_features, 
                    n_batch_trials=self.n_batch_trials_testing, 
                    shuffle=False):

                ypred = self._run_model(features.F, features.curr_skill, testing=True)
                preds[features.trial_index, i] = ypred[:,0] 
        
        preds = 1/(1+np.exp(-preds))

        
        return np.mean(preds, axis=1) 

    def _make_seqs(self, df):
        seqs = make_sequences(df, 'student')
        features = vectorize(seqs, self.n_kcs, self.n_windows)
        return features

    def _run_model(self, F, curr_skill, testing=False):
        """
            F: [n_batch, n_kcs, 1+2W]
            curr_skill: [n_batch, n_kcs]
        """
        
        F = tf.convert_to_tensor(F, dtype=tf.float32)
        curr_skill = tf.convert_to_tensor(curr_skill, dtype=tf.float32)

        # [n_kcs, n_groups]
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
        logit_p = tf.squeeze(self._model(tf.convert_to_tensor(Ft, dtype=tf.float32)))

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


def vectorize(seqs, n_kcs, n_windows):
    
    n_seqs = len(seqs)

    list_F, list_curr_skill, list_curr_correct, list_curr_index = [], [], [], []
    for seq in seqs:
        curr_skill, curr_correct, curr_index = to_matrix_format(seq, n_kcs)
        F_opp, F_corr = make_dash_features(curr_skill, curr_correct, n_windows)
        F = np.concatenate((F_opp, F_corr), axis=2)

        list_F.append(F)
        list_curr_skill.append(curr_skill)
        list_curr_correct.append(curr_correct)
        list_curr_index.extend(curr_index)

    F = np.vstack(list_F)
    curr_skill = np.vstack(list_curr_skill)
    curr_correct = np.vstack(list_curr_correct)
    curr_index = np.array(list_curr_index)

    F = np.concatenate((np.ones((F.shape[0], F.shape[1], 1)), np.log(1+F)), axis=2)
    #F = np.ones((F.shape[0], F.shape[1], 1))

    return F, curr_skill, curr_correct, curr_index 

def create_loader(F, curr_skill, curr_correct, curr_index, n_batch_trials, shuffle=True):

    DashFeatures = namedtuple('DashFeatures', 'F curr_skill curr_correct trial_index')

    if shuffle:
        ix = rng.permutation(F.shape[0])
        F = F[ix,:,:]
        curr_skill = curr_skill[ix,:]
        curr_correct = curr_correct[ix,:]
        curr_index = curr_index[ix]
    

    for start in range(0, F.shape[0], n_batch_trials):
        endoffset = start + n_batch_trials

        batch_F = F[start:endoffset,:,:]
        batch_curr_skill = curr_skill[start:endoffset,:]
        batch_curr_correct = curr_correct[start:endoffset,:]
        batch_curr_index = curr_index[start:endoffset]

        yield DashFeatures(batch_F, batch_curr_skill, batch_curr_correct, batch_curr_index)


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

        # quantize it
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

    rows = [
        { "student" : 0, "correct" : 1, "skill" : 0 },
        { "student" : 0, "correct" : 0, "skill" : 1 },
        { "student" : 1, "correct" : 0, "skill" : 1 },
        { "student" : 0, "correct" : 1, "skill" : 2 },
        { "student" : 1, "correct" : 0, "skill" : 0 },
        { "student" : 1, "correct" : 1, "skill" : 2 },
        { "student" : 1, "correct" : 1, "skill" : 2 },
        { "student" : 1, "correct" : 1, "skill" : 1 },
    ]
    df = pd.DataFrame(rows)

    seqs = make_sequences(df, 'student')

    #KC_seq, corr_seq, index_seq = to_matrix_format(seqs[0], 3)
    # print(KC_seq)
    # print(corr_seq)
    # print(index_seq)
    
    # n_windows = 2

    # F_opp, F_corr = make_dash_features(KC_seq, corr_seq, n_windows)
    # print("F_opp:")
    # print(F_opp)

    # print("F_corr:")
    # print(F_corr)

    # for features in create_loader(seqs, 3, 3, 5, shuffle=False):
    #     print(features.F.shape)

    df = pd.read_csv("./datasets/synthetic.csv")

    splits = np.load("./splits/synthetic.npy")
    split_id = 0

    split = splits[split_id, :]

    train_ix = split == 2
    valid_ix = split == 1
    test_ix = split == 0

    train_df = df[train_ix]
    valid_df = df[valid_ix]
    test_df = df[test_ix]

    cfg = {
        "model" : "dash",
        "n_batch_trials" : 500,
        "n_batch_trials_testing" : 5000,
        "epochs" : 10,
        "lr" : 0.1,
        "n_windows" : 5,
        "patience" : 20,
        "n_groups" : 5,
        "temperature" : 0.5
    }

    model = DashModel(np.max(df['skill']), cfg)

    model.train(train_df, valid_df)

    preds = model.predict(valid_df)
    print(preds.shape)
    print(preds)
    