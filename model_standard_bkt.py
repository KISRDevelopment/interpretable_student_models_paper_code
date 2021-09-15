#
# Standard BKT Model
#
import cell_bkt
import tensorflow as tf
import sequence_funcs as sf 
import utils 
import tensorflow.keras as keras 
import numpy as np
class StandardBkt:

    def __init__(self, 
                 max_kc_id, 
                 n_batch_seqs, 
                 n_batch_trials, 
                 epochs, 
                 lr,
                 patience,
                 model_params_path):
        self.n_batch_seqs = n_batch_seqs
        self.n_batch_trials = n_batch_trials
        self.max_kc_id = max_kc_id
        self.epochs = epochs
        self.lr = lr 
        self.patience = patience
        self.model_params_path = model_params_path

        self.placeholder_kc_id = self.max_kc_id + 1
        self.n_kcs = self.placeholder_kc_id + 1

    def train(self, train_df, valid_df):
        

        # prepare data for training and validation
        train_seqs = self._make_seqs(train_df)
        valid_seqs = self._make_seqs(valid_df)

        # initialize model components
        probs_module = StandardBktProbs(self.n_kcs)
        rnn_module = cell_bkt.BktCell(self.n_kcs)
        optimizer = keras.optimizers.Nadam(learning_rate=self.lr)

        # train
        min_loss = float("inf")
        waited = 0
        for e in range(self.epochs):
            batch_losses = []
            for features, new_seqs in sf.create_loader(train_seqs, self.n_batch_seqs, self.n_batch_trials, sf.create_kt_transformer(self.n_kcs)):

                with tf.GradientTape() as t:
                    
                    # acquire BKT's transition and emission parameters
                    logit_probs_prev = probs_module(features.prev_skill)
                    logit_probs_curr = probs_module(features.curr_skill)

                    # run BKT
                    ypred = rnn_module(features.prev_skill, features.prev_corr, features.curr_skill, new_seqs, logit_probs_prev, logit_probs_curr)

                    # calculate loss
                    current_loss = utils.xe_loss(features.curr_corr, ypred, features.curr_mask)


                trainables = [t[1] for t in rnn_module.trainables+probs_module.trainables]
                grads = t.gradient(current_loss, trainables)
                optimizer.apply_gradients(zip(grads, trainables))

                batch_losses.append(current_loss.numpy())
            
            # compute validation loss
            valid_batch_losses = []
            for features, new_seqs in sf.create_loader(valid_seqs, self.n_batch_seqs, self.n_batch_trials, sf.create_kt_transformer(self.n_kcs)):

                # acquire BKT's transition and emission parameters
                logit_probs_prev = probs_module(features.prev_skill)
                logit_probs_curr = probs_module(features.curr_skill)

                # run BKT
                ypred = rnn_module(features.prev_skill, features.prev_corr, features.curr_skill, new_seqs, logit_probs_prev, logit_probs_curr)

                # calculate loss
                current_loss = utils.xe_loss(features.curr_corr, ypred, features.curr_mask)

                valid_batch_losses.append(current_loss.numpy())
            valid_loss = np.mean(valid_batch_losses)

            if valid_loss < min_loss:
                min_loss = valid_loss
                params = utils.save_params([rnn_module, probs_module])
                np.savez(self.model_params_path, **params)
                waited = 0
            else:
                waited += 1
            
            print("Epoch %d, First Trial = %d, Running loss = %8.4f, Validation loss = %8.2f" % (e, new_seqs, np.mean(batch_losses), np.mean(valid_batch_losses)))

            if waited >= self.patience:
                break
    
    def _make_seqs(self, df):

        seqs = sf.make_sequences(df, 'student')
        seqs = sf.pad_seqs(seqs, self.n_batch_trials)
        seqs = sf.featurize_seqs(seqs, {
            "correct" : 0,
            "skill" : self.placeholder_kc_id
        })

        return seqs 


class StandardBktProbs(object):

    def __init__(self, n_kcs):
        self.n_kcs = n_kcs

        # [n_skills, 4]
        self.logit_probs = tf.Variable(tf.random.normal((self.n_kcs,4), mean=0, stddev=0.1), name="bktprobs")

        self.trainables = [
            ('logit_probs', self.logit_probs)
        ]
        
    def __call__(self, skill):
        """ 
            skill: [n_batch, n_steps, n_skills]
            Returns:
                BKT Probabilities per skill (pL, pF, pC0, pC1) [n_batch, n_steps, 4]
        """
        return tf.matmul(skill, self.logit_probs)
        

if __name__ == "__main__":

    import sys 
    import pandas as pd 

    df = pd.read_csv(sys.argv[1])

    split = np.load(sys.argv[2])
    
    max_kc_id = np.max(df['skill'])

    train_ix = split[0, :] == 2
    valid_ix = split[0, :] == 1
    
    train_df = df[train_ix]
    valid_df = df[valid_ix]

    print("Training: %d, Validation: %d" % (train_df.shape[0], valid_df.shape[0]))
    bkt = StandardBkt(max_kc_id, 500, 50, 50, 0.01, 5, "tmp/bktparams")
    bkt.train(train_df, valid_df)
