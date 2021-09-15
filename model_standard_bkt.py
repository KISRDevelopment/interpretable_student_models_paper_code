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

    def __init__(self, max_kc_id, n_batch_seqs, n_batch_trials, epochs, lr):
        self.n_batch_seqs = n_batch_seqs
        self.n_batch_trials = n_batch_trials
        self.max_kc_id = max_kc_id
        self.epochs = epochs
        self.lr = lr 

    def train(self, train_df):
        
        placeholder_kc_id = self.max_kc_id + 1
        n_kcs = placeholder_kc_id + 1

        # prepare data for training
        seqs = sf.make_sequences(train_df, 'student')
        seqs = sf.pad_seqs(seqs, self.n_batch_trials)
        seqs = sf.featurize_seqs(seqs, {
            "correct" : 0,
            "skill" : placeholder_kc_id
        })

        # initialize model components
        probs_module = cell_bkt.StandardBktProbs(n_kcs)
        rnn_module = cell_bkt.BktCell(n_kcs)

        # optimization algorithm
        optimizer = keras.optimizers.Nadam(learning_rate=self.lr)

        # train
        for e in range(self.epochs):
            batch_losses = []
            for features, new_seqs in sf.create_loader(seqs, self.n_batch_seqs, self.n_batch_trials, sf.create_kt_transformer(placeholder_kc_id)):

                with tf.GradientTape() as t:
                    
                    # acquire BKT's transition and emission parameters
                    logit_probs_prev = probs_module(features.prev_skill)
                    logit_probs_curr = probs_module(features.curr_skill)

                    # run BKT
                    ypred = rnn_module(features.prev_skill, features.prev_corr, features.curr_skill, new_seqs, logit_probs_prev, logit_probs_curr)

                    # calculate loss
                    current_loss = utils.xe_loss(features.curr_corr, ypred, features.curr_mask)


                trainables = probs_module.trainables + rnn_module.trainables
                grads = t.gradient(current_loss, trainables)
                optimizer.apply_gradients(zip(grads, trainables))

                batch_losses.append(current_loss.numpy())
            
            print("Epoch %d, First Trial = %d, Running loss = %8.4f" % (e, new_seqs, np.mean(batch_losses)))
        
if __name__ == "__main__":

    import sys 
    import pandas as pd 

    df = pd.read_csv(sys.argv[1])

    bkt = StandardBkt(np.max(df['skill']), 500, 50, 10, 0.01)
    bkt.train(df)
