#
# Standard BKT Model
#
import tensorflow as tf
import sequence_funcs as sf 
import utils 
import tensorflow.keras as keras 
import numpy as np

class StudentModel:

    def __init__(self, cfg):
        self.n_batch_seqs = cfg['n_batch_seqs']
        self.n_batch_trials = cfg['n_batch_trials']
        self.epochs = cfg['epochs']
        self.lr = cfg['lr']
        self.patience = cfg['patience']
        self.model_params_path = cfg['model_params_path']

        self._components = []
        self._init_model_components(self._components)

#
# These functions are what needs to be customized on a per-model basis
#
    def _init_model_components(self, components):
        """
            Initializes the components of the model and stores them in the _components attribute
        """
        pass


    def _make_seqs(self, df):
        """
            Creates and featurizes the sequences from the dataframe
        """
        pass

    def _run_model(self, features, new_seqs):
        """
            Executes the model
        """
        pass

    def _create_feature_transformer(self):
        """
            Transforms features into numeric arrays
        """
        pass
#
# End of functions that need to be customized
#

    def _iterate(self, seqs, shuffle=False):
        """
            Iterates over sequences in batches
        """
        return sf.create_loader(seqs, self.n_batch_seqs, self.n_batch_trials, self._create_feature_transformer() , shuffle=shuffle)


    def get_trainables(self, new_seqs):
        trainables = []
        for comp in self._components:
            trainables.extend(comp.get_trainables(new_seqs))
        trainables = [t[1] for t in trainables]
        return trainables

    def save(self):
        public_params = { k: v for k,v in self.__dict__.items() if not k.startswith('_') }
        comp_params = utils.save_params(self._components)
        np.savez(self.model_params_path, **public_params, **comp_params)

    def load(self):
        print("Loading weights ...")
        d = np.load(self.model_params_path)
        utils.load_params(self._components, d)

    def train(self, train_df, valid_df):
        
        # prepare data for training and validation
        train_seqs = self._make_seqs(train_df)
        valid_seqs = self._make_seqs(valid_df)

        optimizer = keras.optimizers.Nadam(learning_rate=self.lr)

        # train
        min_loss = float("inf")
        waited = 0
        for e in range(self.epochs):
            batch_losses = []
            for features, new_seqs in self._iterate(train_seqs, shuffle=True):

                with tf.GradientTape() as t:
                    ypred = self._run_model(features, new_seqs)
                    current_loss = utils.xe_loss(features.curr_corr, ypred, features.curr_mask)

                trainables = self.get_trainables(new_seqs)

                grads = t.gradient(current_loss, trainables)
                optimizer.apply_gradients(zip(grads, trainables))

                batch_losses.append(current_loss.numpy())
            
            # compute validation loss
            valid_batch_losses = []
            for features, new_seqs in self._iterate(valid_seqs, shuffle=False):
                ypred = self._run_model(features, new_seqs)
                current_loss = utils.xe_loss(features.curr_corr, ypred, features.curr_mask)

                valid_batch_losses.append(current_loss.numpy())
            valid_loss = np.mean(valid_batch_losses)

            if valid_loss < min_loss:
                min_loss = valid_loss
                self.save()
                waited = 0
            else:
                waited += 1
            
            print("Epoch %d, First Trial = %d, Running loss = %8.4f, Validation loss = %8.2f" % (e, new_seqs, np.mean(batch_losses), np.mean(valid_batch_losses)))

            if waited >= self.patience:
                break
        
        # restore 
        self.load()

    def predict(self, df):

        seqs = self._make_seqs(df)
        preds = np.zeros(df.shape[0])
        for features, new_seqs in self._iterate(seqs, shuffle=False):
            ypred = self._run_model(features, new_seqs)
            for i in range(ypred.shape[0]):
                for j in range(ypred.shape[1]):
                    trial_index = features.trial_index[i,j]
                    if trial_index > -1:
                        preds[trial_index] = ypred[i,j]

        return preds 
