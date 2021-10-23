#
# Standard BKT Model
#
import tensorflow as tf
import sequence_funcs as sf 
import utils 
import tensorflow.keras as keras 
import numpy as np
import uuid 

class StudentModel:

    def __init__(self, cfg):
        self.p_n_batch_seqs = cfg['p_n_batch_seqs']
        self.n_batch_seqs = cfg['n_batch_seqs']
        self.n_batch_trials = cfg['n_batch_trials']
        self.epochs = cfg['epochs']
        self.lr = cfg['lr']
        self.patience = cfg['patience']
        self.model_params_path = "tmp/%s.npz" % (str(uuid.uuid4()))
        self.n_batch_seqs_testing = cfg['n_batch_seqs_testing']
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
    
    def _post_update(self):
        """
            Gets called after gradient update
        """
        pass
    
    def _on_before_training(self, train_df, valid_df):
        """
            Gets called just before training starts
        """
        pass
#
# End of functions that need to be customized
#

    def _iterate(self, seqs, shuffle=False, testing=False):
        """
            Iterates over sequences in batches
        """
        n_batch_seqs = self.n_batch_seqs
        if testing:
            n_batch_seqs = self.n_batch_seqs_testing
        
        if self.p_n_batch_seqs:
            n_batch_seqs = int(n_batch_seqs * len(seqs))
        
            #print("Effective batchsize: (%d,%d)" % (n_batch_seqs, self.n_batch_trials))
        return sf.create_loader(seqs, n_batch_seqs, self.n_batch_trials, self._create_feature_transformer() , shuffle=shuffle)


    def get_trainables(self, new_seqs):
        trainables = []
        for comp in self._components:
            trainables.extend(comp.get_trainables(new_seqs))

        trainables = [t[1] for t in trainables]
        return trainables

    def get_params(self):
        public_params = { k: v for k,v in self.__dict__.items() if not k.startswith('_') }
        comp_params = utils.save_params(self._components)
        p = { **public_params, **comp_params }
        return p 

    def save(self, path=None):
        p = self.get_params()
        if path is None:
            path = self.model_params_path
        np.savez(path, **p)

    def load_params(self, p):
        utils.load_params(self._components, p)
    
    def load(self, path=None):
        print("Loading weights ...")

        if path is None:
            path = self.model_params_path

        d = np.load(path)
        self.load_params(d)

    def train(self, train_df, valid_df):
        
        self._on_before_training(train_df, valid_df)
        
        # prepare data for training and validation
        train_seqs = self._make_seqs(train_df)
        valid_seqs = self._make_seqs(valid_df)

        optimizer = keras.optimizers.Nadam(learning_rate=self.lr)

        # train
        min_loss = float("inf")
        best_params = None
        waited = 0

        for e in range(self.epochs):
            batch_losses = []
            for features, new_seqs in self._iterate(train_seqs, shuffle=True):
                
                #print("Batch %d" % len(batch_losses))
                with tf.GradientTape() as t:
                    ypred = self._run_model(features, new_seqs, testing=False)
                    current_loss = utils.xe_loss(features.curr_corr, ypred, features.curr_mask)

                trainables = self.get_trainables(new_seqs)
                
                grads = t.gradient(current_loss, trainables)
                optimizer.apply_gradients(zip(grads, trainables))
                self._post_update()

                #print("Loss: %f" % current_loss.numpy())
                batch_losses.append(current_loss.numpy())
            
            valid_loss = self.evaluate(valid_seqs)
            
            # if np.isnan(valid_loss):
            #     break
            if valid_loss < min_loss:
                min_loss = valid_loss
                #self.save()
                best_params = self.get_params()
                waited = 0
            else:
                waited += 1
            
            print("Epoch %d, First Trial = %d, Train loss = %8.4f, Validation loss = %8.4f" % (e, new_seqs, np.mean(batch_losses), valid_loss))

            if waited >= self.patience:
                break
        
        # restore 
        self.load_params(best_params)

        return min_loss 

    def evaluate(self, seqs):

        # compute validation loss
        valid_batch_losses = []
        for features, new_seqs in self._iterate(seqs, shuffle=False, testing=True):
            ypred = self._run_model(features, new_seqs, testing=True)
            current_loss = utils.xe_loss(features.curr_corr, ypred, features.curr_mask)

            valid_batch_losses.append(current_loss.numpy())
        valid_loss = np.mean(valid_batch_losses)

        return valid_loss 

    def predict(self, df):

        seqs = self._make_seqs(df)
        return self.predict_seqs(df.shape[0], seqs)

    def predict_seqs(self, n, seqs):
        preds = np.zeros(n)
        for features, new_seqs in self._iterate(seqs, shuffle=False, testing=True):
            ypred = self._run_model(features, new_seqs, testing=True)
            flat_preds = ypred.numpy().flatten()
            flat_indecies = features.trial_index.flatten()

            ix = flat_indecies > -1
            flat_preds = flat_preds[ix]
            flat_indecies = flat_indecies[ix]

            preds[flat_indecies] = flat_preds

        return preds 
