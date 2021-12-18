import tensorflow as tf 
import tensorflow.keras as keras 
import module_context
import module_kc_discovery
import module_transition_probs
import module_bktcell
import sequences
import numpy as np 
from collections import defaultdict, namedtuple
TrialFeatures = namedtuple('TrialFeatures', 'correct skill item deltat included trial_index')

def transform(subseqs, n_kcs, n_items, prev_trial=False):
    n_batch = len(subseqs)
    n_trials = len(subseqs[0])
    correct = np.zeros((n_batch, n_trials), dtype=np.float32)
    skill = np.zeros((n_batch, n_trials, n_kcs), dtype=np.float32)
    item = np.zeros((n_batch, n_trials, n_items), dtype=np.float32)
    deltat = np.zeros((n_batch, n_trials), dtype=np.float32)
    included = np.zeros((n_batch, n_trials), dtype=np.float32)
    trial_index = np.zeros((n_batch, n_trials), dtype=np.int)
    tuple_idx = 0 if prev_trial else 1 
    for s, seq in enumerate(subseqs):
        for t, elm in enumerate(seq):
            
            trial = elm[tuple_idx]
            if trial is None:
                correct[s, t] = 0
                deltat[s, t] = 0
                included[s, t] = False 
                trial_index[s, t] = -1
            else:
                correct[s, t] = trial['correct']
                skill[s, t, trial['skill']] = 1
                item[s, t, trial['problem']] = 1
                deltat[s, t] = trial['__deltat__']
                included[s, t] = True 
                trial_index[s, t] = trial['__index__']
            
    return TrialFeatures(
        tf.convert_to_tensor(correct), 
        tf.convert_to_tensor(skill), 
        tf.convert_to_tensor(item), 
        tf.convert_to_tensor(deltat), 
        tf.convert_to_tensor(included), 
        trial_index) 

class ModelSuperBkt:

    def __init__(self, cfg):
        self.cfg = cfg 

        self._context_module = module_context.ModuleContext(**cfg['context_module_args'])
        self._kc_module = module_kc_discovery.ModuleKCDiscovery(**cfg['kc_module_args'])
        self._learning_prob_module = module_transition_probs.ModuleTransitionProbs(name="learning_prob", **cfg['transition_probs_module_args'])
        self._not_forgetting_prob_module = module_transition_probs.ModuleTransitionProbs(name="not_forgetting_prob", **cfg['transition_probs_module_args'])
        self._bkt_module = module_bktcell.ModuleBktCell()

        self._modules = [self._context_module, self._kc_module, self._learning_prob_module, self._not_forgetting_prob_module, self._bkt_module]

    
    def train(self, df, train_students, valid_students):
        
        max_kc_id = np.max(df['skill'])
        max_item_id = np.max(df['problem'])

        n_kcs = max_kc_id + 1
        n_items = max_item_id + 1


        for module in self._modules:
            print("Initializing %s" % module)
            module.on_train(df, train_students, valid_students)

        optimizer = keras.optimizers.Nadam(learning_rate=self.cfg['learning_rate'])

        # extract sequences
        # split long sequences
        # iterate unbatched
        train_seqs = sequences.make_sequences(df, train_students)
        sequences.calculate_deltat(train_seqs)
        print("Total sequences: %d" % len(train_seqs))
        print("Sequences with more than %d trials: %d" % (self.cfg['max_seq_len'], np.sum([1 for s in train_seqs if len(s) > self.cfg['max_seq_len']])))
        train_seqs = sequences.split_long_seqs(train_seqs, self.cfg['max_seq_len'])

        valid_seqs = sequences.make_sequences(df, valid_students)

        # train
        min_loss = float("inf")
        waited = 0

        for e in range(self.cfg['n_epochs']):
            batch_losses = []
            for batch_seqs, newseqs in sequences.iterate_unbatched(train_seqs, self.cfg['n_batch_seqs']):
                curr_trial_features = transform(batch_seqs, n_kcs, n_items, False)
                prev_trial_features = transform(batch_seqs, n_kcs, n_items, True)

                print("%d: %d, %d" % (e, curr_trial_features.correct.shape[0], curr_trial_features.correct.shape[1]))
                with tf.GradientTape() as t:
                    ypred = self._run(prev_trial_features, curr_trial_features, newseqs)
                    #current_loss = utils.xe_loss(features.curr_corr, ypred, features.curr_mask)

        #         trainables = self.get_trainables(new_seqs)
                
        #         grads = t.gradient(current_loss, trainables)
        #         optimizer.apply_gradients(zip(grads, trainables))
        #         self._post_update()

        #         #print("Loss: %f" % current_loss.numpy())
        #         batch_losses.append(current_loss.numpy())
            
        #     valid_loss = self.evaluate(valid_seqs)
            
        #     # if np.isnan(valid_loss):
        #     #     break
        #     if valid_loss < min_loss:
        #         min_loss = valid_loss
        #         #self.save()
        #         best_params = self.get_params()
        #         waited = 0
        #     else:
        #         waited += 1
            
        #     print("Epoch %d, First Trial = %d, Train loss = %8.4f, Validation loss = %8.4f" % (e, new_seqs, np.mean(batch_losses), valid_loss))

        #     if waited >= self.patience:
        #         break
        
        # # restore 
        # self.load_params(best_params)

        # return min_loss 

    def _run(self, prev_trial_features, curr_trial_features, newseqs):

        # get KC assignments
        S = self._kc_module()
        prev_trial_features_skill = tf.matmul(prev_trial_features.skill, S)
        curr_trial_features_skill = tf.matmul(curr_trial_features.skill, S)

        # get probs of answering correctly
        pC = self._context_module.compute_context(curr_trial_features.item)

        # get transition probabilities
        pL = self._learning_prob_module(prev_trial_features_skill, curr_trial_features.deltat, newseqs)
        pF = 1 - self._not_forgetting_prob_module(prev_trial_features_skill, curr_trial_features.deltat, newseqs)

        # concat probabilities
        probs_curr = tf.concat((pL, pF, pC), axis=2)

        preds = self._bkt_module(prev_trial_features_skill, prev_trial_features.correct, curr_trial_features_skill, probs_curr, newseqs)
        
        return preds 
        

def reduce_items(df, top):
    """
        For datasets with very large number of items, this reduces the unique
        items into ones that are practiced frequently. Items that are less frequent 
        are remapped into their corresponding KCs.
    """
    cnts = df['problem'].value_counts()
    print(cnts)
    print("# items: %d" % len(set(df['problem'])))
    min_trials = cnts.iloc[top-1]
    
    print("Items with more than %d trials: %d" % (min_trials, np.sum(cnts >= min_trials)))
    print("Items with less than %d trials: %d" % (min_trials, np.sum(cnts < min_trials)))
    ineligible_items = set(cnts[cnts < min_trials].index) 
    ineligible_ix = df['problem'].isin(ineligible_items)   
    print("ineligible: %d" % len(ineligible_items))
    kcs = np.array(df['skill'])
    items = np.array(df['problem'])
    items[ineligible_ix] = -kcs[ineligible_ix]
    df['problem'] = items 
    unique_items = set(items)
    print("Unique items: %d" % len(unique_items))
    
    remapped = dict(zip(unique_items, range(len(unique_items))))
    df['problem'] = [remapped[p] for p in df['problem']]
    print("New number of items: %d" % len(set(df['problem'])))
if __name__ == "__main__":
    import pandas as pd 
    import numpy as np 
    cfg = {
        "context_module_args" : {},
        "transition_probs_module_args" : {
            "scale" : 60 * 60 * 24,
            "x0" : 0.,
            "x1" : 3
        },
        "kc_module_args" : {
            "temperature" : 0.5,
            "n_groups" : 10
        },
        "n_batch_trials" : 50,
        "learning_rate" : 0.001,
        "n_epochs" : 100,
        "n_batch_seqs" : 50,
        "max_seq_len" : 100
    }
    df = pd.read_csv("data/datasets/gervetetal_assistments09.csv")
    reduce_items(df, 1000)
    splits = np.load("data/splits/gervetetal_assistments09.npy")
    split = splits[0, :]
    train_ix = split == 2
    valid_ix = split == 1
    test_ix = split == 0
    train_df = df[train_ix]
    valid_df = df[valid_ix]
    test_df = df[test_ix]
    train_students = set(train_df['student'])
    valid_students = set(valid_df['student'])
    model = ModelSuperBkt(cfg)
    model.train(df, train_students, valid_students)