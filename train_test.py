import numpy as np 
import pandas as pd
import sklearn.metrics 
import model_bkt
import model_dkt 
import model_ldkt 
import model_bkt_skill_discovery
import model_ldkt_skill_discovery
import model_dash_skill_discovery
import model_dash_alt
import utils 
from multiprocessing import Pool, TimeoutError, Manager
import copy 
import model_bkt_items 

MAPPING = {
    "bkt" : model_bkt.create_model,
    "dkt" : model_dkt.create_model,
    "ldkt" : model_ldkt.create_model,
    "bkt-sd" : model_bkt_skill_discovery.create_model,
    "ldkt-sd" : model_ldkt_skill_discovery.create_model,
    "dash" : model_dash_alt.create_model,
    "dash-sd" : model_dash_skill_discovery.create_model,
    "bkt-items" : model_bkt_items.create_model
}

def train_test(cfg, df, split, train_id, valid_id, test_id, model_params_path=None):

    train_ix = split == train_id
    valid_ix = split == valid_id
    test_ix = split == test_id

    train_df = df[train_ix]
    valid_df = df[valid_ix]
    test_df = df[test_ix]


    test_only_skills = set(np.unique(test_df['skill'])) - set(np.unique(train_df['skill'])) 
    
    remove_ix = test_df['skill'].isin(test_only_skills)
    print("Removing %d trials from test due to lack of skill in training" % np.sum(remove_ix))
    test_df = test_df[~remove_ix]

    print("Training: %d, Validation: %d, Test: %d" % (train_df.shape[0], valid_df.shape[0], test_df.shape[0]))

    # train model        
    model = MAPPING[cfg['model']](cfg, df)
    if model_params_path is None:
        model.train(train_df, valid_df)
        model.save()
    else:
        model.load(model_params_path)
    
    # sweep decision thresholds on validation set and pick one that achieves highest BACC
    thresholds = np.linspace(0, 1, 50)
    baccs = np.zeros_like(thresholds)
    ytrue_valid = np.array(valid_df['correct'])
    preds_valid = model.predict(valid_df)

    for i in range(len(thresholds)):
        t = thresholds[i]
        hard_preds = preds_valid >= t 
        baccs[i] = sklearn.metrics.balanced_accuracy_score(ytrue_valid, hard_preds)
    
    ix = np.argmax(baccs)
    best_threshold = thresholds[ix]
    print("Decision threshold: %0.2f" % best_threshold)

    preds = model.predict(test_df)
    actual = np.array(test_df['correct'])

    xe = -(actual * np.log(preds) + (1-actual) * np.log(1-preds))
    print("Test XE: %0.2f" % np.mean(xe))

    auc = sklearn.metrics.roc_auc_score(actual, preds)
    print("Test AUC-ROC: %0.2f" % auc)

    hard_preds = preds >= best_threshold
    cm = sklearn.metrics.confusion_matrix(actual, hard_preds)
    bacc = sklearn.metrics.balanced_accuracy_score(actual, hard_preds)
    print("Test BACC: %0.2f" % bacc)
    print(cm)

    return {
        "xe" : np.mean(xe),
        "auc-roc" : auc,
        "threshold" : best_threshold,
        "bacc" : bacc,
        "cm" : cm.tolist(),
        "model_params_path" : model.model_params_path
    }

if __name__ == "__main__":

    import sys 
    import json 

    cfg_path = sys.argv[1]
    df_path = sys.argv[2]
    split_path = sys.argv[3]
    split_id = int(sys.argv[4])
    output_path = sys.argv[5]
    train_id, valid_id, test_id = 2, 1, 0
    if len(sys.argv) > 6:
        train_id, valid_id, test_id = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
   
    model_params_path = None
    if len(sys.argv) > 9:
        model_params_path = sys.argv[9]

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    df = pd.read_csv(df_path)
    cnts = df['problem'].value_counts()
    print(cnts)
    print("# items: %d" % len(set(df['problem'])))
    #q = np.percentile(cnts, 50)
    q = 50
    print(q)
    print("Items with more than %d trials: %d" % (q, np.sum(cnts >= q)))
    print("Items with less than %d trials: %d" % (q, np.sum(cnts < q)))

    ineligible_items = set(cnts[cnts < q].index) 

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
    splits = np.load(split_path)

    split = splits[split_id, :]

    r = train_test(cfg, df, split, train_id, valid_id, test_id, model_params_path)

    print(r)

    r['cfg'] = cfg
    with open(output_path, 'w') as f:
        json.dump(r, f, indent=4, cls=utils.NumpyEncoder)
    