import numpy as np 
import pandas as pd
import sklearn.metrics 
import model_bkt
import model_dkt 
import model_ldkt 
import model_bkt_skill_discovery
import model_ldkt_skill_discovery
import utils 

MAPPING = {
    "bkt" : model_bkt.create_model,
    "dkt" : model_dkt.create_model,
    "ldkt" : model_ldkt.create_model,
    "bkt-sd" : model_bkt_skill_discovery.create_model,
    "ldkt-sd" : model_ldkt_skill_discovery.create_model
}

def train_test(cfg, df, split):

    train_ix = split == 2
    valid_ix = split == 1
    test_ix = split == 0

    train_df = df[train_ix]
    valid_df = df[valid_ix]
    test_df = df[test_ix]

    print("Training: %d, Validation: %d, Test: %d" % (train_df.shape[0], valid_df.shape[0], test_df.shape[0]))

    # perform hyper parameter optimization
    min_loss = float("inf")
    best_model = None
    best_combination = None
    for comb in utils.hyperparam_combs(cfg['hyperparams']):
        cfg.update(comb)

        print(cfg)
        model = MAPPING[cfg['model']](cfg, df)
        loss = model.train(train_df, valid_df)

        if loss < min_loss:
            min_loss = loss
            best_model = model
            best_combination = comb
    
    # sweep decision thresholds on validation set and pick one that achieves highest BACC
    thresholds = np.linspace(0, 1, 50)
    baccs = np.zeros_like(thresholds)
    ytrue_valid = np.array(valid_df['correct'])
    preds_valid = best_model.predict(valid_df)

    for i in range(len(thresholds)):
        t = thresholds[i]
        hard_preds = preds_valid >= t 
        baccs[i] = sklearn.metrics.balanced_accuracy_score(ytrue_valid, hard_preds)
    
    ix = np.argmax(baccs)
    best_threshold = thresholds[ix]
    print("Decision threshold: %0.2f" % best_threshold)

    preds = best_model.predict(test_df)
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
        "best_combination" : best_combination
    }

if __name__ == "__main__":

    import sys 
    import json 

    cfg_path = sys.argv[1]
    df_path = sys.argv[2]
    split_path = sys.argv[3]
    split_id = int(sys.argv[4])

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    df = pd.read_csv(df_path)

    splits = np.load(split_path)

    split = splits[split_id, :]

    r = train_test(cfg, df, split)

    print(r)
    