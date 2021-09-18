import numpy as np 
import pandas as pd
import sklearn.metrics 
import model_bkt
import model_dkt 
import model_ldkt 

MAPPING = {
    "bkt" : model_bkt.create_model,
    "dkt" : model_dkt.create_model,
    "ldkt" : model_ldkt.create_model
}

def train_test(cfg, df, split):

    train_ix = split == 2
    valid_ix = split == 1
    test_ix = split == 0

    train_df = df[train_ix]
    valid_df = df[valid_ix]
    test_df = df[test_ix]

    print("Training: %d, Validation: %d, Test: %d" % (train_df.shape[0], valid_df.shape[0], test_df.shape[0]))
    model = MAPPING[cfg['model']](cfg, df)

    model.train(train_df, valid_df)

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
        "cm" : cm.tolist()
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

    train_test(cfg, df, split)