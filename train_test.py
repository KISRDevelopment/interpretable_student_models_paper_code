import numpy as np 
import pandas as pd
import sklearn.metrics 
import model_standard_bkt

MAPPING = {
    "model_standard_bkt" : model_standard_bkt.create_model,
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

    preds = model.predict(test_df)
    actual = np.array(test_df['correct'])

    xe = -(actual * np.log(preds) + (1-actual) * np.log(1-preds))
    print("Test XE: %0.2f" % np.mean(xe))

    auc = sklearn.metrics.roc_auc_score(actual, preds)
    print("AUC-ROC: %0.2f" % auc)

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