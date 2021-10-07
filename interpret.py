import numpy as np 
import pandas as pd
import sklearn.metrics 
import model_bkt
import model_dkt 
import model_ldkt 
import model_bkt_skill_discovery
import model_ldkt_skill_discovery
import model_dash_alt
import utils 
from multiprocessing import Pool, TimeoutError, Manager
import copy 

MAPPING = {
    "bkt-sd" : model_bkt_skill_discovery.create_model,
    "ldkt-sd" : model_ldkt_skill_discovery.create_model,
    "dash" : model_dash_alt.create_model
}

def interpret(cfg, df, split, model_params_path):

    train_ix = split == 2
    valid_ix = split == 1
    test_ix = split == 0

    # for interpretation, lets combine train and test sets
    train_ix = train_ix | test_ix 

    train_df = df[train_ix]
    valid_df = df[valid_ix]
    print("Training: %d, Validation: %d" % (train_df.shape[0], valid_df.shape[0]))

    # train model        
    model = MAPPING[cfg['model']](cfg, df)
    model.train(train_df, valid_df)
    model.save(model_params_path)
    

if __name__ == "__main__":

    import sys 
    import json 

    cfg_path = sys.argv[1]
    df_path = sys.argv[2]
    split_path = sys.argv[3]
    model_params_path = sys.argv[4]
    split_id = 0

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    df = pd.read_csv(df_path)

    splits = np.load(split_path)
    split = splits[split_id, :]

    r = interpret(cfg, df, split, model_params_path)

    