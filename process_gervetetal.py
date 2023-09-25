import numpy as np
import pandas as pd
import glob
import split_dataset
import os
from collections import defaultdict

os.makedirs('./data/datasets', exist_ok=True)
os.makedirs('./data/splits', exist_ok=True)

datasets = glob.glob("../learner-performance-prediction/data/*")
for dataset in datasets:
    path = "%s/preprocessed_data.csv" % dataset
    df = pd.read_csv(path, sep='\t')
    
    if 'Unnamed: 0' in df.columns:
        df = df.set_index('Unnamed: 0')
    print(df.columns)
    
    dataset_name = os.path.basename(dataset)
    print(dataset_name)
    
    df.columns = ['student', 'problem', 'timestamp', 'correct', 'skill']
    
    df.to_csv("data/datasets/gervetetal_%s.csv" % dataset_name, index=False)
    full_splits = split_dataset.main(df)
    np.save("data/splits/gervetetal_%s.npy" % dataset_name, full_splits)

datasets = glob.glob("../learner-performance-prediction/data/*")
for dataset in datasets:
    
    dataset_name = os.path.basename(dataset)
    print(dataset_name)
    
    df = pd.read_csv("data/datasets/gervetetal_%s.csv" % dataset_name)
    splits = np.load("data/splits/gervetetal_%s.npy" % dataset_name)
    
    train_ix = splits[0, :] == 2
    valid_ix = splits[0, :] == 1
    test_ix = splits[0, :] == 0
    
    train_df = df[train_ix]
    test_df = df[test_ix]
    
    train_items = set(train_df['problem'])
    test_items = set(test_df['problem'])
    
    print("Items in testing not in training: %d (total items: %d)" % 
          (len(test_items - train_items), len(set(df['problem']))))
    
    items_to_kc = defaultdict(set)
    for item, skill in zip(df['problem'], df['skill']):
        items_to_kc[item].add(skill)
    
    for item, kcs in items_to_kc.items():
        if len(kcs) > 1:
            print("Warning: Item %d has more than one KC" % item)
