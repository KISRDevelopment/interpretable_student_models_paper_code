import numpy as np
import numpy.random as rng 
import pandas as pd 
from sklearn.model_selection import GroupKFold

rng.seed(41)

def main(dataset_path, output_path, n_folds=5, n_folds_inner=5):

    df = pd.read_csv(dataset_path)

    students = np.array(df['student'])
    group_kfold = GroupKFold(n_splits=n_folds)

    # used to split training into training/validation
    group_kfold_inner = GroupKFold(n_splits=n_folds_inner)

    full_splits = []
    tested_counts = np.zeros(np.max(students)+1)
    for train_index, test_index in group_kfold.split(np.zeros((students.shape[0],1)), groups=students):
        
        train_students = np.array(students[train_index])
        
        test_students = list(set(students[test_index]))
        tested_counts[test_students] += 1
        
        for train_index_inner, valid_index in group_kfold_inner.split(np.zeros((train_students.shape[0],1)), groups=train_students):
            
            full_split = np.zeros(df.shape[0],dtype=int)
            full_split[ train_index[train_index_inner] ] = 2
            full_split[ train_index[valid_index] ] = 1
            full_splits.append(full_split)
            
            set_train = set(df[ full_split == 2]['student'])
            set_valid = set(df[ full_split == 1]['student'])
            set_test = set(df[ full_split == 0]['student'])
            
            # sanity checks for overlaps between sets
            assert set_train & set_valid == set()
            assert set_train & set_test == set()
            assert set_valid & set_test == set()
            assert set_train | set_valid | set_test == set(df['student'])
            assert len(set_train) > 0
            assert len(set_valid) > 0
            assert len(set_test) > 0
            
            sizes = np.array([len(set_train), len(set_valid), len(set_test)])
            print((sizes / np.sum(sizes)))
            break # only need once
            
    # ensure that all students are tested exactly once
    student_set = list(set(students))
    assert np.all(tested_counts[student_set] == 1)

    full_splits = np.array(full_splits).astype('uint8')

    # ensure no repeated splits
    assert np.unique(full_splits, axis=0).shape[0] == full_splits.shape[0]

    np.save(output_path, full_splits)

if __name__ == "__main__":
    import sys 

    main(sys.argv[1], sys.argv[2])