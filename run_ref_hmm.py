import numpy as np 
import pandas as pd 
import subprocess 
import shlex 
import re 
import metrics 

TRAIN_COMMAND = "../hmm-scalable/trainhmm -s 1.1 -m 1 -p 1 -l 0,0,0,0,0,0,0,0,0,0 -u 1,1,1,1,1,1,1,1,1,1 %s tmp/model.txt tmp/predict.csv"
PREDICT_COMMAND = "../hmm-scalable/predicthmm %s tmp/model.txt tmp/predict.csv"

def run(train_df, test_df):

    train_df[['correct', 'student', 'problem', 'skill']].to_csv("tmp/tmpformat_train.csv", sep="\t", index=False, header=None)
    test_df[['correct', 'student', 'problem', 'skill']].to_csv("tmp/tmpformat_test.csv", sep="\t", index=False, header=None)
    
    output = subprocess.check_output(shlex.split(TRAIN_COMMAND % ('tmp/tmpformat_train.csv',)))
    last_line = str(output, 'utf-8').split("\n")[-2]
    assert "timing" in last_line
    parts = last_line.split(" ")
    fit_time_sec = float(parts[7][:-1])
    
    output = subprocess.check_output(shlex.split(PREDICT_COMMAND % ('tmp/tmpformat_test.csv',)))
    lines = str(output, 'utf-8').split("\n")
    time_line = lines[-4].split(" ")
    predict_time_sec = float(time_line[4])
    
    predict_df = pd.read_csv("tmp/predict.csv", header=None, sep='\t')
    predict_df['correct'] = test_df['correct']

    run_result = metrics.calculate_metrics(np.array(predict_df['correct'] == 1), predict_df[0].to_numpy())
    run_result['time_diff_sec'] = predict_time_sec + fit_time_sec
    return run_result
    
if __name__ == "__main__":
    dataset_name = 'perf_128'
    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    splits = np.load("data/splits/%s.npy" % dataset_name)
    
    df['correct'] = 1 + (1-df['correct'])

    results = []
    for s in range(splits.shape[0]):
        split = splits[s, :]

        train_ix = split > 0
        test_ix = split == 0

        train_df = df[train_ix]
        test_df = df[test_ix]

        results.append(run(train_df, test_df))
    
    results_df = pd.DataFrame(results, index=["Split %d" % s for s in range(splits.shape[0])])
    print(results_df)

    #results_df.to_csv(output_path)
