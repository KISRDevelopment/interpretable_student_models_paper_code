from multiprocessing import Pool, TimeoutError, Manager
import sys 
import os 
import json 
import numpy as np 
import train_test 
import pandas as pd 

def caller(packed):
    q = packed[0]

    cfg = packed[1]

    q, cfg, df, splits, split_id = packed 

    r = train_test.train_test(cfg, df, splits[split_id,:])
    r['split_id'] = split_id

    q.put(r)

    print("Finished %d" % (r['split_id']))

    return r 

def results_listener(q, cfg, output_path):
    results = []

    while True:
        m = q.get()
        if m is None:
            break 
        
        results.append(m)
        
        with open(output_path, 'w') as f:
            json.dump({
                "cfg" : cfg,
                "results" : results
            }, f, indent=4, cls=NumpyEncoder)

def main(cfg_path, df_path, splits_path, output_path, n_workers):

    # load cfg
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    # load data
    df = pd.read_csv(df_path)

    # load CV splits
    splits = np.load(splits_path)

    # initialize processing pool
    pool = Pool(processes=n_workers+1)
    m = Manager()
    q = m.Queue()

    # results accumulator
    listener = pool.apply_async(results_listener, (q, cfg, output_path))
        
    # distribute tasks
    args = [(q, cfg, df, splits, i) for i in range(splits.shape[0])]
    results = pool.map(caller, args)

    q.put(None)
    pool.close()
    pool.join()
    
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    cfg_path = sys.argv[1]
    df_path = sys.argv[2]
    splits_path = sys.argv[3]
    output_path = sys.argv[4]
    n_workers = int(sys.argv[5])

    main(cfg_path, df_path, splits_path, output_path, n_workers)
