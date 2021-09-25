import subprocess
import json
import utils
import copy
import os 

def main(cfg_path, df_path, splits_path, split_id, final_output_path, n_workers, show_output=False):

    cfg = utils.load_json(cfg_path)
    basename = os.path.basename(cfg_path).replace('.json','')

    # generate cfgs for hyperparam optimization
    cfg_paths = []
    for i, comb in enumerate(utils.hyperparam_combs(cfg['hyperparams'])):
        comb_cfg = copy.deepcopy(cfg)
        comb_cfg.update(comb)
        cfg_path = "tmp/%s_%d.json" % (basename, i)
        utils.write_json(cfg_path, comb_cfg)
        cfg_paths.append(cfg_path)

    # distribute tasks
    output_paths = []
    finished = 0
    for i in range(0, len(cfg_paths), n_workers):
        
        subset = cfg_paths[i:(i+n_workers)]
        processes = []
        for cfg_path in subset:
            output_path = "tmp/%s__results.json" % (os.path.basename(cfg_path))
            output_paths.append((cfg_path, output_path))

            print("Spawning %s" % cfg_path)
            kwargs = {}
            if not show_output:
                kwargs["stdout"] = subprocess.DEVNULL
                kwargs["stderr"] = subprocess.DEVNULL

            processes.append(subprocess.Popen(['python', 
                "train_test.py", 
                cfg_path, 
                df_path,
                splits_path,
                str(split_id),
                output_path,
                str(2),
                str(1),
                str(1), # we don't want to evaluate performance on test just yet. We want validation
            ],  **kwargs))


        for p in processes:
            p.wait()
            finished += 1
            print("Finished %d" % finished)
        
    # get results
    results = []
    for p in output_paths:
        if os.path.exists(p[1]):
            r = utils.load_json(p[1])
            results.append(r)
        else:
            print("Warning: No output from %s" % p)
        
    
    # pick the best on the validation set
    best_result_ix = min(range(len(results)), key=lambda r: results[r]['xe'])
    best_result = results[best_result_ix]
    best_cfg_path = output_paths[best_result_ix][0]

    print("Best result:")
    print(best_result)
    print(best_cfg_path)

    
    # execute on test
    subprocess.call(['python', 
        "train_test.py", 
        best_cfg_path, 
        df_path,
        splits_path,
        str(split_id),
        final_output_path,
        str(2),
        str(1),
        str(0),
        best_result['model_params_path']
    ])

if __name__ == "__main__":

    import sys 

    cfg_path = sys.argv[1]
    df_path = sys.argv[2]
    split_path = sys.argv[3]
    split_id = int(sys.argv[4])
    output_path = sys.argv[5]
    n_workers = int(sys.argv[6])
    main(cfg_path, df_path, split_path, split_id, output_path, n_workers=n_workers)