import sys 
import os 
import numpy as np 
import utils 
import subprocess 
import glob 
def main(cfg_path, df_path, splits_path, cv_output_path, n_workers):
    
    base_cv_output_path = os.path.basename(cv_output_path)

    # load CV splits
    splits = np.load(splits_path)
    n_splits = splits.shape[0]

    output_paths = []
    for s in range(n_splits):
        files = glob.glob('tmp/*')
        for file in files:
            os.remove(file)

        output_path = "results/%s_split%d.json" % (base_cv_output_path.replace('.json',''), s)
        output_paths.append(output_path)

        subprocess.call([
            "python",
            "train_test_with_hyperparam_opt.py",
            cfg_path,
            df_path,
            splits_path,
            str(s),
            output_path,
            str(n_workers)
        ])
    
    results = [utils.load_json(p) for p in output_paths]

    utils.write_json(cv_output_path, results)

    for p in output_paths:
        os.remove(p)
    
if __name__ == "__main__":
    cfg_path = sys.argv[1]
    df_path = sys.argv[2]
    splits_path = sys.argv[3]
    cv_output_path = sys.argv[4]
    n_workers = int(sys.argv[5])

    main(cfg_path, df_path, splits_path, cv_output_path, n_workers)
