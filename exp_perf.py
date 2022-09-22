import subprocess
import glob 
import os 

datasets = [os.path.basename(p).replace('.csv','') for p in 
    glob.glob("data/datasets/perf_*")]

os.makedirs("data/results-perf", exist_ok=True)

cfg_name = 'bkt-brute-force'
datasets = sorted(datasets, key=lambda d: int(d.split('_')[1]))
for dataset in datasets:
    if os.path.exists("data/results-perf/%s_%s.csv"%(cfg_name, dataset)):
        continue
    
    print(cfg_name, dataset)
    dataset_path = "data/datasets/%s.csv" % dataset
    splits_path = "data/splits/%s.npy" % dataset 
    output_path = "data/results-perf/%s_%s.csv" % (cfg_name, dataset)


    subprocess.call(['python', "model_brute_force_bkt.py",  dataset_path, splits_path, output_path])

cfg_name = 'torch-bkt'
cfg_path = "cfgs/bkt-pytorch.json"
model_script = "torch_bkt.py"

for dataset in datasets:
    if os.path.exists("data/results-perf/%s_%s.csv"%(cfg_name, dataset)):
        continue
    
    print(cfg_name, dataset)
    dataset_path = "data/datasets/%s.csv" % dataset
    splits_path = "data/splits/%s.npy" % dataset 
    output_path = "data/results-perf/%s_%s.csv" % (cfg_name, dataset)


    subprocess.call(['python', model_script, cfg_path, dataset, output_path])
