import subprocess
import glob 
import os 

datasets = [os.path.basename(p).replace('.csv','') for p in glob.glob("data/datasets/gervetetal_*") if 'attempt' not in p]
cfg_path = "cfgs/bkt-pytorch.json"
model_script = "torch_bkt.py"
cfg_name = "bkt"

for dataset in datasets:
    print(dataset)
    subprocess.call(['python', model_script, cfg_path, dataset, "data/results-pytorch/%s_%s.json"%(cfg_name, dataset)])
