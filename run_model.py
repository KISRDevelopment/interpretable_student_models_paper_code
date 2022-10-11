import subprocess
import glob 
import os 

datasets = [os.path.basename(p).replace('.csv','') for p in glob.glob("data/datasets/gervetetal_*") if 'attempt' not in p]
# cfg_path = "cfgs/bkt-pytorch.json"
# model_script = "torch_bkt.py"
# cfg_name = "bkt"

# cfg_path = "cfgs/bkt-pytorch.json"
# model_script = "torch_bkt_problems.py"
# cfg_name = "bkt-problems"

# cfg_path = "cfgs/dkt-pytorch.json"
# model_script = "dkt.py"
# cfg_name = "dkt"

# cfg_path = "cfgs/bkt-abilities-pytorch.json"
# model_script = "torch_bkt_irt.py"
# cfg_name = "bkt-irt"

# cfg_path = "cfgs/bkt-irt-lr.json"
# model_script = "torch_bkt_irt_lr.py"
# cfg_name = "bkt-lr"

# cfg_path = "cfgs/dkt-pytorch.json"
# model_script = "dkt_items.py"
# cfg_name = "dkt-items"

cfg_path = "cfgs/bkt-pytorch-skill-discovery.json"
model_script = "torch_bkt_skill_discovery.py"
cfg_name = "bkt-sd"


for dataset in datasets:
    if os.path.exists("data/results-pytorch/%s_%s.csv"%(cfg_name, dataset)):
        continue
    
    print(dataset)
    subprocess.call(['python', model_script, cfg_path, dataset, "data/results-pytorch/%s_%s.csv"%(cfg_name, dataset)])
