mkdir -p data/exp_equations

# BKT
python model_bkt_irt.py cfgs/exp_equations/fbkt.json equations data/exp_equations/fbkt_equations.csv
cp cfgs/exp_equations/fbkt.json data/exp_equations/fbkt_equations.json

# # Single KC BKT
python model_bkt_irt.py cfgs/exp_equations/single-kc.json equations data/exp_equations/single-kc_equations.csv
cp cfgs/exp_equations/single-kc.json data/exp_equations/single-kc_equations.json

# # Problem as KC BKT
python model_bkt_irt.py cfgs/exp_equations/no-sd.json equations data/exp_equations/no-sd_equations.csv
cp cfgs/exp_equations/no-sd.json data/exp_equations/no-sd_equations.json

# Skill discovery with embeddings
python model_sd_old.py cfgs/exp_equations/sd.json equations data/exp_equations/sd-20-embeddings_equations.csv data/datasets/equations.embeddings.npy &
cp cfgs/exp_equations/sd.json data/exp_equations/sd-20-embeddings_equations.json

# Skill discovery without embeddings
python model_sd_old.py cfgs/exp_equations/sd.json equations data/exp_equations/sd-20_equations.csv &
cp cfgs/exp_equations/sd.json data/exp_equations/sd-20_equations.json

# Skill discovery with embeddings
python model_sd_old.py cfgs/exp_equations/sd-50.json equations data/exp_equations/sd-50-embeddings_equations.csv data/datasets/equations.embeddings.npy &
cp cfgs/exp_equations/sd-50.json data/exp_equations/sd-50-embeddings_equations.json

# Skill discovery without embeddings
python model_sd_old.py cfgs/exp_equations/sd-50.json equations data/exp_equations/sd-50_equations.csv &
cp cfgs/exp_equations/sd-50.json data/exp_equations/sd-50_equations.json

