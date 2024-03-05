import numpy as np 
import pandas as pd 
import json 

def main():

    datasets = ['gervetetal_assistments09', 'gervetetal_bridge_algebra06', 'gervetetal_statics']

    counts = {}
    for dataset in datasets:

        df = pd.read_csv("data/datasets/sd-realistic_%s.csv" % dataset)
        gdf = df.groupby('skill')['problem'].count()
        counts[dataset] = gdf.tolist()
    
    with open('tmp/shadow_kc_distrib.json', 'w') as f:
        json.dump(counts, f)

if __name__ == "__main__":
    main()
