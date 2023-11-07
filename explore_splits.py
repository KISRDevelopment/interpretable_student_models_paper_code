import numpy as np 
import pandas as pd 
from collections import defaultdict

def main():

    datasets = ['gervetetal_algebra05', 
                'gervetetal_assistments09',
                'gervetetal_assistments12',
                'gervetetal_assistments15',
                'gervetetal_assistments17',
                'gervetetal_bridge_algebra06',
                'gervetetal_spanish',
                'gervetetal_statics',
                'equations'
    ]

    summaries = []
    for dataset_name in datasets:
        print(dataset_name)
        df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
        splits = np.load("data/splits/%s.npy" % dataset_name)
        results_df = explore(df, splits)
        summary_df = np.mean(results_df, axis=0)
        summary_df['dataset'] = dataset_name
        summaries.append(summary_df)
    
    final_summary = pd.concat(summaries, axis=1).T
    print(final_summary)
    final_summary.to_csv("tmp/splits_summary.csv", index=False)
    
def explore(df, splits):
    
    results = defaultdict(list)
    for s in range(splits.shape[0]):
        split = splits[s, :]

        train_ix = split == 2
        valid_ix = split == 1
        test_ix = split == 0

        train_df = df[train_ix]
        valid_df = df[valid_ix]
        test_df = df[test_ix]

        train_problems = set(train_df['problem'])
        valid_problems = set(valid_df['problem'])
        test_problems = set(test_df['problem'])

        novel_test_problems = test_problems - train_problems
        
        results['students'].append(len(set(df['student'])))
        results['skills'].append(len(set(df['skill'])))
        results['problems'].append(len(set(df['problem'])))

        results['train_problems'].append(len(train_problems))
        results['test_problems'].append(len(test_problems))
        
        n_trials_with_novel_problem = np.sum(test_df['problem'].isin(novel_test_problems))
        results['p_novel_trials'] = n_trials_with_novel_problem / test_df.shape[0]

        results['p_novel_problems'] = len(novel_test_problems) / len(test_problems)
        results['p_novel_skills'] = len(set(test_df['skill']) - set(train_df['skill'])) / len(set(test_df['skill']))
        gdf = test_df.groupby('student')['skill'].agg('count')
        results['med_seq_len'] = np.median(gdf)
        
    output_df = pd.DataFrame(results)
    return output_df

    #print(np.mean(output_df, axis=0))

def create_one_to_many(a, b):
    mapping = defaultdict(list)
    for i,j in zip(a,b):
        mapping[i].append(j)
    return mapping 

if __name__ == "__main__":
    main()