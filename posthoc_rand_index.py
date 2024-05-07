import numpy as np 
import pandas as pd 
import sys 
import sklearn.metrics
import sklearn.cluster
import re 
import cluster_metrics
def main():
    raw_results_dir = sys.argv[1]
    results_file = sys.argv[2]

    all_results_df = pd.read_csv(results_file)
    
    datasets = set(all_results_df['dataset'])
    models = set(all_results_df['model'])
    ri_col = np.zeros(all_results_df.shape[0])
    fi_col = np.zeros_like(ri_col)
    vi_col = np.zeros_like(fi_col)
    for dataset_name in datasets:
        
        df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
        
        ref_assignment = get_problem_skill_assignment(df)
        problem_feature_mat = np.load("data/datasets/%s.embeddings.npy" % dataset_name.replace('_blocked', '').replace('_interleaved',''))
        
        for model in models:

            ri = []
            fi = []
            vi = []

            pred_assignments = []
            if model.startswith('sd'):
                if 'realistic' in dataset_name:

                    params_path = "%s/%s_%s.params.npy.npz" % (raw_results_dir, re.sub(r'^sd', 'sd-50kcs', model), dataset_name)
                else:
                    params_path = "%s/%s_%s.params.npy.npz" % (raw_results_dir, model, dataset_name)
                params = np.load(params_path)
                Aprior = params['Aprior'] # Splits x Problems x KCs
            
                for i in range(Aprior.shape[0]):
                    Q = Aprior[i, :, :] 
                    pred_assignment = np.argmax(Q, axis=1)
                    pred_assignments.append(pred_assignment)

            elif model.startswith('clustering') and not dataset_name.startswith('sd_1_'):
                splits = np.load("data/splits/%s.npy" % dataset_name)
                
                for s in range(splits.shape[0]):
                    split = splits[s, :]

                    train_ix = split == 2
                    train_df = df[train_ix]

                    #
                    # build problem clustering model based on training problems only
                    #
                    train_problems = sorted(pd.unique(train_df['problem']))
                    train_problem_features = problem_feature_mat[train_problems, :]
                    kmeans_model = sklearn.cluster.KMeans(n_clusters=20, n_init='auto', random_state=0).fit(train_problem_features)
                    problem_labels = kmeans_model.predict(problem_feature_mat) # predict labels for all problems
                    pred_assignments.append(problem_labels)
                
            else:
                continue 
            
            for pred_assignment in pred_assignments:
                ri.append(sklearn.metrics.rand_score(ref_assignment, pred_assignment))
                #fi.append(cluster_metrics.fmeasure(ref_assignment, pred_assignment))
                vi.append(cluster_metrics.recovered(ref_assignment, pred_assignment))
            
            ix = (all_results_df['model'] == model) & (all_results_df['dataset'] == dataset_name)
            ri_col[ix] = ri
            #fi_col[ix] = fi 
            vi_col[ix] = vi 

    all_results_df['raw_rand_index'] = ri_col
    #all_results_df['fmeasure'] = fi_col
    all_results_df['recovered'] = vi_col

    all_results_df.to_csv(results_file, index=False)
    print(all_results_df)

def get_problem_skill_assignment(df):

    problems_to_skills = dict(zip(df['problem'], df['skill']))
    n_problems = np.max(df['problem']) + 1
    return np.array([problems_to_skills[p] for p in range(n_problems)])


if __name__ == "__main__":
    main()
