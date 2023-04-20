import numpy as np 
import pandas as pd 
import os 
import sys 
from collections import defaultdict

def main():
    dataset_name = sys.argv[1]
    d = int(sys.argv[2])
    output_path = sys.argv[3]

    path = "./data/datasets/%s.csv" % dataset_name
    df = pd.read_csv(path)
    n_problems = np.max(df['problem']) + 1
    
    #P = encode_problem_positions(df, d, n_problems)
    P = encode_problem_pos_distribs(df, n_problems)
    print(P)

    
def encode_problem_pos_distribs(df, n_problems):

    gdf = df.groupby('student')['student'].count()

    max_seq_len = np.max(gdf)
    
    seqs = defaultdict(lambda: {
        "problem" : [],
        "correct" : []
    })
    for r in df.itertuples():
        seqs[r.student]['problem'].append(r.problem)
        seqs[r.student]['correct'].append(r.correct)
    

    P = np.zeros((n_problems, max_seq_len))
    Pc = np.zeros((n_problems, max_seq_len))

    for student in seqs.keys():
        problems = seqs[student]['problem']
        corrects = seqs[student]['correct']
        for t in range(len(problems)):
            problem = problems[t]
            correct = corrects[t]
            P[problem, t] += 1
            Pc[problem, t] += correct
    
    P = P / np.sum(P, axis=1, keepdims=True)
    Pc = Pc / np.sum(Pc, axis=1, keepdims=True)
     
    return np.hstack((P, Pc))
    
    

def encode_pos(pos, d, n=10000):
    P = np.zeros(d)
    for i in np.arange(int(d/2)):
        denominator = np.power(n, 2*i/d)
        P[2*i] = np.sin(pos/denominator)
        P[2*i+1] = np.cos(pos/denominator)
    return P 


if __name__ == "__main__":
    main()
