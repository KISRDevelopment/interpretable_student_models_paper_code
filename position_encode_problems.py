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
def encode_problem_positions(df, d, n_problems):


    problem_pos = defaultdict(list)
    problem_seqs = defaultdict(list)
    for r in df.itertuples():
        problem_seqs[r.student].append(r.problem)
        problem_pos[r.problem].append(encode_pos(len(problem_seqs[r.student])-1, d))
    
    P = np.zeros((n_problems, d))
    for problem, positions in problem_pos.items():
        P[problem, :] = np.sum(np.vstack(positions).T, axis=1)
    
    return P 
    
def encode_problem_pos_distribs(df, n_problems):

    problem_pos = defaultdict(list)
    problem_seqs = defaultdict(list)
    max_seq_len = 0
    for r in df.itertuples():
        problem_seqs[r.student].append(r.problem)
        problem_pos[r.problem].append(len(problem_seqs[r.student])-1)
        max_seq_len = max(max_seq_len, len(problem_seqs[r.student]))
    
    print("Maximum sequence length: %d" % max_seq_len)

    P = np.zeros((n_problems, max_seq_len))
    for problem, positions in problem_pos.items():
        for pos in positions:
            P[problem, pos] = P[problem, pos] + 1
        P[problem, :] = P[problem, :] / len(positions)
    
    
    return P
    
    

def encode_pos(pos, d, n=10000):
    P = np.zeros(d)
    for i in np.arange(int(d/2)):
        denominator = np.power(n, 2*i/d)
        P[2*i] = np.sin(pos/denominator)
        P[2*i+1] = np.cos(pos/denominator)
    return P 


if __name__ == "__main__":
    main()
