import numpy as np 
import numpy.random as rng 
import json 

def main():

    bool_mat = np.array([
        [0, 0, 1, 0],
        [1, 1, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 0]
    ])
    bool_mat = np.array([
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1]
    ])
    rules = sparsify(bool_mat)
    print(rules)
    
    
def sparsify(bool_mat):

    n_rows, n_cols = bool_mat.shape

    rules = set()
    rem_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]

    for r in range(n_rows):
        if np.sum(bool_mat[r,:]) == n_cols:
            rules.add((r, None))
            rem_cells = [rc for rc in rem_cells if rc[0] != r]
        
    for c in range(n_cols):
        if np.sum(bool_mat[:,c]) == n_rows:
            rules.add((None, c))
            rem_cells = [rc for rc in rem_cells if rc[1] != c]
    
    for rc in rem_cells:
        if bool_mat[rc[0], rc[1]]:
            rules.add(rc)
    
    return rules

if __name__ == "__main__":
    main()