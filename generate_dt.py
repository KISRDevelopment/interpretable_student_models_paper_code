import numpy as np 
import numpy.random as rng 
import json 

def main():

    n_rows = 16
    n_cols = 16
    ns_one_dim = np.arange(n_cols)
    ns_two_dim = np.arange(n_rows)
    
    reps = 10

    all_rules = set()
    for n_one_dim in ns_one_dim:
        for n_two_dim in ns_two_dim:
            
            for r in range(reps):
                rules, bool_mat = generate(n_rows, n_cols, n_one_dim, n_two_dim)
                
                d1 = calculate_difficulty(rules)
                neg_rules = sparsify(1-bool_mat)
                d2 = calculate_difficulty(neg_rules)

                if d2 < d1:
                    rules = neg_rules
                
                rules = tuple(sorted(rules))
                if rules not in all_rules:
                    all_rules.add(rules)
                
    print("# unique rules: %d"%len(all_rules))

    difficulties = [calculate_difficulty(rules) for rules in all_rules]

    import matplotlib.pyplot as plt 

    f,ax = plt.subplots(1,1,figsize=(10,10))
    bin_counts = np.bincount(difficulties)
    print(np.min(bin_counts))
    ax.bar(np.arange(0, np.max(difficulties)+1), bin_counts)
    f.savefig("tmp/hist.png")

    print(np.histogram(difficulties))


def generate(n_rows, n_cols, n_one_dim, n_two_dim):

    one_dim_rules = []
    bool_mat = np.zeros((n_rows, n_cols),dtype=bool)
    
    while len(one_dim_rules) < n_one_dim:
        gen_col = rng.binomial(1, 0.5) == 1
        new_mat = np.zeros((n_rows, n_cols), dtype=bool)
        if gen_col:
            col = rng.choice(n_cols)
            new_mat[:,col] = 1
            rule = (-1, col)
        else:
            row = rng.choice(n_rows)
            new_mat[row,:] = 1
            rule = (row, -1)

        if np.any(new_mat | bool_mat != bool_mat):
            one_dim_rules.append(rule)
            bool_mat |= new_mat 
    
        if np.sum(bool_mat) == n_rows*n_cols:
            one_dim_rules = []
            bool_mat = np.zeros((n_rows, n_cols),dtype=bool)
    
        
    two_dim_rules = []
    while len(two_dim_rules) < n_two_dim:
        row = rng.choice(n_rows)
        col = rng.choice(n_cols)
        new_mat = np.zeros((n_rows, n_cols), dtype=bool)
        new_mat[row, col] = 1
        if np.any(new_mat | bool_mat != bool_mat):
            two_dim_rules.append((row,col))
            bool_mat |= new_mat 

    return set(one_dim_rules + two_dim_rules), bool_mat
    
def sparsify(bool_mat):

    n_rows, n_cols = bool_mat.shape

    rules = set()
    rem_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]

    for r in range(n_rows):
        if np.sum(bool_mat[r,:]) == n_cols:
            rules.add((r, -1))
            rem_cells = [rc for rc in rem_cells if rc[0] != r]
        
    for c in range(n_cols):
        if np.sum(bool_mat[:,c]) == n_rows:
            rules.add((-1, c))
            rem_cells = [rc for rc in rem_cells if rc[1] != c]
    
    for rc in rem_cells:
        if bool_mat[rc[0], rc[1]]:
            rules.add(rc)
    
    return rules
def calculate_difficulty(rules):
    n_one_dim = 0
    n_two_dim = 0
    for rule in rules:
        if -1 in rule:
            n_one_dim += 1
        else:
            n_two_dim += 1
    return n_one_dim + n_two_dim*2

def to_matrix(rules, n_rows, n_cols):
    mat = np.zeros((n_rows, n_cols))
    for r in rules:
        if r[0] == -1:
            mat[:, r[1]] = 1
        elif r[1] == -1:
            mat[r[0],:] = 1
        else:
            mat[r[0],r[1]] = 1
    return mat 
if __name__ == "__main__":
    main()