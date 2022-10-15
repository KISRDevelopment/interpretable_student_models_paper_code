import numpy as np 
import numpy.random as rng 
import sklearn.metrics 

n_skills = 5
n_problems_per_skill = 40
n_problems = n_skills * n_problems_per_skill

skill = np.repeat(np.arange(n_skills), n_problems_per_skill)
problem = rng.permutation(n_problems)
actual_label = np.zeros(problem.shape[0],dtype=int)
actual_label[problem] = skill 

corruption_rates = np.linspace(0, 1, 11)
for rate in corruption_rates:

    n_problms_to_corrupt = int(rate * problem.shape[0])
    rand_indices = []

    for s in range(100):
        if n_problms_to_corrupt > 0:
            indices = rng.choice(np.arange(n_problems), size=n_problms_to_corrupt, replace=False)
            corrupted_label = actual_label.copy()
            corrupted_label[indices] = rng.choice(np.arange(n_skills), size=n_problms_to_corrupt)
        else:
            corrupted_label = actual_label
        
        rand_index = sklearn.metrics.adjusted_rand_score(actual_label, corrupted_label)
        rand_indices.append(rand_index)
    
    print("%0.2f Rand: %0.2f" % (rate, np.mean(rand_indices)))