import numpy as np 

def main():

    rule = PatienceRule(5, 0.05, True)

    seq = [1, 2, 3, 4, 5, 0.96, 7]

    for e in seq:
        r =rule.log(e)

        print("Value = %0.2f, Stop? %s %s" % (e, r[0], '***' if r[1] else ''))

    exit()
    
    print("Linear Rule")
    rule = LinearRule(5, 0.01, False)

    seq = [0.5, 0.51, 0.52, 0.53, 0.56, 0.56, 0.54, 0.54]

    for e in seq:
        r = rule.log(e)

        print("Value = %0.2f, Stop? %s %s" % (e, r[0], '***' if r[1] else ''))

class LinearRule():
    """
        Stop training if improvement over last N trials is less than X 
    """

    def __init__(self, n, thres, minimize):
        self.n = n 
        self.thres = thres 
        self.minimize = minimize
        self._vals = []
        self._best_value = None 
    
    def log(self, value):

        self._vals.append(value)

        # is this new best?
        new_best = self._best_value is None or (value < self._best_value if self.minimize else value > self._best_value)
        if new_best:
            self._best_value = value 
        
        # don't stop if not enough trials
        if len(self._vals) < self.n:
            return (False, new_best)
        
        last_trials = self._vals[-self.n:]

        # fit a line
        z = np.polyfit(np.arange(self.n), last_trials, 1)
        slope = np.abs(z[0])
        print(slope)
        return (slope < self.thres, new_best)

class PatienceRule():

    def __init__(self, patience, thres, minimize):
        self.patience = patience
        self.thres = thres 
        self.minimize = minimize
        self._waited = 0
        self._best_value = None 

    def log(self, value):
        c = -1 if self.minimize else 1

        if self._best_value is None:
            perc_improvement = np.inf 
        else:
            perc_improvement = c * (value - self._best_value) / self._best_value
        
        new_best = perc_improvement > 0
        if new_best:
            self._best_value = value 
            
        # only reset counter if percent improvement is greater than threshold
        self._waited = 0 if (perc_improvement >= self.thres) else (self._waited+1)
        
        # have we reached max patience?, is this a new best?
        return self._waited == self.patience, new_best

if __name__ == "__main__":
    main()
