import numpy as np 

def main():

    rule = PatienceRule(5, 0.05, True)

    seq = [1, 2, 3, 4, 5, 0.96, 7]

    for e in seq:
        print("Value = %d, Stop? %s %s" % (e, rule.log(e), '***' if rule._waited == 0 else ''))

    

class PatienceRule():

    def __init__(self, patience, thres, minimize):
        self.patience = patience
        self.thres = thres 
        self.minimize = minimize
        self._waited = 0
        self._best_value = None 

    def log(self, value):

        if self._best_value is None:
            self._best_value = value 
            self._waited = 0
            return (False, True) # do not stop, new best
        
        c = -1 if self.minimize else 1

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
