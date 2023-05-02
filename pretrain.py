import torch as th 
import torch.nn as nn 
import numpy as np
import numpy.random as rng 
import itertools

cfg = {
    "learning_rate" : 0.01,
    "epochs" : 100,
    "p_batch_size" : 0.1,
    "n_hidden" : 10
}
def main():

    #
    # generate reference BKT data
    #
    X, y = generate_synth_data(n_probs=10, n_prev_hs=100)

    #
    # train
    #
    model = EmaFilteringMultiplexor()
    
    optimizer = th.optim.NAdam(model.parameters(), lr=cfg['learning_rate'])
    n_batch_size = int(cfg['p_batch_size'] * X.shape[0])

    for e in range(cfg['epochs']):
        #
        # shuffle
        #
        ix = rng.permutation(X.shape[0])
        X = X[ix, :]
        y = y[ix]

        losses = []
        for offset in range(0, X.shape[0], n_batch_size):
            end = offset + n_batch_size

            prev_h = th.tensor(X[offset:end, [0]]).float()
            prev_y = th.tensor(X[offset:end, [1]]).float()
            bkt_params = th.tensor(X[offset:end, 2:]).float()
            bkt_logits = th.log( bkt_params / (1-bkt_params) )
            actual_h = th.tensor(y[offset:end])[:,None]
            
            next_h = model(prev_h, prev_y, bkt_logits)
            
            loss = (next_h - actual_h).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        
        print("%4d Train loss: %8.4f" % (e, np.mean(losses)))
def generate_synth_data(n_probs, n_prev_hs):

    probs = np.linspace(0.01, 0.99, n_probs)
    prev_hs = np.linspace(0.01, 0.99, n_prev_hs)
    prev_ys = [0, 1]

    combs = np.array(list(itertools.product(
        prev_hs, prev_ys, probs, probs, probs, probs)))
    
    next_h = bkt_filtering(combs[:, 0], 
        combs[:, 1], 
        combs[:, 2],
        combs[:, 3],
        combs[:, 4],
        combs[:, 5])
    
    return combs, next_h

def bkt_filtering(prev_h, prev_y, pL, pF, pG, pS):
    
    output_prob_0 = np.power(pG, prev_y) * np.power(1-pG, 1-prev_y)
    output_prob_1 = np.power(1-pS, prev_y) * np.power(pS, 1-prev_y)
    
    unnormed_0 = output_prob_0 * ((1-pL) * (1-prev_h) + pF * prev_h)
    unnormed_1 = output_prob_1 * (pL * (1-prev_h) + (1-pF) * prev_h)
    
    return unnormed_1 / (unnormed_0 + unnormed_1)
 
class EmaFilteringMultiplexor(nn.Module):

    def __init__(self):
        super(EmaFilteringMultiplexor, self).__init__()

        self.linear = EmaFilteringLinear()
        self.logit = EmaFilteringLogit()

        self.logit_weight_linear = nn.Linear(4, 1)
     
    def forward(self, prev_h, prev_y, bkt_params):
        
        weight = th.sigmoid( self.logit_weight_linear(bkt_params) ) # Bx1

        linear_output = self.linear(prev_h, prev_y, bkt_params)
        logit_output = th.sigmoid(self.logit(prev_h, prev_y, bkt_params))

        return weight * linear_output + (1-weight) * logit_output

class EmaFilteringLinear(nn.Module):

    def __init__(self):
        super(EmaFilteringLinear, self).__init__()
        

        self._logit_y1_anchors = nn.Linear(4, 2)
        self._logit_y0_anchors = nn.Linear(4, 2)

    
    def forward(self, prev_h, prev_y, bkt_params):
        """
            prev_h: Bx1
            prev_y: Bx1
            bkt_params: Bx4
        """
        
        y1_anchors = self._logit_y1_anchors(bkt_params)
        y1_anchors = th.sigmoid(y1_anchors)

        y0_anchors = self._logit_y0_anchors(bkt_params)
        y0_anchors = th.sigmoid(y0_anchors)

        # Bx1
        return self.forward_(prev_h, prev_y, y1_anchors, y0_anchors)

    def forward_(self, prev_h, prev_y, y1_anchors, y0_anchors):
        y1_offset = y1_anchors[:, [0]]
        y1_final = y1_anchors[:, [1]]
        y0_offset = y0_anchors[:, [0]]
        y0_final = y0_anchors[:, [1]]

        # Bx1
        return prev_y * (y1_offset + (y1_final - y1_offset) * prev_h) \
            +  (1-prev_y) * (y0_offset + (y0_final - y0_offset) * prev_h)

class EmaFilteringLogit(nn.Module):

    def __init__(self):
        super(EmaFilteringLogit, self).__init__()
        

        self._logit_y1_anchors = nn.Linear(4, 2)
        self._logit_y0_anchors = nn.Linear(4, 2)

    
    def forward(self, prev_h, prev_y, bkt_params):
        """
            prev_h: Bx1
            prev_y: Bx1
            bkt_params: Bx4
        """
        
        y1_anchors = self._logit_y1_anchors(bkt_params)
        
        y0_anchors = self._logit_y0_anchors(bkt_params)
        
        # Bx1
        return self.forward_(prev_h, prev_y, y1_anchors, y0_anchors)

    def forward_(self, prev_h, prev_y, y1_anchors, y0_anchors):
        y1_offset = y1_anchors[:, [0]]
        y1_final = y1_anchors[:, [1]]
        y0_offset = y0_anchors[:, [0]]
        y0_final = y0_anchors[:, [1]]

        prev_h_logit = th.log(prev_h / (1-prev_h))

        # Bx1
        return prev_y * (y1_offset + y1_final * prev_h_logit) \
            +  (1-prev_y) * (y0_offset + y0_final * prev_h_logit)

if __name__ == "__main__":
    main()