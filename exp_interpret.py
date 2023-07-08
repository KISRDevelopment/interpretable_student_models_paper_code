import model_bkt_irt_multidim_abilities
import pandas as pd 
import numpy as np 
import torch as th 
import sys 
import json 
import utils 
from torch.nn.utils.rnn import pad_sequence
def main():
    cfg_path = sys.argv[1]
    dataset_name = sys.argv[2]
    state_dict_path = sys.argv[3]
    output_path = sys.argv[4]

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    splits = np.load("data/splits/%s.npy" % dataset_name)

    cfg['n_kcs'] = np.max(df['skill']) + 1
    cfg['n_problems'] = np.max(df['problem']) + 1
    cfg['device'] = 'cuda:0'

    state_dicts = th.load(state_dict_path)

    split_id = 0

    model = model_bkt_irt_multidim_abilities.BktModel(cfg)
    model.load_state_dict(state_dicts[split_id])
    model = model.to(cfg['device'])

    seqs = utils.to_seqs(df)

    test_seqs = [seqs[s] for s in seqs.keys()] 
    
    cfg['n_test_batch_seqs'] = 10
    print("# test sequences: %d" % len(test_seqs))
    seq_lens, ypred, ytrue, posteriors, seq_loss, seq_mu, all_problem, all_kc = predict(cfg, model, test_seqs)
    
    # compute prototype counts
    posterior_cnts = prototype_freq(seq_lens, posteriors)
    print("Posterior counts:")
    print(posterior_cnts)

    # select some examples
    eligible_ix = seq_lens >= 500
    print("eligible sequences: %d out of %d" % (np.sum(eligible_ix), eligible_ix.shape[0]))
    ypred = ypred[eligible_ix, :]
    ytrue = ytrue[eligible_ix, :]
    posteriors = posteriors[eligible_ix, :, :]
    seq_mu = seq_mu[eligible_ix]
    seq_lens = seq_lens[eligible_ix]
    all_problem = all_problem[eligible_ix, :]
    all_kc = all_kc[eligible_ix, :]

    chosen_seq_ids = select_examples(seq_mu, np.linspace(0.05, 0.95, 10))

    np.savez(output_path, 
                 seq_lens=seq_lens[chosen_seq_ids],
                 ypred=ypred[chosen_seq_ids,:],
                 ytrue=ytrue[chosen_seq_ids,:],
                 posteriors=posteriors[chosen_seq_ids, :, :],
                 seq_mu=seq_mu[chosen_seq_ids],
                 student_prototypes=state_dicts[split_id]['student_prototypes'].cpu().numpy(),
                 obs_logits_kc=model.obs_logits_kc.cpu().detach().numpy(),
                 obs_logits_problem=model.obs_logits_problem.cpu().detach().numpy(),
                 problem_seqs=all_problem[chosen_seq_ids,:],
                 kc_seqs=all_kc[chosen_seq_ids,:],
                 posterior_cnts=posterior_cnts)
    
def predict(cfg, model, seqs):

    seqs = sorted(seqs, reverse=True, key=lambda s: len(s['kc']))
    seq_lens = np.array([len(s['kc']) for s in seqs])

    max_seq_len = seq_lens[0]
    n_seqs = len(seqs)

    all_pred = np.zeros((n_seqs, max_seq_len))
    all_ytrue = np.zeros((n_seqs, max_seq_len))
    all_problem = np.zeros((n_seqs, max_seq_len))
    all_kc = np.zeros((n_seqs, max_seq_len))
    all_posteriors = np.zeros((len(seqs), cfg['n_student_prototypes'], max_seq_len))
    seq_loss = np.zeros(n_seqs)
    seq_mu = np.zeros(n_seqs)

    model.eval()
    with th.no_grad():

        for offset in range(0, len(seqs), cfg['n_test_batch_seqs']):
            end = offset + cfg['n_test_batch_seqs']
            batch_seqs = seqs[offset:end]
            print("%d - %d out of %d" % (offset, end, len(seqs)))

            # BxT
            
            ytrue = pad_sequence([th.tensor(s['correct']) for s in batch_seqs], batch_first=True, padding_value=0).float().to(cfg['device'])
            
            mask = pad_sequence([th.tensor(s['correct']) for s in batch_seqs],batch_first=True, padding_value=-1).to(cfg['device']) > -1
            
            output, posteriors = model(batch_seqs, ytrue, return_posteriors=True) # BxTx2, BxAxT 
            
            loss = -(ytrue * output[:, :, 1] + (1-ytrue) * output[:, :, 0]) # BxT
            loss = (loss * mask).sum(1) / mask.sum(1) # B 
            seq_mu[offset:end] = ((ytrue*mask).sum(1) / mask.sum(1)).cpu().numpy()

            ypred = output[:, :, 1].cpu().numpy()
            ytrue = ytrue.cpu().int().numpy()

            all_pred[offset:end, :ypred.shape[1]] = ypred
            all_ytrue[offset:end, :ytrue.shape[1]] = ytrue
            all_posteriors[offset:end, :, :posteriors.shape[2]] = posteriors.cpu().numpy()
            seq_loss[offset:end] = loss.cpu().numpy()

            problem_seq = pad_sequence([th.tensor(s['problem']) for s in batch_seqs], batch_first=True, padding_value=0).numpy()
            kc_seq = pad_sequence([th.tensor(s['kc']) for s in batch_seqs], batch_first=True, padding_value=0).numpy()
            all_problem[offset:end, :ypred.shape[1]] = problem_seq
            all_kc[offset:end, :ypred.shape[1]] = kc_seq 

    model.train()
    
    return seq_lens, all_pred, all_ytrue, all_posteriors, seq_loss, seq_mu, all_problem, all_kc

def prototype_freq(seq_lens, posteriors):
    
    posterior_cnts = np.zeros(posteriors.shape[1])
    for i in range(posteriors.shape[0]):
        seq_posteriors = posteriors[i, :, seq_lens[i]-1]
        posterior_id = np.argmax(seq_posteriors)
        posterior_cnts[posterior_id] += 1
    return posterior_cnts

def select_examples(seq_mu, thresholds):
    ix = np.argsort(seq_mu)
    n_seqs = seq_mu.shape[0]

    chosen_seq_ids = []
    for thres in thresholds:
        seq_id = ix[int(n_seqs * thres)]
        print("Threshold %0.2f , Sequence ID: %d, Mu: %0.2f" % (thres, seq_id, seq_mu[seq_id]))
        chosen_seq_ids.append(seq_id)
    
    return chosen_seq_ids

if __name__ == "__main__":
    main()