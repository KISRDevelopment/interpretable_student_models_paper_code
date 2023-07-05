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

    split = splits[split_id, :]
    test_ix = split == 0
    test_df = df[test_ix]
    test_students = set(test_df['student'])
    test_seqs = [seqs[s] for s in test_students]

    print("# test sequences: %d" % len(test_seqs))
    seq_lens, ypred, ytrue, posteriors, seq_loss, seq_mu = predict(cfg, model, test_seqs)
    
    np.random.seed(64563)

    #chosen_seqs = np.random.choice(len(test_seqs), replace=False, size=10)
    chosen_seqs = seq_lens >= 500
    print("chosen sequences: %d out of %d" % (np.sum(chosen_seqs), chosen_seqs.shape[0]))
    #print(seq_lens[chosen_seqs])
    
    np.savez(output_path, seq_lens=seq_lens[chosen_seqs],
                 ypred=ypred[chosen_seqs,:],
                 ytrue=ytrue[chosen_seqs,:],
                 posteriors=posteriors[chosen_seqs, :, :],
                 seq_loss=seq_loss[chosen_seqs],
                 seq_mu=seq_mu[chosen_seqs],
                 student_prototypes=state_dicts[split_id]['student_prototypes'].cpu().numpy())
    
def predict(cfg, model, seqs):

    seqs = sorted(seqs, reverse=True, key=lambda s: len(s['kc']))
    seq_lens = np.array([len(s['kc']) for s in seqs])

    max_seq_len = seq_lens[0]
    n_seqs = len(seqs)

    all_pred = np.zeros((n_seqs, max_seq_len))
    all_ytrue = np.zeros((n_seqs, max_seq_len))
    all_posteriors = np.zeros((len(seqs), cfg['n_student_prototypes'], max_seq_len))
    seq_loss = np.zeros(n_seqs)
    seq_mu = np.zeros(n_seqs)

    model.eval()
    with th.no_grad():

        for offset in range(0, len(seqs), cfg['n_test_batch_seqs']):
            end = offset + cfg['n_test_batch_seqs']
            batch_seqs = seqs[offset:end]
            
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
            
    model.train()
    
    return seq_lens, all_pred, all_ytrue, all_posteriors, seq_loss, seq_mu

if __name__ == "__main__":
    main()