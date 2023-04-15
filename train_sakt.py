import argparse
import pandas as pd
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score

import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

from model_sakt import SAKT
#from utils import *
import sys
import json 
import numpy as np 
import torch 


def get_data(df, max_length, train_split=0.8, randomize=True):
    """Extract sequences from dataframe.

    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        max_length (int): maximum length of a sequence chunk
        train_split (float): proportion of data to use for training
    """
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]

    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i + 1))[:-1] for i in item_ids]
    skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s + 1))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

    def chunk(list):
        if list[0] is None:
            return list
        list = [torch.split(elem, max_length) for elem in list]
        return [elem for sublist in list for elem in sublist]

    # Chunk sequences
    lists = (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels)
    chunked_lists = [chunk(l) for l in lists]

    data = list(zip(*chunked_lists))
    if randomize:
        shuffle(data)

    # Train-test split across users
    train_size = int(train_split * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    return train_data, val_data


def prepare_batches(data, batch_size, randomize=True):
    """Prepare batches grouping padded sequences.

    Arguments:
        data (list of lists of torch Tensor): output by get_data
        batch_size (int): number of sequences per batch
    Output:
        batches (list of lists of torch Tensor)
    """
    if randomize:
        shuffle(data)
    batches = []

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        seq_lists = list(zip(*batch))
        inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                          if (seqs[0] is not None) else None for seqs in seq_lists[:-1]]
        labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
        batches.append([*inputs_and_ids, labels])

    return batches


def compute_loss(preds, labels, criterion):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    return criterion(preds, labels)


def train(train_data, val_data, model, optimizer, num_epochs, batch_size, grad_clip, patience):
    """Train SAKT model.

    Arguments:
        train_data (list of tuples of torch Tensor)
        val_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        saver: wrapper for torch saving
        num_epochs (int): number of epochs to train for
        batch_size (int)
        grad_clip (float): max norm of the gradients
    """
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc_roc = 0.5
    waited = 0

    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(val_data, batch_size)

        # for each batch
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in train_batches:
            item_inputs = item_inputs.cuda()
            skill_inputs = skill_inputs.cuda()
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            skill_ids = skill_ids.cuda()

            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            loss = compute_loss(preds, labels.cuda(), criterion)
            preds = torch.sigmoid(preds).detach().cpu()
            
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in val_batches:
            item_inputs = item_inputs.cuda()
            skill_inputs = skill_inputs.cuda()
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            skill_ids = skill_ids.cuda()
            with torch.no_grad():
                preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
                preds = torch.sigmoid(preds).cpu()
                all_preds.append(preds[:,:,0].flatten())
                
                all_labels.append(labels.cpu().flatten())
                
        all_preds = torch.hstack(all_preds)
        all_labels = torch.hstack(all_labels)
        val_auc = roc_auc_score(all_labels[all_labels >= 0], all_preds[all_labels >= 0])

        new_best = False 
        if val_auc > best_auc_roc:
            best_auc_roc = val_auc
            waited = 0
            new_best = True 
            state_dict = model.state_dict()
            problem_embddings = state_dict['item_embeds.weight'].cpu()
        else:
            waited += 1

        print("%4d Validation AUC-ROC: %4.2f %s" % (epoch, val_auc, '***' if new_best else ''))
        model.train()

        if waited == patience:
            break

    return problem_embddings[1:, :] # zero is reserved

def main(cfg_path, dataset_name, output_path):
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    full_df = pd.read_csv("data/datasets/%s.csv" % dataset_name)
    full_df['skill'] = 0
    full_df.rename(columns={
        "student" : "user_id",
        "skill" : "skill_id",
        "problem" : "item_id"
    }, inplace=True)
    splits = np.load("data/splits/%s.npy" % dataset_name)
    
    all_embdeddings = []
    for s in range(splits.shape[0]):
        split = splits[0, :]
        train_ix = split == 2
        valid_ix = split == 1
        test_ix = split == 0

        train_df = full_df[train_ix | valid_ix]
        
        train_data, val_data = get_data(train_df, cfg['max_length'])

        num_items = int(full_df["item_id"].max() + 1)
        num_skills = int(full_df["skill_id"].max() + 1)

        model = SAKT(num_items, 
                    num_skills, 
                    cfg['embed_size'], 
                    cfg['num_attn_layers'], 
                    cfg['num_heads'],
                    cfg['encode_pos'], 
                    cfg['max_pos'], 
                    cfg['drop_prob']).cuda()
        optimizer = Adam(model.parameters(), lr=cfg['lr'])
        
        problem_embds = train(train_data, val_data, model, optimizer, cfg['num_epochs'],
                    cfg['batch_size'], cfg['grad_clip'], cfg['patience'])
        
        all_embdeddings.append(problem_embds.numpy())

    all_embdeddings = np.array(all_embdeddings)
    print(all_embdeddings.shape)

    embd_output_path = output_path.replace(".csv", ".embeddings.npy")
    np.save(embd_output_path, all_embdeddings)

if __name__ == "__main__":
    cfg_path = sys.argv[1]
    dataset_name = sys.argv[2]
    output_path = sys.argv[3]

    main(cfg_path, dataset_name, output_path)