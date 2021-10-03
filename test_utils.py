'''
Project: CMPSCI696ds IBM2
Author: Yuchen Zeng
'''

import numpy as np
import random
import torch
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader

import json
from sklearn.metrics import precision_recall_curve
import wandb

@torch.no_grad()
def test(model, scoring_fn, data, split_edge, batch_size, device):
    model.eval()
    scoring_fn.eval()
    print('Evaluating full-batch GNN on CPU...')
    data = data.to(device)
    h = model(data.x,data.edge_index)
    
    
    def test_split(split, threshold):
        if split=='train':
            pass
            #pos_data = split_edge['train']
            #neg_data = [method_replace(n.copy(),data.x.size(0),pos_data) for n in pos_data+pos_data] 
        elif split=='valid':
            pos_data = split_edge['valid_pos']
            neg_f1_data = split_edge['valid_neg_f1']
            neg_hit_data = split_edge['valid_neg_hit']
        elif split=='test':
            pos_data = split_edge['test_pos']
            neg_f1_data = split_edge['test_neg_f1']
            neg_hit_data = split_edge['test_neg_hit']

        pos_pred = scoring_fn(h, pos_data)
        neg_pred = scoring_fn(h, neg_f1_data)
        
        # F1 socre
        F1,precision,recall = F1_score(pos_pred, neg_pred, threshold=0.42)
        #wandb.log({"DEV 0.42 F1": F1, "DEV 0.42 Prec": precision, "DEV 0.42 recall": recall})
        F1,precision,recall = F1_score(pos_pred, neg_pred, threshold=0.45)
        #wandb.log({"DEV 0.45 F1": F1, "DEV 0.45 Prec": precision, "DEV 0.45 recall": recall})
        F1,precision,recall = F1_score(pos_pred, neg_pred, threshold=0.5)
        #wandb.log({"DEV 0.5 F1": F1, "DEV 0.5 Prec": precision, "DEV 0.5 recall": recall})
        F1,precision,recall = F1_score(pos_pred, neg_pred, threshold=0.55)
        #wandb.log({"DEV 0.55 F1": F1, "DEV 0.55 Prec": precision, "DEV 0.55 recall": recall})
        F1,precision,recall = F1_score(pos_pred, neg_pred, threshold=0.48)
        #wandb.log({"DEV 0.48 F1": F1, "DEV 0.48 Prec": precision, "DEV 0.48 recall": recall})
        F1,precision,recall = F1_score(pos_pred, neg_pred, threshold=0.52)
        #wandb.log({"DEV 0.52 F1": F1, "DEV 0.52 Prec": precision, "DEV 0.52 recall": recall})

        # Hitsatk
        methods = [i for i in pos_data for iter in range(100)]
        neg_methods = [method_replace(n.copy(),data.x.size(0),pos_data) for n in methods]
        neg_hits_pred = scoring_fn(h, neg_methods).view(-1,100)
        hits = mmr(pos_pred, neg_hits_pred)
            
        return (hits, F1, precision, recall)

    #threshold = get_best_threshold(data, split_edge['valid'], model, scoring_fn)
    threshold = 0.5
    #train_result = test_split('train', threshold)
    valid_result  = test_split('valid', threshold)
    #test_result = test_split('test', threshold)
 
    return valid_result


def method_replace(p, max_node, existing_method):
    # randomly replace half number of concepts
    num_replace = len(p)//2
    replace_index = random.sample(list(range(len(p))), num_replace)
    neg_p = p.copy()
    for ind in replace_index:
        neg_p[ind]=random.randint(0,max_node-1)
    sorted_neg_p = sorted(neg_p)
    if sorted_neg_p in existing_method:
        sorted_neg_p = method_replace(p, max_node, existing_method)
    return sorted_neg_p


# hit at k: num of top k pred value in pos_pred/ total prediction
def hit_at_k(pos_pred, neg_pred, k):
    pred = torch.cat((pos_pred, neg_pred), 0)
    argsort = torch.argsort(pred, dim = 0, descending = True)
    hit_num = torch.sum(argsort[:len(pos_pred)]<k)
    hit_rate = hit_num/(len(pred))
    return hit_rate.item()

# F1 score
# F1 = 2/ (1/recall + 1/precision);
# recall = number of correct predicted positives/ number of positive samples in test
# precision = number of correct predicted positives/ number of predicted positives (including both correct prediction and false positive)
# 0.5 determine the prediction correctness
def F1_score(pos_pred, neg_pred, threshold):
    recall = torch.sum(pos_pred>=threshold)/len(pos_pred)
    precision = torch.sum(pos_pred>=threshold)/(torch.sum(pos_pred>=threshold)+torch.sum(neg_pred>=threshold))
    F1 = 2/(1/recall + 1/precision)
    return (F1.item(),precision.item(),recall.item())

def get_best_threshold(data, pos_data, model, scoring_fn):
    print ("get best threshold")
    model.eval()
    scoring_fn.eval()
    neg_data = [method_replace(n.copy(),data.x.size(0),pos_data) for n in pos_data]

    h = model(data.x,data.edge_index)
    pos_pred = scoring_fn(h, pos_data).reshape(-1)
    neg_pred = scoring_fn(h, neg_data).reshape(-1)

    print (len(pos_pred))
    print (len(neg_pred))

    pred = pos_pred.tolist() + neg_pred.tolist()
    labels = [1 for i in range(len(pos_pred))] + [0 for i in range(len(neg_pred))]
    
    precision, recall, thresholds = precision_recall_curve(labels, pred)
    f1 = 2*(recall*precision)/(recall+precision)
    best_t = thresholds[np.argmax(f1)]
    print (best_t)
    return best_t

def F1_score2(pos_pred, neg_pred):
    TP = torch.sum(pos_pred>=0.5)
    FP = torch.sum(pos_pred<0.5)
    FN = torch.sum(neg_pred>=0.5)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    F1 = (2*precision*recall)/(precision+recall)
    return (F1,precision,recall)

def mmr(pos_pred, neg_pred):
    y_pred = torch.cat([pos_pred.view(-1,1), neg_pred], dim = 1)
    argsort = torch.argsort(y_pred, dim = 1, descending = True)
    ranking_list = torch.nonzero(argsort == 0)
    ranking_list = ranking_list[:, 1] + 1
    hits1_list = (ranking_list <= 1).cpu().numpy()#.to(torch.float)
    hits10_list = (ranking_list <= 10).cpu().numpy()#.to(torch.float)
    hits30_list = (ranking_list <= 30).cpu().numpy()#.to(torch.float)
    hits50_list = (ranking_list <= 50).cpu().numpy()#.to(torch.float)
    length = len(pos_pred)
    hits1 = sum(hits1_list)/length
    hits10 = sum(hits10_list)/length
    hits30 = sum(hits30_list)/length
    hits50 = sum(hits50_list)/length
    #print ("hits at 1, 10, 30, 50:", hits1, hits10, hits30, hits50)
    return [hits1,hits10,hits30,hits50]


def generate_random_comb(num_comb, num_nodes, num_concept):
    pred_comb = []
    for i in range(num_comb):
        comb = random.sample(range(0, num_nodes), num_concept)
        pred_comb.append(sorted(comb))
    return pred_comb

def save_pred(data, comb, model, scoring_fn, device, name):
    model.eval()
    scoring_fn.eval()
    data = data.to(device)
    h = model(data.x, data.edge_index)
    scores = scoring_fn(h, comb)
    
    # get top 100 scores
    values,indices = scores.reshape(-1).topk(100)
    highest_comb, highest_score = [], []
    for ind,value in enumerate(indices):
        highest_comb.append(comb[value])
        highest_score.append(values[ind].detach().cpu().item())

    with open(str(name)+"highest_comb.json", 'w') as comb_f:
        json.dump(highest_comb, comb_f, indent=2)
    with open(str(name)+"highest_score.json", 'w') as score_f:
        json.dump(highest_score, score_f, indent=2)
 

