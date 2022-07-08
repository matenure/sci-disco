'''
Project: IBM Research: Scientific Discovery
Author: Yuchen Zeng
Combinatorial Prediction Method:
LP: use link prediction as the baseline method
(sentence BERT to get node embeddings)
'''

import re
import os
import json
import argparse
import wandb
import random 
import numpy as np
import math
from termcolor import colored
from tqdm import tqdm
from tqdm.contrib import tzip
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
# torch geometircs
from torch_geometric.data import Data
#from torch_geometric.loader import DataLoader
#import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv, Sequential
from sklearn.metrics import precision_recall_curve

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Set seed for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# load dataset
# dataset type: NLP, CV, NLP+CV
# cluster level: 0.1, 0.3, 0.5
# 1. entities: all entities name, ex. ['GNN', 'NN', ...]
# 2. links: training links, ex. [[0, 1], [1, 3], ...]
# 3. valid_pos, valid_neg_f1, valid_neg_hit: validation combination
# 4. test_pos, test_neg_f1, test_neg_hit: test combination
def get_data_split(data_dir, papers):
    # get the all the links
    entities = torch.load(os.path.join(data_dir,'entities.pt'))
    links = []
    mappings = {}
    for i,e in enumerate(entities):
        mappings[e] = i
    for p in tqdm(papers):
        cluster_concepts = p["cluster_concepts"]
        combs = list(combinations(cluster_concepts, 2))
        for i,j in combs:
            links.append([mappings[i], mappings[j]])
            links.append([mappings[j], mappings[i]])
    valid_pos = torch.load(os.path.join(data_dir,'valid_pos.pt'))
    valid_neg_f1 = torch.load(os.path.join(data_dir, 'valid_neg_f1.pt'))
    valid_neg_hit = torch.load(os.path.join(data_dir, 'valid_neg_hit.pt'))
    test_pos = torch.load(os.path.join(data_dir,'test_pos.pt'))
    test_neg_f1 = torch.load(os.path.join(data_dir,'test_neg_f1.pt'))
    test_neg_hit = torch.load(os.path.join(data_dir,'test_neg_hit.pt'))

    return entities, {'links': links,
            'valid_pos': valid_pos, 'valid_neg_f1': valid_neg_f1, 'valid_neg_hit': valid_neg_hit,
            'test_pos': test_pos, 'test_neg_f1':test_neg_f1, 'test_neg_hit': test_neg_hit}

def get_papers(data_dir, config):
    file_list = config.dataset_type.split("_")
    file_list = [file_list[0], "data", file_list[1]]
    file_name = "_".join(file_list)+".json"
    file_dir = os.path.join(data_dir, config.dataset_type, file_name)
    with open(file_dir) as json_file:
        papers = json.load(json_file)
    return papers

def get_node_embedding(entities):
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens', device='cpu')
    embeddings = []
    for e in tqdm(entities):
        sentence = "This paper use " + str(e) + "."
        embeddings.append(list(sbert_model.encode([sentence])[0]))
    embeddings = torch.tensor(embeddings)
    return embeddings
    
# generate whole graph
def generate_graph_data(entities, papers):
    mappings = {}
    for i,e in enumerate(entities):
        mappings[e] = i
    edge_index = []
    for p in tqdm(papers):
        cluster_concepts = p["cluster_concepts"]
        combs = list(combinations(cluster_concepts, 2))
        for i,j in combs:
            edge_index.append([mappings[i], mappings[j]])
            edge_index.append([mappings[j], mappings[i]])
    # get node embeddings
    node_embeddings = get_node_embedding(entities)
    graph_data = Data(x=node_embeddings, edge_index=torch.LongTensor(edge_index).T)
    return graph_data

    
# Create the Model class
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # gcns
        self.gcn1 = GCNConv(768, 768)
        self.gcn2 = GCNConv(768, 768)
        # mlps
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def graph_features(self, graph):
        h = self.gcn1(graph.x, graph.edge_index)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.gcn2(h, graph.edge_index) # (# of node, 768)
        return h
    
    def link_prediction(self, h, source_node, target_node):
        source_h, target_h = h[source_node], h[target_node]
        h = source_h * target_h # combine link (batch, 768), https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/citation2/mlp.py
        h = self.fc1(h)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.fc2(h)
        output = torch.sigmoid(h.reshape(-1))
        return output

    def forward(self, graph, source_node, target_node):
        h = self.graph_features(graph)
        output = self.link_prediction(h, source_node, target_node)
        return output
        
def save_model(model, name, output_path='/shared/scratch/0/v_yuchen_zeng/sci_disco/models/'):
    model_state_dict = model.state_dict()
    checkpoint = {
        'model': model_state_dict,
    }
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    checkpoint_path = os.path.join(output_path, name+'.pt')
    torch.save(checkpoint, checkpoint_path)

def load_model(model, name, output_path='/shared/scratch/0/v_yuchen_zeng/sci_disco/models'):
    checkpoint_path = os.path.join(output_path, name+'.pt')
    if not os.path.exists(checkpoint_path):
        print (f"Model {checkpoint_path} does not exist.")
        return 0
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
def batch_train(model, data_split, optimizer, batch_size, config):
    model.train()
    graph = data_split['graph']
    links = data_split['links']
    total_entity = len(graph.x)
   
    dataloader = DataLoader(links, batch_size, shuffle=True, collate_fn=lambda x: x)
    bceloss = nn.BCELoss(reduction='mean')
    total_loss, total_batch = 0, 0
    
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch_links = torch.tensor(batch).T # (2, k)
        source_node = batch_links[0]
        target_node = batch_links[1]
        randn_source = source_node.repeat(1, 3)
        randn_target = torch.randint(0, total_entity, (len(randn_source),))

        pos_out = model(graph.cuda(), source_node.cuda(), target_node.cuda())
        neg_out = model(graph.cuda(), randn_source.cuda(), randn_target.cuda())
        pos_loss = bceloss(pos_out, torch.ones(pos_out.shape[0], dtype=torch.float).cuda())
        neg_loss = bceloss(neg_out, torch.zeros(neg_out.shape[0], dtype=torch.float).cuda())
        loss = 3*pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        
        if i%(len(dataloader)//20)==0:
            print (f"[{i}/{len(dataloader)}] Loss: {loss.item():.4f}")
            if config.wandb:
                wandb.log({'loss': loss.item()})
        total_loss += loss.item()
        total_batch += 1
    return total_loss/total_batch
    
def pred_F1(model, data_split, flag):
    model.eval()
    if flag == 'train':
        pos_data = data_split['train_pos']
        neg_data = data_split['train_neg']
    elif flag == 'valid':
        pos_data = data_split['valid_pos']
        neg_data = data_split['valid_neg_f1']
    elif flag == 'test':
        pos_data = data_split['test_pos']
        neg_data = data_split['test_neg_f1']
        
    graph_h = model.graph_features(data_split['graph'])
        
    pos_out, neg_out = [], []
    for comb in tqdm(pos_data):
        pairwise_combs = list(combinations(comb, 2))
        combs = torch.tensor(pairwise_combs).T
        source_node, target_node = combs[0], combs[1]
        out = model.link_prediction(graph_h, source_node, target_node)
        pos_out.append(1 if torch.sum(out>0.5).item() == len(out) else 0)
        
    for comb in tqdm(neg_data):
        pairwise_combs = list(combinations(comb, 2))
        combs = torch.tensor(pairwise_combs).T
        source_node, target_node = combs[0], combs[1]
        out = model.link_prediction(graph_h, source_node, target_node)
        neg_out.append(1 if torch.sum(out>0.5).item() == len(out) else 0)
    
    pos_out = torch.tensor(pos_out)
    neg_out = torch.tensor(neg_out)
    return pos_out, neg_out

def get_threshold(pos_pred, neg_pred):
    y_true_pos = torch.ones(len(pos_pred))
    y_true_neg = torch.zeros(len(neg_pred))
    y_true = torch.cat([y_true_pos, y_true_neg])
    y_scores = torch.cat([pos_pred, neg_pred])
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    F1 = [round(2*(p*r)/(p+r+1e-8), 5) for p, r in zip(precision, recall)]
    ind = F1.index(max(F1))
    return thresholds[ind]

def F1_score(pos_pred, neg_pred, threshold):
    TP = torch.sum(pos_pred>=threshold)
    FN = torch.sum(pos_pred<threshold)
    FP = torch.sum(neg_pred>=threshold)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision*recall) / (precision+recall)
    return (F1.item(),precision.item(),recall.item())
    
def test_Hit(model, data_split, flag, device):
    model.eval()
    if flag == 'valid':
        pos_data = data_split['valid_pos']
        neg_data = data_split['valid_neg_hit']
    elif flag == 'test':
        pos_data = data_split['test_pos']
        neg_data = data_split['test_neg_hit']
        
    graph_h = model.graph_features(data_split['graph'])
    pos_out, neg_out = [], []
    for comb in tqdm(pos_data):
        pairwise_combs = list(combinations(comb, 2))
        combs = torch.tensor(pairwise_combs).T
        source_node, target_node = combs[0], combs[1]
        out = model.link_prediction(graph_h, source_node, target_node)
        pos_out.append(torch.mean(out))
        
    for comb in tqdm(neg_data):
        pairwise_combs = list(combinations(comb, 2))
        combs = torch.tensor(pairwise_combs).T
        source_node, target_node = combs[0], combs[1]
        out = model.link_prediction(graph_h, source_node, target_node)
        neg_out.append(torch.mean(out))
   
    pos_out = torch.tensor(pos_out)
    neg_out = torch.tensor(neg_out).reshape(-1, 100)
    hits10, hits20, hits30 = mmr(pos_out, neg_out)
    return hits10, hits20, hits30
    
def mmr(pos_pred, neg_pred):
    y_pred = torch.cat([pos_pred.view(-1,1), neg_pred], dim = 1)
    argsort = torch.argsort(y_pred, dim = 1, descending = True)
    ranking_list = torch.nonzero(argsort == 0)
    ranking_list = ranking_list[:, 1] + 1
    hits1_list = (ranking_list <= 1).cpu().numpy()#.to(torch.float)
    hits10_list = (ranking_list <= 10).cpu().numpy()#.to(torch.float)
    hits20_list = (ranking_list <= 20).cpu().numpy()#.to(torch.float)
    hits30_list = (ranking_list <= 30).cpu().numpy()#.to(torch.float)
    length = len(pos_pred)
    hits1 = sum(hits1_list)/length
    hits10 = sum(hits10_list)/length
    hits20 = sum(hits20_list)/length
    hits30 = sum(hits30_list)/length
    #print ("hits at 1, 10, 30, 50:", hits1, hits10, hits30, hits50)
    return hits10, hits20, hits30


def main(config):

    print (colored("Program Start", 'red'))
    # args
    print(colored(config, 'red'))
    set_seed(config.seed)
    device = f'cuda:{config.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print (device)

    # wandb
    if config.wandb:
        wandb.init(project="sci_disco_emnlp2022", settings=wandb.Settings(start_method="fork"))
        wandb.config.update(config, allow_val_change=True)
 
    # Dataset
    print(colored('Retrieve dataset', 'red'))
    papers = get_papers("/shared/scratch/0/v_yuchen_zeng/sci_disco/Dataset/", config)
    entities, data_split = get_data_split(os.path.join("/shared/scratch/0/v_yuchen_zeng/sci_disco/Dataset/", config.dataset_type), papers)
    graph_data = generate_graph_data(entities, papers)
    print (graph_data)
    data_split['graph'] = graph_data
    print (data_split.keys())

    # Model
    print(colored('Retrieve model', 'red'))
    model = Model().to(device)
    print (model)
    
    # Optimizer
    print (colored('Get optimizer', 'red'))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-8)
    print (optimizer)

    # Train
    f1_log, hits_log = [], []
    model_name = "_".join(["lp", config.dataset_type, str(config.learning_rate), str(config.seed)])
    print (model_name)
    best_f1, best_hit, patience = 0, 0, 0
    
    for epoch in range(1, 1 + config.epochs):
        print ("Epoch: ", epoch)
        # train
        loss = batch_train(model, data_split, optimizer, config.batch_size, config)
        print(f'Epoch: {epoch:02d}, avg_loss: {loss:.4f}')

        if config.wandb:
            wandb.log({'epochs': epoch, 'avg_loss': loss})
    
        # test F1
        if epoch >= 0 and (epoch-1) % 1 == 0:
            #train_f1, train_prec, train_recall = test_F1(model, data_split, flag='train', device=device)
            #valid_f1, valid_prec, valid_recall = test_F1(model, data_split, flag='valid', device=device)
            valid_pos_pred, valid_neg_pred = pred_F1(model, data_split, flag='valid')
            test_pos_pred, test_neg_pred = pred_F1(model, data_split, flag='test') 
            threshold = get_threshold(valid_pos_pred, valid_neg_pred)
            valid_f1, valid_prec, valid_recall = F1_score(valid_pos_pred, valid_neg_pred, threshold)
            test_f1, test_prec, test_recall = F1_score(test_pos_pred, test_neg_pred, 0.5)

            print(f'Epoch: {epoch:02d}',
                    f',\nThreshold: {threshold:.4f}'
                    #f',\nTrain_F1: {train_f1}, Prec: {train_prec}, Recall: {train_recall}',
                    f',\nValid_F1: {valid_f1}, Prec: {valid_prec}, Recall: {valid_recall}',
                    f',\nTest_F1: {test_f1}, Prec: {test_prec}, Recall: {test_recall}'
                    )

            if config.wandb:
                #wandb.log({"Train F1": train_f1, "Train Prec": train_prec, "Train Recall": train_recall})
                wandb.log({"Valid F1": valid_f1, "Valid Prec": valid_prec, "Valid Recall": valid_recall})
                wandb.log({"Test F1": test_f1, "Test Prec": test_prec, "Test Recall": test_recall})
            
            if valid_f1 > best_f1:
                print ("Find new F1 best!")
                best_f1 = valid_f1
                patience = 0
                save_model(model, name=model_name)
            else:
                patience += 1

            # Early stop
            if patience > config.patience:
                print ("Early Stopping")
                break
            
        # test hit@k
        if epoch >= 0 and (epoch-1) % 2 == 0:
            valid_hit10, valid_hit20, valid_hit30 = test_Hit(model, data_split, flag='valid', device=device)
            # test_hit10, test_hit20, test_hit30 = test_Hit(model, data_split, flag='test', device=device)
            
            print(f'Valid_Hits@10: {valid_hit10}, Valid_Hits@20: {valid_hit20}, Valid_Hits@30: {valid_hit30}',
                  #f',\nTest_Hits@10: {test_hit10}, Test_Hits@20: {test_hit20}, Test_Hits@30: {test_hit30}'
                  )

            if config.wandb:
                wandb.log({"Valid hits@10": valid_hit10, "Valid hits@20": valid_hit20, "Valid hits@30": valid_hit30})
                # wandb.log({"Test hits@10": test_hit10, "Test hits@20": test_hit20, "Test hits@30": test_hit30})
                
            if valid_hit10 > best_hit:
                print ("Find new hit@k best!")
                best_hit = valid_hit10

    # load best model
    print(colored('Load Best Model', 'red'))
    load_model(model, name=model_name)
    valid_pos_pred, valid_neg_pred = pred_F1(model, data_split, flag='valid')
    test_pos_pred, test_neg_pred = pred_F1(model, data_split, flag='test')
    threshold = get_threshold(valid_pos_pred, valid_neg_pred)
    test_f1, test_prec, test_recall = F1_score(test_pos_pred, test_neg_pred, threshold)
    test_hit10, test_hit20, test_hit30 = test_Hit(model, data_split, flag='test', device=device)
    print (f'Test_f1: {test_f1}, Test_prec: {test_prec}, Test_recall:{test_recall}')
    print (f'Test_hit@10: {test_hit10}, Test_hit@20: {test_hit20}, Test_hit@30:{test_hit30}')

    if config.wandb:
        wandb.log({"Final Test F1": test_f1, "Final Test Prec": test_prec, "Final Test Recall": test_recall})
        wandb.log({"Final Test hits@10": test_hit10, "Final Test hits@20": test_hit20, "Final Test hits@30": test_hit30})
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sci_disco_EMNLP2022")
    parser.add_argument("--dataset_type", type=str, default="NLP_0.1")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--wandb", type=int, default=0) # 0 for False
    config = parser.parse_args()

    main(config)



