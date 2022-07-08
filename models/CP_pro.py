'''
Project: IBM Research: Scientific Discovery
Author: Yuchen Zeng
Combinatorial Prediction Method:
CP-pro: use prompt-based learning to combine feature
'''

import re
import os
import argparse
import wandb
import random 
import numpy as np
import math
from termcolor import colored
from tqdm import tqdm
from tqdm.contrib import tzip
import torch
from transformers import BertTokenizer
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from transformers import BertModel, AdamW
from sklearn.metrics import precision_recall_curve

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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
# 2. train_pos, train_neg: training combinations, ex. [[0, 1], [1, 2, 3], ...]
# 3. valid_pos, valid_neg_f1, valid_neg_hit: validation combination
# 4. test_pos, test_neg_f1, test_neg_hit: test combination
def get_data_split(data_dir=None):
    train_pos = torch.load(os.path.join(data_dir,'train_pos.pt'))
    train_neg = torch.load(os.path.join(data_dir,'train_neg.pt'))
    valid_pos = torch.load(os.path.join(data_dir,'valid_pos.pt'))
    valid_neg_f1 = torch.load(os.path.join(data_dir, 'valid_neg_f1.pt'))
    valid_neg_hit = torch.load(os.path.join(data_dir, 'valid_neg_hit.pt'))
    test_pos = torch.load(os.path.join(data_dir,'test_pos.pt'))
    test_neg_f1 = torch.load(os.path.join(data_dir,'test_neg_f1.pt'))
    test_neg_hit = torch.load(os.path.join(data_dir,'test_neg_hit.pt'))
    entities = torch.load(os.path.join(data_dir,'entities.pt'))

    return entities, {'train_pos': train_pos, 'train_neg': train_neg,
            'valid_pos': valid_pos, 'valid_neg_f1': valid_neg_f1, 'valid_neg_hit': valid_neg_hit,
            'test_pos': test_pos, 'test_neg_f1':test_neg_f1, 'test_neg_hit': test_neg_hit}
   
def text_preprocessing(text):
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)
    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
 
# Create a function to tokenize a set of texts
# generate inputs and masks for each combinations
def preprocessing_for_bert(entities, combinations):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # store outputs
    input_ids = []
    attention_masks = []
    # For every sentence...
    for comb in tqdm(combinations):
        whole_sent_list = ["This paper use"]
        for c in comb:
            whole_sent_list.append(entities[c])
        whole_sent = " and ".join(whole_sent_list) + "."
        
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(whole_sent),    # Preprocess sentence
            add_special_tokens=True,                # Add `[CLS]` and `[SEP]`
            max_length=32,                          # Max length to truncate/pad
            pad_to_max_length=True,                 # Pad sentence to max length
            #return_tensors='pt',                   # Return PyTorch tensor
            return_attention_mask=True,             # Return attention mask
            truncation=True,
            padding='max_length'
            )
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return {"input_ids": input_ids, "attention_masks": attention_masks}

def tokenization(entities, data_split):
    keys = data_split.keys()
    tokenized_data = {}
    for key in keys:
        tokenized_data[key] = preprocessing_for_bert(entities, data_split[key])
    return tokenized_data
    
# Create the Model class
class Model(nn.Module):
    def __init__(self, freeze_bert):
        super(Model, self).__init__()
        # all entities tokens
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # mlps
        self.mlps = nn.Sequential(
           nn.Linear(768, 128),
           nn.ReLU(),
           nn.Dropout(p=0.5),
           nn.Linear(128, 1),
           nn.Sigmoid()
           )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
    def scoring(self, x):
        h = self.mlps(x)
        return h
    
    def forward(self, input_ids, attention_masks):
        h = self.bert(input_ids=input_ids, attention_mask=attention_masks)[0][:,0,:]
        s = self.scoring(h)
        return s
        
def save_model(model, name, output_path='/shared/scratch/0/v_yuchen_zeng/sci_disco/saved_models/'):
    model_state_dict = model.state_dict()
    checkpoint = {
        'model': model_state_dict,
    }
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    checkpoint_path = os.path.join(output_path, name+'.pt')
    torch.save(checkpoint, checkpoint_path)

def load_model(model, name, output_path='/shared/scratch/0/v_yuchen_zeng/sci_disco/saved_models'):
    checkpoint_path = os.path.join(output_path, name+'.pt')
    if not os.path.exists(checkpoint_path):
        print (f"Model {checkpoint_path} does not exist.")
        return 0
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
class MyDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.atten_masks = attention_mask
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self,index):
        return {"input_ids": self.input_ids[index],
                "attention_masks": self.atten_masks[index]}
    
def batch_train(model, data_split, optimizer, batch_size, config):
    model.train()
    pos_dataset = MyDataset(data_split['train_pos']['input_ids'], data_split['train_pos']['attention_masks'])
    neg_dataset = MyDataset(data_split['train_neg']['input_ids'], data_split['train_neg']['attention_masks'])
    
    bceloss = nn.BCELoss(reduction='mean')
    total_loss, total_batch = 0, 0
    pos_dataloader = DataLoader(pos_dataset, batch_size, shuffle=True)
    neg_dataloader = DataLoader(neg_dataset, 3*batch_size, shuffle=True) # pos comb : neg comb = 1 : 3
    
    for i, (pos_batch, neg_batch) in enumerate(zip(pos_dataloader, neg_dataloader)):
        optimizer.zero_grad()
        pos_out = model(pos_batch['input_ids'].cuda(), pos_batch['attention_masks'].cuda()).reshape(-1)
        neg_out = model(neg_batch['input_ids'].cuda(), neg_batch['attention_masks'].cuda()).reshape(-1)
        pos_loss = bceloss(pos_out, torch.ones(pos_out.shape[0], dtype=torch.float).cuda())
        neg_loss = bceloss(neg_out, torch.zeros(neg_out.shape[0], dtype=torch.float).cuda())
        loss = 3*pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        if i%(len(pos_dataloader)//20)==0:
            print (f"[{i}/{len(pos_dataloader)}] Loss: {loss.item():.4f}")
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
    pos_out, neg_out = [], []
    pos_dataset = MyDataset(pos_data['input_ids'], pos_data['attention_masks'])
    neg_dataset = MyDataset(neg_data['input_ids'], neg_data['attention_masks'])
    pos_dataloader = DataLoader(pos_dataset, 64, shuffle=True)
    neg_dataloader = DataLoader(neg_dataset, 64, shuffle=True)
    for batch in tqdm(pos_dataloader):
        pos_out.append(model(batch['input_ids'].cuda(), batch['attention_masks'].cuda()).reshape(-1).detach().cpu())
    for batch in tqdm(neg_dataloader):
        neg_out.append(model(batch['input_ids'].cuda(), batch['attention_masks'].cuda()).reshape(-1).detach().cpu())
    pos_out, neg_out = torch.cat(pos_out), torch.cat(neg_out)
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
    
    pos_out, neg_out = [], []
    pos_dataset = MyDataset(pos_data['input_ids'], pos_data['attention_masks'])
    neg_dataset = MyDataset(neg_data['input_ids'], neg_data['attention_masks'])
    pos_dataloader = DataLoader(pos_dataset, 64, shuffle=True)
    neg_dataloader = DataLoader(neg_dataset, 64, shuffle=True)
    for batch in tqdm(pos_dataloader):
        pos_out.append(model(batch['input_ids'].cuda(), batch['attention_masks'].cuda()).reshape(-1).detach().cpu())
    for batch in tqdm(neg_dataloader):
        neg_out.append(model(batch['input_ids'].cuda(), batch['attention_masks'].cuda()).reshape(-1).detach().cpu())

    pos_out = torch.cat(pos_out)
    neg_out = torch.cat(neg_out).reshape(-1, 100)
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

    # wandb
    if config.wandb:
        wandb.init(project="sci_disco_emnlp2022", settings=wandb.Settings(start_method="fork"))
        wandb.config.update(config, allow_val_change=True)

    # Dataset
    print(colored('Retrieve dataset', 'red'))
    entities, data_split = get_data_split(os.path.join("/shared/scratch/0/v_yuchen_zeng/sci_disco/Dataset/", config.dataset_type))
    data_split = tokenization(entities, data_split)
    print (data_split.keys())
    
    # Model
    print(colored('Retrieve model', 'red'))
    model = Model(False).to(device)
    # print (model)
    
    # Optimizer
    print (colored('Get optimizer', 'red'))
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)
    print (optimizer)

    # Train
    f1_log, hits_log = [], []
    model_name = "_".join(["pro", config.dataset_type, str(config.learning_rate), str(config.seed)])
    print (model_name)
    best_f1, best_hit, patience = 0, 0, 0
    
    for epoch in range(1, 1 + config.epochs):
        print ("Epoch: ", epoch)
        # train
        loss = batch_train(model, data_split, optimizer, config.batch_size, config)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

        if config.wandb:
            wandb.log({'epochs': epoch, 'avg_loss': loss})
           
        # test F1
        if epoch >= 0 and (epoch-1) % 1 == 0:
            valid_pos_pred, valid_neg_pred = pred_F1(model, data_split, flag='valid')
            test_pos_pred, test_neg_pred = pred_F1(model, data_split, flag='test')
            threshold = get_threshold(valid_pos_pred, valid_neg_pred)
            valid_f1, valid_prec, valid_recall = F1_score(valid_pos_pred, valid_neg_pred, 0.5)
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
    parser.add_argument("--dataset_type", type=str, default="CV_0.5")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--wandb", type=int, default=0) # 0 for False
    config = parser.parse_args()

    main(config)




