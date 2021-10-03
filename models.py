import os
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()
        
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        
    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


'''
Our scoring function
input: a list of node index
output: combination of paper probability
'''
class ScoringFunction(torch.nn.Module):
    def __init__(self, feature_dimension, num_nodes):
        super(ScoringFunction, self).__init__()
        self.b = torch.nn.Parameter(torch.empty(feature_dimension, 1))
        self.W = torch.nn.Parameter(torch.empty(feature_dimension, feature_dimension))
        self.c = torch.nn.Parameter(torch.empty(1))
        self.reset_parameters()
    def scoring(self, comb, h, b, W, c):
        # concat(A_i, .. A_j) * b
        concat_matrix = torch.mean(torch.index_select(h,0,torch.tensor(comb).cuda()), 0)
        first_term_value =  torch.matmul(concat_matrix, b)
        # \sum_ij A_i.T * W_ij * A_j
        pairwise_comb = torch.combinations(torch.tensor(comb).cuda())
        second_term_value = torch.stack([h[i] @ W @ h[j] for i,j in pairwise_comb]).sum(0)
        return first_term_value+second_term_value + self.c
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.b, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        torch.nn.init.constant_(self.c, 0)
    def forward(self, h, x):
        # x is the list of the combination of concepts, h is the feature vector from graph
        scores = torch.stack([self.scoring(p, h, self.b, self.W, self.c) for p in x]).reshape(-1,1)

        return torch.sigmoid(scores)


class GCNInference(torch.nn.Module):
    def __init__(self, weights):
        super(GCNInference, self).__init__()
        self.weights = weights
    
    def forward(self, x, adj):
        for i, (weight, bias) in enumerate(self.weights):
            x = adj @ x @ weight + bias
            x = np.clip(x, 0, None) if i < len(self.weights) - 1 else x
        return x

def save_model(model, scoring_fn, opt, name, output_path='result/'):
    model_state_dict = model.state_dict()
    scoring_state_dict = scoring_fn.state_dict()
    checkpoint = {
        'model': model_state_dict,
        'scoring': scoring_state_dict,
        'opt': opt,
    }
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    checkpoint_path = os.path.join(output_path, name+'.pt')
    torch.save(checkpoint, checkpoint_path)

def load_model(model, scoring_fn, name, output_path='result/'):
    checkpoint_path = os.path.join(output_path, name+'.pt')
    if not os.path.exists(checkpoint_path):
        print (f"Model {checkpoint_path} does not exist.")
        return 0
    checkpoint = torch.load(checkpoint_path)
    opt = vars(checkpoint['opt'])
    model.load_state_dict(checkpoint['model'])
    scoring_fn.load_state_dict(checkpoint['scoring'])


