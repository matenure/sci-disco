'''
Project: CMPSCI696ds IBM2
Author: Yuchen Zeng
'''

import torch
import random
from torch.utils.data import DataLoader

def train(model, scoring_fn, loader, optimizer, device, split_edge):
    model.train()
    scoring_fn.train()
    pos_train = split_edge['train_pos']
    bceloss = torch.nn.BCELoss()

    total_loss = total_examples = 0
    for i, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # GCN
        #h = model(data.x, data.edge_index)

        for i, pos_batch in enumerate(DataLoader(pos_train, 64, collate_fn=lambda x: x)):

            # GCN
            h = model(data.x, data.edge_index)

            # Just do some trivial random sampling for negative combination
            neg_batch = [method_replace(n.copy(), data.x.size(0), pos_train) for n in pos_batch]

            # positive scoring
            pos_out = scoring_fn(h, pos_batch)
            #pos_loss = -torch.log(pos_out).mean() 
            pos_loss = bceloss(pos_out.squeeze(1), torch.ones(pos_out.shape[0], dtype=torch.float).to(device))

            # negative scoing 
            neg_out = scoring_fn(h, neg_batch)
            #neg_loss = -torch.log(1 - neg_out).mean() 
            neg_loss = bceloss(neg_out.squeeze(1), torch.zeros(neg_out.shape[0], dtype=torch.float).to(device))

            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()

            if i//10==0:
                print ("Loss: ",loss.item())
                                
            num_examples = len(pos_train)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        #loss.backward()
        #optimizer.step()

    return total_loss / total_examples


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


