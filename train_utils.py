'''
Project: CMPSCI696ds IBM2
Author: Yuchen Zeng
'''

import torch

def train(model, predictor, loader, optimizer, device):
    model.train()
    
    total_loss = total_examples = 0
    for i, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        h = model(data.x, data.edge_index)
        
        src, dst = data.edge_index
        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        
        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, data.x.size(0), src.size(), dtype=torch.long, device=device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
                                
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
                                
        num_examples = src.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples
