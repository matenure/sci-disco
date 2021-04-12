'''
Project: CMPSCI696ds IBM2
Author: Yuchen Zeng
'''

import numpy as np
import torch
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader

@torch.no_grad()
def test(model, predictor, data, split_edge, batch_size, device):
    model.eval()
    predictor.eval()
    print('Evaluating full-batch GNN on CPU...')
    h = model(data.x,data.edge_index)
    
    def test_split(split):
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        
        topk_links = get_topk_links(source, target)
        unique_links = torch.unique(torch.stack((source,target),0),dim=1)
        source, target = unique_links
     
        pos_preds = []
        for perm in DataLoader(range(len(source)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)
        
        # topk prediction result
        topk_pred = torch.topk(pos_pred, 100)
        topk_link_pred = torch.transpose(unique_links, 0, 1)[topk_pred.indices]
        UOI = sum([link in topk_links for link in topk_link_pred.tolist()])
        precision = UOI/100
        return (precision)

    train_test = test_split('train')
    valid_test = test_split('valid')
    test_test = test_split('test')
    predictor.train()
    return train_test, valid_test, test_test

def get_topk_links(sources, targets):
    import collections
    links = [sources.tolist(), targets.tolist()]
    counter = collections.Counter(tuple(item) for item in np.array(links).T.tolist())
    topk = counter.most_common(100)
    topk_links = [[s,t] for (s,t),c in topk]
    return topk_links


