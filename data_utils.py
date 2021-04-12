'''
Project: CMPSCI696ds IBM2
Author: Yuchen Zeng
'''

import os
import numpy as np
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset
from ogb.io.read_graph_pyg import read_graph_pyg
from ogb.utils.torch_util import replace_numpy_with_torchtensor

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root='dataset', transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['edge.csv.gz','node-feat.csv.gz','num-edge-list.csv.gz','num-node-list.csv.gz']

    @property
    def processed_file_names(self):
        return osp.join('geometric_data_processed.pt') 

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        additional_node_files = []
        additional_edge_files = []
        add_inverse_edge = False
        
        # return data list
        # https://github.com/snap-stanford/ogb/blob/1d6dde8080261931bc6ce2491e9149298af1ea98/ogb/io/read_graph_pyg.py#L9
        data = read_graph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, 
            additional_edge_files = additional_edge_files, binary=False)[0]
    
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        data, slices = self.collate([data])
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_edge_split(self, data_dir=None, model_type=None, threshold=None):
        train_dir = os.path.join(data_dir,model_type,threshold,'train.pt')
        valid_dir = os.path.join(data_dir,model_type,threshold,'valid.pt')
        test_dir = os.path.join(data_dir,model_type,threshold,'test.pt')
        train = replace_numpy_with_torchtensor(torch.load(train_dir))
        valid = replace_numpy_with_torchtensor(torch.load(valid_dir))
        test = replace_numpy_with_torchtensor(torch.load(test_dir))
        return {'train': train, 'valid': valid, 'test': test}    

if __name__ == '__main__':
    DS = MyOwnDataset()
    split_edge = DS.get_edge_split(data_dir='data', model_type='link_pred',threshold='threshold=0.50')
    print (DS[0])
