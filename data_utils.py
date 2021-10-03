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



    def get_edge_split(self, data_dir=None):
        train_pos_dir = os.path.join(data_dir,'co_train.pt')
        valid_pos_dir = os.path.join(data_dir,'co_valid.pt')

        valid_neg_f1_dir = os.path.join(data_dir,'co_valid_neg_f1.pt')
        valid_neg_hit_dir = os.path.join(data_dir,'co_valid_neg_hit.pt')

        test_pos_dir = os.path.join(data_dir,'test_pos.pt')
        test_neg_f1_dir = os.path.join(data_dir,'test_neg_f1.pt')
        test_neg_hit_dir = os.path.join(data_dir,'test_neg_hit.pt')

        train_pos = replace_numpy_with_torchtensor(torch.load(train_pos_dir))
        valid_pos = replace_numpy_with_torchtensor(torch.load(valid_pos_dir))
        valid_neg_f1 = replace_numpy_with_torchtensor(torch.load(valid_neg_f1_dir))
        valid_neg_hit = replace_numpy_with_torchtensor(torch.load(valid_neg_hit_dir))
        test_pos = replace_numpy_with_torchtensor(torch.load(test_pos_dir))
        test_neg_f1 = replace_numpy_with_torchtensor(torch.load(test_neg_f1_dir))
        test_neg_hit = replace_numpy_with_torchtensor(torch.load(test_neg_hit_dir))


        return {'train_pos': train_pos, 'valid_pos': valid_pos, 'valid_neg_f1': valid_neg_f1, 'valid_neg_hit': valid_neg_hit, 'test_pos': test_pos, 'test_neg_f1':test_neg_f1, 'test_neg_hit': test_neg_hit}

def sort_list(data):
    sorted_data = []
    for i,d in enumerate(data):
        d = sorted(d)
        sorted_data.append(d)
    return sorted_data

if __name__ == '__main__':
    DS = MyOwnDataset()
    split_edge = DS.get_edge_split(data_dir='data')
    print (DS[0])
