"""
Authors: Yuchen Zeng
Reference: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
"""

import os
import numpy as np
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset
from ogb.io.read_graph_pyg import read_graph_pyg
from ogb.utils.torch_util import replace_numpy_with_torchtensor

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root='dataset', transform=None, pre_transform=None):
        self.original_root = root
        self.root = self.root = osp.join(root, 'citation-v2')
        super(MyOwnDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        file_names = ['edge','node-feat']
        return [file_name + '.csv.gz' for file_name in file_names]
    
    @property
    def processed_file_names(self):
        return osp.join('geometric_data_processed.pt')    

    def download(self):
        pass
    
    def process(self):
        # Read data into huge `Data` list.
        additional_node_files = ['node_year']
        additional_edge_files = []
        add_inverse_edge = False
        
        # return data list
        # https://github.com/snap-stanford/ogb/blob/1d6dde8080261931bc6ce2491e9149298af1ea98/ogb/io/read_graph_pyg.py#L9
        data = read_graph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=False)[0]
    
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        data, slices = self.collate([data])
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_edge_split(self,split_type=None):
        split_type = 'time'
        path = osp.join(self.root, 'split', split_type)
        
        train = replace_numpy_with_torchtensor(torch.load(osp.join(path, 'train.pt')))
        valid = replace_numpy_with_torchtensor(torch.load(osp.join(path, 'valid.pt')))
        test = replace_numpy_with_torchtensor(torch.load(osp.join(path, 'test.pt')))
        return {'train': train, 'valid': valid, 'test': test}

if __name__ == '__main__':
    pyg_dataset = MyOwnDataset()
    split_edge = pyg_dataset.get_edge_split()
    print (pyg_dataset[0]) # Output: Data(edge_index=[2, 30387995], node_year=[2927963, 1], x=[2927963, 128])
