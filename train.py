'''
Project: CMPSCI696ds IBM2
Author: Yuchen Zeng
'''

import argparse
from models import GCN, LinkPredictor
from data_utils import MyOwnDataset
from train_utils import train
from logger_utils import Logger
#from eval_utils import Evaluator
from test_utils import test

import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_undirected

def main():
    parser = argparse.ArgumentParser(description='Scientific Discovery - IBM2 (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print (device)

    # Load dataset
    dataset = MyOwnDataset()
    split_edge = dataset.get_edge_split(data_dir='data',model_type='link_pred',threshold='threshold=0.50')
    data = dataset[0]
    print (data)
    dataset.edge_index = to_undirected(data.edge_index, data.num_nodes)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    split_edge['eval_train'] = {
        'source_node': split_edge['train']['source_node'][idx],
        'target_node': split_edge['train']['target_node'][idx]
    }

    # create model
    model = GCN(data.x.size(-1), args.hidden_channels, args.hidden_channels,args.num_layers, args.dropout).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,args.num_layers, args.dropout).to(device)
    logger = Logger(args.runs, args)
    print (model)
    print (predictor)

    # train
    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()),lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            # train
            loss = train(model, predictor, loader, optimizer, device)
            #if epoch % 10==0:
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')
            # test
            if epoch >= 0 and epoch % args.eval_steps == 0:
                result = test(model, predictor, data, split_edge, batch_size=args.batch_size, device=device)
                logger.add_result(run, result)                                                           
                train_precision, valid_precision, test_precision = result
                print(f'Run: {run + 1:02d}', 
                      f',\nEpoch: {epoch:02d}', 
                      f',\nLoss: {loss:.4f}', 
                      f',\nTrain_Precision: {train_precision:.4f}',
                      f',\nValid_Precision: {valid_precision:.4f}',
                      f',\nTest_Precision: {test_precision:.4f}',)
        
        #print('ClusterGCN')
        #logger.print_statistics(run)
    #print('ClusterGCN')
    #logger.print_statistics()

if __name__ == "__main__":
    main()
