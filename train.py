import argparse

from models import GCN, LinkPredictor
from data_utils import MyOwnDataset
from train_utils import train

import torch
from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.utils import to_undirected

def main():
    parser = argparse.ArgumentParser(description='Scientific Discovery - IBM2 (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_partitions', type=int, default=15000)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--runs', type=int, default=2)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print (device)

    # Load dataset
    dataset = MyOwnDataset()
    split_edge = dataset.get_edge_split()
    data = dataset[0]
    print (data)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    cluster_data = ClusterData(data, num_parts=args.num_partitions,recursive=False, save_dir=dataset.processed_dir)
        
    loader = ClusterLoader(cluster_data, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers)

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    split_edge['eval_train'] = {
        'source_node': split_edge['train']['source_node'][idx],
        'target_node': split_edge['train']['target_node'][idx],
        'target_node_neg': split_edge['valid']['target_node_neg'],
    }

    # create model
    model = GCN(data.x.size(-1), args.hidden_channels, args.hidden_channels,args.num_layers, args.dropout).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,args.num_layers, args.dropout).to(device)
    # evaluator = Evaluator(name='ogbl-citation2')
    # logger = Logger(args.runs, args)
    print (model)
    print (predictor)

    # train
    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()),lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, loader, optimizer, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')
            '''
            if epoch > 49 and epoch % args.eval_steps == 0:
                result = test(model, predictor, data, split_edge, evaluator,batch_size=64 * 1024, device=device)
                logger.add_result(run, result)
                                                               
                train_mrr, valid_mrr, test_mrr = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {train_mrr:.4f}, '
                      f'Valid: {valid_mrr:.4f}, '
                      f'Test: {test_mrr:.4f}')
            '''

        print('ClusterGCN')
    # logger.print_statistics(run)
    print('ClusterGCN')
    # logger.print_statistics()

if __name__ == "__main__":
    main()
