'''
Project: CMPSCI696ds IBM2
'''

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
from models import GCN, LinkPredictor, ScoringFunction, load_model, save_model
from data_utils import MyOwnDataset
from train_utils import train
from logger_utils import Logger
#from eval_utils import Evaluator
from test_utils import test, generate_random_comb, save_pred
from plot import *

import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_undirected

import click
import wandb

@click.command(context_settings=dict(show_default=True),)
@click.option(
    "--device",
    type=int,
    default=0,
    help="cpu/gpu",)
@click.option(
    "--num_workers",
    type=int,
    default=0,
    help="number of workers",)
@click.option(
    "--num_layers",
    type=int,
    default=5,
    help="number of fully connected layer",)
@click.option(
    "--hidden_channels",
    type=int,
    default=128,
    help="hidden channel in the fully connected layer",)
@click.option(
    "--dropout",
    type=float,
    default=0.5,
    help="drop out rate in the fully connected layer",)
@click.option(
    "--batch_size",
    type=int,
    default=128,
    help="train batch size",)
@click.option(
    "--learning_rate",
    type=float,
    default=1e-5,
    help="learning rate",)
@click.option(
    '--weight_decay',
    type=float,
    default=1e-5,
    help="loss regularization",)
@click.option(
    "--epochs",
    type=int,
    default=200,
    help="training epochs",)
@click.option(
    "--eval_steps",
    type=int,
    default=1,
    help="evaluation step",)
@click.option(
    "--runs",
    type=int,
    default=1,
    help="total number of runs",)
@click.option(
    "--patience",
    type=int,
    default=500,
    help="patience for early stop",)
@click.option(
    "--wandb",
    default=True,
    help="enable/disable wandb",)

def main(**config):
    
    print(config)

    if config["wandb"]:
        wandb.init(project="sci_disco", settings=wandb.Settings(start_method="fork"))
        wandb.config.update(config, allow_val_change=True)

    device = f'cuda:{config["device"]}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print (device)

    # Load dataset
    dataset = MyOwnDataset()
    split_edge = dataset.get_edge_split(data_dir='data')
    data = dataset[0]
    print (data)
    dataset.edge_index = to_undirected(data.edge_index, data.num_nodes)

    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

    # create model
    model = GCN(data.x.size(-1), config["hidden_channels"], config["hidden_channels"], config["num_layers"], config["dropout"]).to(device)
    #predictor = LinkPredictor(config["hidden_channels"], config["hidden_channels"], 1, config["num_layers"], config["dropout"]).to(device)
    scoring_fn = ScoringFunction(config["hidden_channels"], data.num_nodes).to(device)
    print (model)
    print (scoring_fn)
    #print (predictor)

    # train
    f1_log, hits_log = [], []
    for run in range(config["runs"]):

        best_f1 = 0
        patience = 0
        model.reset_parameters()
        #predictor.reset_parameters()
        scoring_fn.reset_parameters()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(scoring_fn.parameters()), lr=config["learning_rate"], weight_decay=config['weight_decay'])
    
        
        for epoch in range(1, 1 + config["epochs"]):
            # train
            loss = train(model, scoring_fn, loader, optimizer, device, split_edge)
            #if epoch % 10==0:
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')

            print ("c for observation:", scoring_fn.c)
 
            if config["wandb"]:
               wandb.log({'epochs': epoch, 'loss': loss})

            
            # test
            if epoch >= 0 and epoch % config["eval_steps"] == 0:
                result = test(model, scoring_fn, data, split_edge, batch_size=config["batch_size"], device=device)
                valid_result = result
                valid_hit,valid_f1,valid_prec,valid_recall = valid_result[0],valid_result[1],valid_result[2],valid_result[3]
                #test_hit,test_f1,test_prec,test_recall = test_result[0],test_result[1],test_result[2],test_result[3]
               
                print(f'Run: {run + 1:02d}',
                        f',\nEpoch: {epoch:02d}',
                        f',\nLoss: {loss:.4f}',
                        f',\nValid_F1: {valid_f1}',
                        f',\nValid Hits@1:{valid_hit[0]}, Hits@10:{valid_hit[1]}, Hits@30:{valid_hit[2]}, Hits@50:{valid_hit[3]}')

                if config["wandb"]:
                        wandb.log({"DEV F1": valid_f1, "DEV Prec": valid_prec, "DEV recall": valid_recall, "DEV hits@10": valid_hit[1], "DEV hits@30": valid_hit[2], "DEV hits@50":valid_hit[3]})

                if valid_f1 > best_f1:
                    print ("Find new best!")
                    best_f1 = valid_f1
                    patience = 0
                    save_model(model, scoring_fn, optimizer, name="model_"+str(config["learning_rate"])+str(config["num_layers"])+str(config["hidden_channels"]))
                    #if config["wandb"]:
                    #    wandb.log({"TRAIN F1": train_f1, "DEV F1": valid_f1})
                else:
                    patience += 1

                # Early stop
                if patience > config["patience"]:
                    print ("Early Stopping")
                    break
        

        # load best model
        print ("Load best model ...")
        load_model(model, scoring_fn, name="model_"+str(config["learning_rate"])+str(config["num_layers"])+str(config["hidden_channels"]))

        result = test(model, scoring_fn, data, split_edge, batch_size=config["batch_size"], device=device)
        test_result = result
        hits, test_f1,test_prec,test_recall = test_result[0],test_result[1],test_result[2], test_result[3]
        print ("TEST hits@k:", hits)
        print ("TEST F1: ",test_f1)
        if config["wandb"]:
            wandb.log({"TEST F1": test_f1, "HotsAt1": hits[0], "HitsAt10": hits[1], "HitsAt30": hits[2], "HitsAt50": hits[3]})
            wandb.finish()
        f1_log.append(test_f1)
        hits_log.append(hits)
 

    # get prediction
    random_comb = generate_random_comb(10000, data.num_nodes, 2)
    save_pred(data, random_comb, model, scoring_fn, device, "2")

    random_comb = generate_random_comb(10000, data.num_nodes, 3)
    save_pred(data, random_comb, model, scoring_fn, device, "3")

if __name__ == "__main__":
    main()
