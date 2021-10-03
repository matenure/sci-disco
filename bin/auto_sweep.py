# input: model_type, dim
# output: a list of "wandb sweep" command


import os
import sys
import uuid
import copy
import wandb
import time
import json
import argparse
import subprocess
from pathlib import Path


def config_generation():

    sweep_config = {
        "program": "train.py",
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "${args}",
            "--batch_size=128",
            "--epochs=300",
            "--patience=30",
            "--wandb=True",
        ],
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "DEV F1"},
        "parameters": {
            "learning_rate": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
            "num_layers": {"values": [2, 3, 4, 5]},
            "hidden_channels": {"values": [64, 128, 256]},
        },
    }

    return sweep_config


def main(config):
    sweep_config = config_generation()
    sweep_id = wandb.sweep(sweep_config, project="sci_disco")
    os.system(f"sh bin/launch_train_sweep.sh yuchenzeng/sci_disco/{sweep_id} {config.partition} {config.max_run} {config.num_machine} {config.memory_per_run}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Project: Scientific Discovery IBM2")
    parser.add_argument("--partition", type=str, default="1080ti-short")
    parser.add_argument("--username", type=str, default="yuchenzeng")
    parser.add_argument("--max_run", type=int, default=50)
    parser.add_argument("--num_machine", type=int, default=8)
    parser.add_argument("--memory_per_run", type=int, default=10000)

    
    config = parser.parse_args()

    main(config)

