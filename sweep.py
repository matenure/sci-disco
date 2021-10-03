#!/usr/bin/env python
import wandb
from train import main

if __name__ == '__main__':
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'learning_rate': {
                'values': [1e-2, 1e-3, 1e-4, 1e-5]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=main)
