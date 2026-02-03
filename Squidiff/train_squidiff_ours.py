# -*- coding: utf-8 -*-
# @Author: SiyuHe
# @Last Modified by:   SiyuHe
# @Last Modified time: 2025-02-19

import io
import os
import socket

import torch as th
import torch.distributed as dist
import argparse
from datetime import datetime
from Squidiff import dist_util,logger

from Squidiff.scrna_datasets import prepared_state_data
from Squidiff.resample import create_named_schedule_sampler
from Squidiff.script_util import (
    our_model_and_diffusion_defaults,
    model_and_diffusion_defaults,
    create_our_film_model_and_diffusion,
    create_our_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from Squidiff.train_util import TrainLoop,plot_loss

GPUS_PER_NODE = 4  # Set this to the actual number of GPUs per node

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across ranks.
    """
    chunk_size = 2 ** 30  # Size limit for data chunks
    if dist.get_rank() == 0:
        with open(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        dist.broadcast(th.tensor(num_chunks), 0)
        for i in range(0, len(data), chunk_size):
            dist.broadcast(th.tensor(data[i: i + chunk_size]), 0)
    else:
        num_chunks = dist.broadcast(th.tensor(0), 0).item()
        data = bytes()
        for _ in range(num_chunks):
            chunk = th.zeros(chunk_size, dtype=th.uint8)
            dist.broadcast(chunk, 0)
            data += bytes(chunk.numpy())

    return th.load(io.BytesIO(data), **kwargs)

def run_training(args):
    dist_util.setup_dist()
    logger.configure(dir=args['logger_path'])
    logger.log("creating data loader...")
    data, state_config = prepared_state_data(
        toml_config = args['toml_config'],
        batch_size = args['batch_size'],
        model_path = args['resume_checkpoint'],
        use_hvg = args['use_hvg'],
        use_vae = args['use_vae'],
        use_control_set = args['use_control_set'],
        control_label = args['control_label'],
        control_k = args['control_k'],
    )
    args['state_dataset_config'] = state_config
    logger.log("*********creating model and diffusion**********")
    if args['film']:
        model, diffusion = create_our_film_model_and_diffusion(
            **args_to_dict(args, our_model_and_diffusion_defaults().keys())
        )
    else:
        model, diffusion = create_our_model_and_diffusion(
            **args_to_dict(args, our_model_and_diffusion_defaults().keys())
        )
    
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args['schedule_sampler'], diffusion)

    #logger.log(f'with gpu {dist_util.dev()}')
    start_time = datetime.now()
    logger.log(f'**********training started at {start_time} **********')
    train_ = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args['batch_size'],
        microbatch=args['microbatch'],
        lr=args['lr'],
        ema_rate=args['ema_rate'],
        log_interval=args['log_interval'],
        save_interval=args['save_interval'],
        resume_checkpoint=args['resume_checkpoint'],
        use_fp16=args['use_fp16'],
        fp16_scale_growth=args['fp16_scale_growth'],
        schedule_sampler=schedule_sampler,
        weight_decay=args['weight_decay'],
        lr_anneal_steps=args['lr_anneal_steps'],
        use_drug_structure= False,
        state_dataset=True,
        comb_num=1
    )
    train_.run_loop()
    
    end_time = datetime.now()

    during_time = (end_time-start_time).seconds/60

    logger.log(f'start time: {start_time} end_time: {end_time} time:{during_time} min')
    
    return train_.loss_list


def parse_args():
    """Parse command-line arguments and update with default values."""
    # Define default arguments
    default_args = {}
    default_args.update(our_model_and_diffusion_defaults())
    updated_args = {
        'toml_config': '',
        'schedule_sampler': 'uniform',
        'num_channels': 128,
        'dropout': 0.0,
        'use_checkpoint': False,
        'lr': 1e-4,
        'weight_decay': 0.0,
        'lr_anneal_steps': 1e5,
        'batch_size': 128,
        'microbatch': -1,
        'ema_rate': '0.9999',
        'log_interval': 1e4,
        'save_interval': 1e4,
        'resume_checkpoint': '',
        'use_fp16': False,
        'fp16_scale_growth': 1e-3,
        'output_dim': 100,
        'num_layers': 3,
        'class_cond': False,
        'diffusion_steps': 1000,
        'logger_path': '',
        'use_hvg': True,
        'use_encoder': True,
        'use_vae': False,
        'film': False,
        'use_control_set': False,
        'control_label':"non-targeting",
        'control_k':32,
    }
    default_args.update(updated_args)
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Perturbation-conditioned generative diffusion model')
    
    # Add arguments to the parser (these should correspond to the keys in default_args)
    for key, value in default_args.items():
        parser.add_argument(f'--{key}', default=value, type=type(value), help=f'{key} (default: {value})')

    # Parse command-line arguments
    args = parser.parse_args()

    # Convert the parsed arguments to a dictionary and update the defaults
    updated_args = vars(args)
    
    # Check if 'logger_path' is None and raise an error if so
    if updated_args['logger_path']=='':
        logger.log('ERROR:Please specify the logger path --logger_path.')
        raise ValueError("Logger path is required. Please specify the logger path.")

            # Check if 'logger_path' is None and raise an error if so
    if updated_args['toml_config']=='':
        logger.log("ERROR:Please specify the toml config path --toml_config.")
        raise ValueError("Dataset path is required. Please specify the path where the training adata is.")


    # Return the updated arguments as a dictionary
    return updated_args



if __name__ == "__main__":
    args_train = parse_args()
    print('**************training args*************')
    print(args_train)
    losses = run_training(args_train)
    
    plot_loss(losses,args_train)
    
    

