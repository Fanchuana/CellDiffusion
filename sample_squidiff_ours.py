# -*- coding: utf-8 -*-
# @Author: SiyuHe
# @Last Modified by:   SiyuHe
# @Last Modified time: 2025-02-19


import argparse
import os
from pyexpat import model
import numpy as np
import torch.distributed as dist
import torch
from transformers import model_addition_debugger_context
from Squidiff import dist_util, logger
from Squidiff.script_util import (
    NUM_CLASSES,
    create_our_film_model_and_diffusion,
    our_model_and_diffusion_defaults,
    create_our_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from Squidiff.resample import create_named_schedule_sampler
from sklearn.metrics import r2_score
from Squidiff.VAE import VAE
import scipy
class sampler:
    def __init__(self, model_path = None, perturb_len = None, batch_len = None, cell_line_len = None, gene_size = None, diffusion_steps = None, use_vae = False, film = False, use_ddim = False, subsection_name = None):
        print("load model and diffusion...")
        if use_vae:
            autoencoder = VAE(
                num_genes=2000,
                device='cuda',
                seed=0,
                loss_ae='mse',
                hidden_dim=128,
                decoder_activation='ReLU',
            )
            import torch
            vae_path = f"/work/home/cryoem666/xyf/temp/pycharm/scDiffusion/output/checkpoint/AE/my_VAE_{subsection_name}/model_seed=0_step=199999.pt"
            #vae_path = "/work/home/cryoem666/xyf/temp/pycharm/scDiffusion/output/checkpoint/AE/my_VAE/model_seed=0_step=199999.pt"
            autoencoder.load_state_dict(torch.load(vae_path))
            autoencoder.eval()
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None
        args = self.parse_args(model_path, perturb_len, batch_len, cell_line_len, gene_size, diffusion_steps, use_vae, film, use_ddim)
        print(args)
        if args['film']:
            model, diffusion = create_our_film_model_and_diffusion(
                **args_to_dict(args, our_model_and_diffusion_defaults().keys())
            )
        else:
            model, diffusion = create_our_model_and_diffusion(
                    **args_to_dict(args, our_model_and_diffusion_defaults().keys())
                )

        model.load_state_dict(
            dist_util.load_state_dict(args['model_path'])
        )
        model.to(dist_util.dev())
        if args['use_fp16']:
            model.convert_to_fp16()
        model.eval()
        self.model = model
        self.arg = args
        self.diffusion = diffusion
        self.sample_fn = (diffusion.p_sample_loop if not args['use_ddim'] else diffusion.ddim_sample_loop)
    
    def stochastic_encode(
        self, model, x, t, model_kwargs):
        """
        ddim reverse sample
        """
        sample = x
        sample_t = []
        xstart_t = []
        T = []
        indices = list(range(t))

        for i in indices:
            timestep = torch.full((x.shape[0],), i, device='cuda').long()
            with torch.no_grad():
                out = self.diffusion.ddim_reverse_sample(model, 
                                                    sample, 
                                                    timestep, 
                                                    model_kwargs=model_kwargs)
                sample = out['sample']
                sample_t.append(sample)
                xstart_t.append(out['pred_xstart'])
                T.append(timestep)

        return {
        'sample': sample,
        'sample_t': sample_t,
        'xstart_t': xstart_t,
        'T': T,
    }

    def parse_args(self,model_path, perturb_len, batch_len, cell_line_len, gene_size, diffusion_steps, use_vae, film, use_ddim):
        """Parse command-line arguments and update with default values."""
        # Define default arguments
        default_args = {}
        default_args.update(our_model_and_diffusion_defaults())
        updated_args = {
            'data_path': '',
            'schedule_sampler': 'uniform',
            'lr': 1e-4,
            'weight_decay': 0.0,
            'lr_anneal_steps': 1e5,
            'batch_size': 16,
            'microbatch': -1,
            'ema_rate': '0.9999',
            'log_interval': 1e4,
            'save_interval': 1e4,
            'resume_checkpoint': '',
            'use_fp16': False,
            'fp16_scale_growth': 1e-3,
            'state_dataset_config': {
            'perturb_len': perturb_len,
            'batch_len': batch_len,
            'cell_line_len': cell_line_len,
            'gene_size': gene_size,
            'output_dim': gene_size,
            },
            'output_dim': gene_size,
            'num_layers': 3,
            'class_cond': False,
            'use_encoder': True,
            'use_ddim': use_ddim,
            'diffusion_steps': diffusion_steps,
            'logger_path': '',
            'model_path': model_path,
            'comb_num':1,
            'drug_dimension':1024,
            'use_vae': use_vae,
            'film': film,
        }
        default_args.update(updated_args)

        # Return the updated arguments as a dictionary
        return default_args

    def load_squidiff_model(self):
        print("load model and diffusion...")
        return self.model

    def load_sample_fn(self):
        
        return self.sample_fn

    def get_diffused_data(self,model, x, t, model_kwargs):
        sample = x
        sample_t = [x]  # Store initial data for plotting
        xstart_t = []
        T = []

        indices = list(range(t))

        for i in indices:
            timestep = torch.full((x.shape[0],), i, device='cuda').long()
            with torch.no_grad():
                # Replacing ddim_reverse_sample with a simpler forward diffusion process
                noise = torch.randn_like(sample)  # Add noise at each step
                out = sample + noise * (i / t)    # Simulating diffusion based on time step
                sample = out
                sample_t.append(sample.cpu())  # Store the samples for visualization
                xstart_t.append(sample.cpu())
                T.append(timestep)

        return {
            'sample': sample,
            'sample_t': sample_t,
            'xstart_t': xstart_t,
            'T': T
        }

    def sample_around_point(self, point, num_samples=None, scale=0.7):
        return point + scale * np.random.randn(num_samples, point.shape[0])

    def pred(self, model_kwargs, gene_size, return_latent = False):
        if 'z_mod' in model_kwargs:
            pred_result = self.sample_fn(
                            self.model,
                            shape = (model_kwargs['z_mod']['gamma'].shape[0], gene_size),
                            model_kwargs=model_kwargs,
                            noise =  None
                    )
        elif 'perturb' in model_kwargs:
            pred_result = self.sample_fn(
                            self.model,
                            shape = (model_kwargs['perturb'].shape[0],gene_size),
                            model_kwargs=model_kwargs,
                            noise =  None
                    )
        if self.autoencoder:
            with torch.no_grad():
                pred_result_new = self.autoencoder(pred_result, return_decoded=True).detach().cpu().numpy()
        if return_latent:
            if self.autoencoder:
                return pred_result.detach().cpu().numpy(), pred_result_new
            else:
                return pred_result.detach().cpu().numpy(), None
        else:
            return None, pred_result_new
    
    def interp_with_direction(self, z_sem_origin = None, gene_size = None, direction = None, scale = 1, add_noise_term = True):

        z_sem_origin = z_sem_origin.detach().cpu().numpy()
        z_sem_interp_ = z_sem_origin.mean(axis=0) + direction.detach().cpu().numpy() * scale
        if add_noise_term:
            z_sem_interp_ = self.sample_around_point(z_sem_interp_, num_samples=z_sem_origin.shape[0])

        z_sem_interp_ = torch.tensor(z_sem_interp_,dtype=torch.float32).to('cuda')
        sample_interp = self.sample_fn(
                            self.model,
                            shape = (z_sem_origin.shape[0],gene_size),
                            model_kwargs={
                                'z_mod': z_sem_interp_
                            },
                            noise =  None
        )
        if self.autoencoder:
            with torch.no_grad():
                sample_interp = self.autoencoder(sample_interp, return_decoded=True).detach().cpu().numpy()
        return sample_interp
        
    def cal_metric(self,x1,x2):
        r2 = r2_score(x1.detach().cpu().numpy().mean(axis=0),
                      x2.X.mean(axis=0))
        pearsonr,_ = scipy.stats.pearsonr(x1.detach().cpu().numpy().mean(axis=0),
                      x2.X.mean(axis=0))
        return r2, pearsonr

        

