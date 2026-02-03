from pyexpat import model
import anndata as ad 
import pickle 
import toml 
from Squidiff import scrna_datasets
import sample_squidiff_ours
from tqdm import tqdm 
import numpy as np
import torch
from tqdm import tqdm
fold2name = {
    1: 'hepg2',
    2: 'jurkat',
    3: 'k562',
    4: 'rpe1'
}
def inference_result(input_dir: str, output_dir: str, fold: int, film: bool = False, use_ddim: bool = True, use_control_set: bool = False, control_label: str = 'non-targeting', control_k: int = 8):
    input_dir, output_dir = input_dir.rstrip('/'), output_dir.rstrip('/')
    file = pickle.load(open(f"{input_dir}/vocab_state.pkl","rb"))
    config = toml.load(f"/work/home/cryoem666/xyf/temp/pycharm/Squidiff/data/replogle/{fold2name[fold]}.toml")
    state_Data = ad.read_h5ad(config['datasets']['replogle_proper'])
    differ_gene = list(set(state_Data.obs['gene'])-set(file['perturb']))
    for i, gene in enumerate(differ_gene):
        file['perturb'][gene] = file['perturb']['non-targeting']
    #file['perturb']['non-targeting'] = len(state_Data.obs['gene'].unique()) + 1
    #print("input_dir:", input_dir)
    sampler = sample_squidiff_ours.sampler(
            model_path = f'{input_dir}/model.pt',
            perturb_len = len(file['perturb']),
            batch_len = len(file['batch']),
            cell_line_len = len(file['cell_line']),
            gene_size = 128,
            diffusion_steps = 1000,
            use_vae=True,
            film = film,
            use_ddim = use_ddim,
            subsection_name=fold2name[fold]
        )
    state_Data.obs['perturb_label'] = state_Data.obs['gene'].apply(lambda x: file['perturb'][x])
    state_Data.obs['batch_label'] = state_Data.obs['gem_group'].apply(lambda x: file['batch'][x])
    state_Data.obs['cell_line_label'] = state_Data.obs['cell_line'].apply(lambda x: file['cell_line'][x])
    hepg2_real = ad.read_h5ad(f'/work/home/cryoem666/xyf/temp/pycharm/state-reproduce/baselines/baseline_output/cpa_replogle_v2/fold{fold}/adata_real.h5ad')
    hepg2_real.obs['perturb_label'] = hepg2_real.obs['pert_name'].apply(lambda x: file['perturb'][x])
    hepg2_real.obs['cell_line_label'] = hepg2_real.obs['celltype_name'].apply(lambda x: file['cell_line'][x])
    hepg2_real.obsm['X_squidiff_ours'] = np.zeros_like(hepg2_real.X)
    state_control = state_Data[(state_Data.obs['gene']==control_label)&(state_Data.obs['cell_line']==fold2name[fold])][:, state_Data.var['highly_variable']].copy()
    state_control.obsm['VAE_latent'] = sampler.autoencoder(torch.tensor(state_control.X).float().to('cuda'), return_latent=True).cpu().detach().numpy()
    control_set = state_control.obsm['VAE_latent']
    for gene in tqdm(hepg2_real.obs['pert_name'].unique()):
        mask = (hepg2_real.obs['pert_name'].values == gene)
        n = int(mask.sum())
        if n == 0:
            continue
        model_kwargs = {
            'perturb': torch.tensor(hepg2_real.obs['perturb_label'][mask].values).long().to('cuda'),
            'cell_line': torch.tensor(hepg2_real.obs['cell_line_label'][mask].values).long().to('cuda'),
            'batch': None
        }
        '''
        z_sem = sampler.model.encoder.forward(
            perturb_label = torch.tensor(hepg2_real.obs['perturb_label'][mask].values).long().to('cuda'),
            cell_type = torch.tensor(hepg2_real.obs['cell_line_label'][mask].values).long().to('cuda'),
            batch = None
        )
        #从control set中随机选择n次，每次选择control_k个
        '''
        if use_control_set:
            control_features = []
            for _ in range(n):
                idxs = np.random.choice(
                    control_set.shape[0],
                    size=control_k,
                    replace=True
                )
                control_features.append(control_set[idxs])

            control_features = torch.tensor(
                np.stack(control_features)
            ).float().to('cuda')
        else:
            control_features = None
        with torch.no_grad():
            '''
            _, scrna_pred = sampler.pred(
                z_sem = z_sem,
                gene_size = 128,
                return_latent = True,
                control_set=control_features
            )
            '''
            model_kwargs['control_set'] = control_features
            _, scrna_pred = sampler.pred(
                model_kwargs = model_kwargs.copy(),
                gene_size = 128,
                return_latent = True
            )

        hepg2_real.obsm['X_squidiff_ours'][mask] = scrna_pred
    hepg2_save = hepg2_real.copy()
    hepg2_save.X = hepg2_real.obsm['X_squidiff_ours']
    hepg2_save.obsm = {}
    hepg2_real.obsm = {}
    # 检查负值情况
    def filter(hepg2_save):
        neg_mask = hepg2_save.X < 0
        neg_count = neg_mask.sum()
        total_elements = hepg2_save.X.size

        print(f"Negative values: {neg_count}/{total_elements} ({neg_count/total_elements*100:.2f}%)")
        print(f"Min value: {hepg2_save.X.min():.6f}")
        print(f"Max value: {hepg2_save.X.max():.6f}")
        hepg2_save.X = np.maximum(hepg2_save.X, 0)
        large_mask = hepg2_save.X > 10
        large_count = large_mask.sum()
        print(f"Values greater than 10: {large_count}/{total_elements} ({large_count/total_elements*100:.2f}%)")

        hepg2_save.X = np.minimum(hepg2_save.X, 10)

        print(f"After truncation:")
        print(f"Min value: {hepg2_save.X.min():.6f}")
        print(f"Max value: {hepg2_save.X.max():.6f}")
        return hepg2_save
    hepg2_save = filter(hepg2_save)
    import os 
    if os.path.isdir(output_dir) is False:
        os.makedirs(output_dir)
    hepg2_save.write_h5ad(f'{output_dir}/{fold2name[fold]}_adata_pred.h5ad')
    hepg2_real.write_h5ad(f'{output_dir}/{fold2name[fold]}_adata_real.h5ad')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing model and vocab_state.pkl')
    parser.add_argument('--output_dir', type=str, required=False, default="/work/home/cryoem666/xyf/temp/pycharm/Squidiff/test_conditional_diffusion_result/" ,help='Output directory to save results')
    parser.add_argument('--fold', type=int, required=True, help='Fold number (0-3)')
    parser.add_argument('--film', type=bool, required=False, default=False, help='Whether to use FiLM model')
    parser.add_argument('--use_ddim_reverse', type=bool, required=False, default=False, help='Whether to use DDIM sampling')
    parser.add_argument('--use_control_set', type=bool, required=False, default=False, help='Whether to use control set during inference')
    parser.add_argument('--control_label', type=str, required=False, default='non-targeting', help='Control label name')
    parser.add_argument('--control_k', type=int, required=False, default=32, help='Number of control samples to use per cell')
    args = parser.parse_args()
    args.use_ddim = 1 - args.use_ddim_reverse
    inference_result(args.input_dir, args.output_dir, args.fold, args.film, args.use_ddim, args.use_control_set, args.control_label, args.control_k)