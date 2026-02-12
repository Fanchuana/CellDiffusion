from enum import auto
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import scanpy as sc
import os
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def Drug_dose_encoder(drug_SMILES_list: list, dose_list: list, num_Bits=1024, comb_num=1):
    """
    adopted from PRnet @Author: Xiaoning Qi.
    Encode SMILES of drug to rFCFP fingerprint
    """
    drug_len = len(drug_SMILES_list)
    fcfp4_array = np.zeros((drug_len, num_Bits))

    if comb_num==1:
        for i, smiles in enumerate(drug_SMILES_list):
            smi = smiles
            mol = Chem.MolFromSmiles(smi)
            fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
            fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
            fcfp4_list = fcfp4_list*np.log10(dose_list[i]+1)
            fcfp4_array[i] = fcfp4_list
    else:
        for i, smiles in enumerate(drug_SMILES_list):
            smiles_list = smiles.split('+')
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
                fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
                fcfp4_list = fcfp4_list*np.log10(float(dose_list[i])+1)
                fcfp4_array[i] += fcfp4_list
    return fcfp4_array 

class AnnDataDataset(Dataset):
    def __init__(self, adata, control_adata=None,use_drug_structure=False,comb_num=1):
        self.use_drug_structure = use_drug_structure
        if type(adata.X)==np.ndarray:
            self.features = torch.tensor(adata.X, dtype=torch.float32)
        else:
            self.features = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        
        if self.use_drug_structure:
            if type(control_adata.X)==np.ndarray:
                self.control_features = torch.tensor(control_adata.X, dtype=torch.float32)
            else:
                self.control_features = torch.tensor(control_adata.X.toarray(), dtype=torch.float32)
                
            self.drug_type_list = adata.obs['SMILES'].to_list()
            self.dose_list = adata.obs['dose'].to_list()
            #self.encoded_obs_tensor = torch.tensor(adata.obs['Group'].copy().values, dtype=torch.float32)
            self.encoded_obs_tensor = adata.obs['Group'].copy().values
            self.encode_drug_doses = Drug_dose_encoder(self.drug_type_list, self.dose_list, comb_num=comb_num)
            self.encode_drug_doses = torch.tensor(self.encode_drug_doses, dtype=torch.float32)
        else:
            self.encoded_obs_tensor = adata.obs['Group'].copy().values
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
       
        if self.use_drug_structure:
            return {'feature':self.features[idx], 'drug_dose':self.encode_drug_doses[idx], 'group': self.encoded_obs_tensor[idx],'control_feature':self.control_features[idx]}
        else:
            return {'feature':self.features[idx], 'group': self.encoded_obs_tensor[idx]}


class StateDataset(Dataset):
    def __init__(self, adata, model_path, use_vae = False, use_control_set=False, control_label="non-targeting", control_k=8):
        self.use_control_set = use_control_set
        self.control_label = control_label
        self.control_k = control_k
        if use_vae:
            self.features = torch.tensor(adata.obsm['VAE_latent'], dtype=torch.float32)
        else:
            if type(adata.X)==np.ndarray:
                self.features = torch.tensor(adata.X, dtype=torch.float32)
            else:
                self.features = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        # Convert categorical variables to one-hot encoding tensors
        # First, get unique categories
        # Second, map each category to an index
        # Finally, create one-hot encoded tensors
        perturb_categories = adata.obs['gene'].astype('category')
        batch_categories = adata.obs['gem_group'].astype('category')
        cell_line_categories = adata.obs['cell_line'].astype('category')
        self.perturb_categories = perturb_categories
        self.cell_line_categories = cell_line_categories
        perturb2idx = {cat: idx for idx, cat in enumerate(perturb_categories.cat.categories)}
        batch2idx = {cat: idx for idx, cat in enumerate(batch_categories.cat.categories)}
        cell_line2idx = {cat: idx for idx, cat in enumerate(cell_line_categories.cat.categories)}
        self.vocab = {'perturb': perturb2idx, 'batch': batch2idx, 'cell_line': cell_line2idx}
        self.perturb_tensor = torch.tensor(perturb_categories.cat.codes.values, dtype=torch.int64)
        self.batch_tensor = torch.tensor(batch_categories.cat.codes.values, dtype=torch.int64)
        self.cell_line_tensor = torch.tensor(cell_line_categories.cat.codes.values, dtype=torch.int64)
                # -------- NEW: control index --------
        if self.use_control_set:
            self.control_indices = {}
            for i in range(len(adata)):
                if adata.obs['gene'].iloc[i] == self.control_label:
                    ct = adata.obs['cell_line'].iloc[i]
                    self.control_indices.setdefault(ct, []).append(i)
        '''
        oht_perturb = pd.get_dummies(perturb_categories)
        oht_batch = pd.get_dummies(batch_categories)
        oht_cell_line = pd.get_dummies(cell_line_categories)
        self.perturb_tensor = torch.tensor(oht_perturb.values, dtype=torch.int32)
        self.batch_tensor = torch.tensor(oht_batch.values, dtype=torch.int32)
        self.cell_line_tensor = torch.tensor(oht_cell_line.values, dtype=torch.int32)

        self.perturb_tensor = torch.tensor(perturb_categories.cat.codes.values, dtype=torch.int64)
        '''
        # Save the vocabularies for reference
        if model_path is not None:
            vocab_path = os.path.join(os.path.dirname(model_path+'/'), 'vocab_state.pkl')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            if not os.path.exists(vocab_path):
                with open(vocab_path, 'wb') as f:
                    pickle.dump(self.vocab, f)
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {
            'feature': self.features[idx],
            'batch': self.batch_tensor[idx],
            'perturb': self.perturb_tensor[idx],
            'cell_line': self.cell_line_tensor[idx],
        }
        # -------- NEW: control-set sampling --------
        if self.use_control_set:
            cell_line_name = self.cell_line_categories.iloc[idx]
            pool = self.control_indices.get(cell_line_name, [])

            if len(pool) == 0:
                # fallback: use self as dummy
                control_ids = [idx] * self.control_k
            else:
                control_ids = np.random.choice(
                    pool,
                    size=self.control_k,
                    replace=True
                )

            control_feats = self.features[control_ids]
            item['control_set'] = control_feats

        return item



def prepared_state_data(toml_config, batch_size, model_path, use_hvg = True, use_vae=False, use_control_set=False, control_label="non-targeting", control_k=8):
    ############ Load dataset according to the toml config ############
    import toml
    config = toml.load(toml_config)
    dataset_name, dataset_path = tuple(config['datasets'].items())[0]
    total_adata = sc.read_h5ad(dataset_path)
    if 'fewshot' in dataset_path:
        task = 'fewshot'
        test_celltype = tuple(config['fewshot'].keys())[0].split('.')[1]
        test_valid = list(config['fewshot'].values())[0]
        test_perturb_list, val_perturb_list = test_valid['test'] if 'test' in test_valid else [], test_valid['val'] if 'val' in test_valid else []
        # 去掉cell type为test cell type 且 perturb在gene list中的细胞
        train_adata = total_adata[~((total_adata.obs['cell_line']==test_celltype) & (total_adata.obs['gene'].isin(test_perturb_list+val_perturb_list)))]
        #train_adata = train_adata[train_adata.obs['cell_line']=='k562']
        test_adata, val_adata = total_adata[(total_adata.obs['cell_line']==test_celltype) & (total_adata.obs['gene'].isin(test_perturb_list))], total_adata[(total_adata.obs['cell_line']==test_celltype) & (total_adata.obs['gene'].isin(val_perturb_list))]
        print(f"Train adata shape: {train_adata.shape}, Test adata shape: {test_adata.shape}, Val adata shape: {val_adata.shape} for fewshot setting.")
        # Return basic info of the dataset 
    else:
        task = 'zeroshot'
        test_celltype = tuple(config['zeroshot'].keys())[0].split('.')[1]
        # 去掉cell type除non-targeting 以外的perturbation
        train_adata = total_adata[~((total_adata.obs['cell_line']==test_celltype) & (total_adata.obs['gene']!=control_label))]
        test_adata = total_adata[(total_adata.obs['cell_line']==test_celltype) & (total_adata.obs['gene']!=control_label)]
        print(f"Train adata shape: {train_adata.shape}, Test adata shape: {test_adata.shape} for zeroshot setting.")
    perturb_len, batch_len, cell_line_len, gene_size = len(total_adata.obs['gene'].unique()), len(total_adata.obs['gem_group'].unique()), len(total_adata.obs['cell_line'].unique()), len(total_adata.var_names) if not use_hvg else sum(total_adata.var['highly_variable'])
    print(f"Perturbation types: {perturb_len}, Batch types: {batch_len}, Cell types: {cell_line_len}, Gene size: {gene_size}")
    if use_hvg: 
        print("Using highly variable genes.")
        train_adata = train_adata[:,train_adata.var['highly_variable']].copy()
    if use_vae:
        # load VAE model 
        from Squidiff.VAE import VAE
        autoencoder = VAE(
            num_genes=gene_size,
            device='cuda',
            seed=0,
            loss_ae='mse',
            hidden_dim=128,
            decoder_activation='ReLU',
        )
        subsection_name = tuple(config[task].keys())[0].split('.')[1]
        vae_path = f"/mnt/shared-storage-user/lvying/s2-project/virtual_cell/scDiffusion_revised/output/checkpoint/AE/state_{task}_VAE_{subsection_name}/model_seed=0_step=199999.pt"
        #vae_path = "/work/home/cryoem666/xyf/temp/pycharm/scDiffusion/output/checkpoint/AE/my_VAE/model_seed=0_step=199999.pt"
        autoencoder.load_state_dict(torch.load(vae_path))
        autoencoder.eval()
        with torch.no_grad():
            cell_data_embed = autoencoder(torch.tensor(train_adata.X).cuda(), return_latent=True)
            cell_data_embed = cell_data_embed.cpu().detach().numpy()
        train_adata.obsm['VAE_latent'] = cell_data_embed
        gene_size = cell_data_embed.shape[1]
        print("VAE encoding completed.")
    train_dataset = StateDataset(train_adata, model_path, use_vae, use_control_set, control_label, control_k)
    dataloader = DataLoader(
                train_dataset, 
                batch_size=batch_size,
                shuffle=True, 
                )
    
        
    return dataloader, {'perturb_len':perturb_len, 'batch_len':batch_len, 'cell_line_len':cell_line_len, 'gene_size': gene_size, 'output_dim': gene_size}
    

def prepared_data(data_dir=None,control_data_dir=None, batch_size=64,use_drug_structure=False,comb_num=1):
     
    
    train_adata = sc.read_h5ad(data_dir)
    print(use_drug_structure)
    if use_drug_structure:
        control_adata = sc.read_h5ad(control_data_dir)
    else:
        control_adata = None
    
    _data_dataset = AnnDataDataset(train_adata,control_adata,use_drug_structure,comb_num)


    dataloader = DataLoader(
                _data_dataset, 
                batch_size=batch_size,
                shuffle=True, 
                )
        
    return dataloader