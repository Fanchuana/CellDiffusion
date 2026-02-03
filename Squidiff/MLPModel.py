from abc import abstractmethod

import math
from tkinter.tix import Control

import numpy as np
from sklearn import base
import state
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, z_sem):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        
        

class MLPBlock(TimestepBlock):
    """
    Basic MLP block with an optional timestep embedding.
    """

    def __init__(self, input_dim, output_dim, time_embed_dim=None, latent_dim = None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.time_embed_dim = time_embed_dim
        if time_embed_dim is not None:
            self.time_dense = nn.Linear(time_embed_dim, output_dim)
        if latent_dim is not None:
            self.zsem_dense = nn.Linear(latent_dim, output_dim)
    
    def forward(self, x, emb, z_sem):
        
        h = F.silu(self.layer_norm1(self.fc1(x))) 
        if ((emb is not None)&(z_sem is None)):
            h = h + self.time_dense(emb)
        elif ((emb is not None)&(z_sem is not None)):
            h = h + self.time_dense(emb)+self.zsem_dense(z_sem)
        h = F.silu(self.layer_norm2(self.fc2(h)))
        return h
    
    


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, z_sem):
        
        for layer in self:
            
            
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, z_sem)
            else:
                x = layer(x)
        return x

class MLPModel(nn.Module):
    """
    MLP model for single-cell RNA-seq data with timestep embedding.
    """

    def __init__(self, 
                 gene_size, 
                 output_dim, 
                 num_layers, 
                 hidden_sizes=2048,
                 time_pos_dim=2048,
                 num_classes = None,
                 latent_dim=60,
                 use_checkpoint=False,
                 use_fp16 = False,
                 use_scale_shift_norm =False,
                 dropout=0,
                 time_embed_dim=2048,
                 use_encoder=False,
                 use_drug_structure = False,
                 drug_dimension = 1024,
                 comb_num=1,
                 
                ):
        super().__init__()
        
        self.use_encoder = use_encoder
        self.time_embed_dim = time_embed_dim
        self.latent_dim = latent_dim
        self.time_embed = None
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.drug_dimension = drug_dimension
        if use_encoder: 
            self.encoder = EncoderMLPModel(gene_size,self.hidden_sizes, self.num_classes, use_drug_structure, self.drug_dimension, comb_num)
        
        
        if time_embed_dim is not None:
            self.time_embed = nn.Sequential(
                nn.Linear(time_pos_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        if self.use_encoder: 
            layers = []
            for _ in range(num_layers):
                layers.append(MLPBlock(hidden_sizes, hidden_sizes, time_embed_dim, latent_dim))
            self.mlp_blocks = TimestepEmbedSequential(*layers)
        else:
            layers = []
            for _ in range(num_layers):
                layers.append(MLPBlock(hidden_sizes, hidden_sizes, time_embed_dim))
            self.mlp_blocks = TimestepEmbedSequential(*layers)
        
        self.input_layer = nn.Linear(gene_size, hidden_sizes)
        self.output_layer = nn.Linear(hidden_sizes, output_dim)
        
    def forward(self, x, timesteps=None, **model_kwargs):
        
        
        if self.time_embed is not None and timesteps is not None:
            
            emb = self.time_embed(timestep_embedding(timesteps, self.hidden_sizes))
            
        else:
            emb = None
            
        if self.use_encoder: 
            if 'z_mod' in model_kwargs.keys():
                z_sem = model_kwargs['z_mod']
            elif self.num_classes is None:
                z_sem = self.encoder(model_kwargs['x_start'],label = None,drug_dose = model_kwargs['drug_dose'],control_feature = model_kwargs['control_feature'])
            else: 
                z_sem = self.encoder(model_kwargs['x_start'],label = model_kwargs['group'],drug_dose = model_kwargs['drug_dose'],control_feature = model_kwargs['control_feature'])

            h = self.input_layer(x)
            
            h = self.mlp_blocks(x=h, emb=emb, z_sem=z_sem)
            h = self.output_layer(h)
        else:
            z_sem = None
            h = self.input_layer(x)
            h = self.mlp_blocks(x=h, emb=emb, z_sem=z_sem)
            h = self.output_layer(h)
        return h

    
class EncoderMLPModel(nn.Module):

    def __init__(self, input_size, hidden_sizes, num_classes=None, use_drug_structure=False, drug_dimension=1024,comb_num=1,output_size=60, dropout=0.1, use_fp16=False):
        super(EncoderMLPModel, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dtype = th.float16 if use_fp16 else th.float32
        self.drug_dimension = drug_dimension
    
        if num_classes is None:
            l1 = 0
        else: 
            l1 = hidden_sizes
        if use_drug_structure:
            l2 = drug_dimension
        else:
            l2 = 0
        
        self.fc1 = nn.Linear(input_size+l1+l2, hidden_sizes)
        self.bn1 = nn.BatchNorm1d(hidden_sizes)
        self.bn2 = nn.BatchNorm1d(hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, output_size)
       
        self.label_embed = nn.Linear(1, hidden_sizes)
    
    def forward(self, x_start, label=None, drug_dose=None, control_feature = None):
        
        if label is not None:
            label_emb = self.label_embed(label)
            x_start = th.concat([x_start,label_emb],axis=1)
        
        if drug_dose is not None:
            x_start = th.concat([control_feature,drug_dose],axis=1)
            
        h = x_start.type(self.dtype)
        h = F.relu(self.bn1(self.fc1(h)))
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.fc3(h)
        return h
    

    
class EncoderMLPModel2(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes=None, output_size=60, dropout=0.1, use_fp16=False):
        super(EncoderMLPModel2, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.dtype = th.float16 if use_fp16 else th.float32
        
        self.fc1 = nn.Linear(input_size, hidden_sizes)
        self.bn1 = nn.BatchNorm1d(hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.bn2 = nn.BatchNorm1d(hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, output_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.label_embed = nn.Linear(1, hidden_sizes)  

    def forward(self, x_start, label=None):
        h = x_start.type(self.dtype)
        h = F.relu(self.bn1(self.fc1(h)))
    

        if label is not None:
            label = label.type(self.dtype) 
            label_emb = self.label_embed(label)
            
            h = h + label_emb  # Add label embedding as a residual connection


        h = self.dropout_layer(h)
        h = self.fc3(h)
       
        return h
    






class our_MLPModel(nn.Module):
    """
    MLP model for single-cell RNA-seq data with timestep embedding.
    """

    def __init__(self, 
                 gene_size, 
                 output_dim, 
                 num_layers, 
                 hidden_sizes=2048,
                 time_pos_dim=2048,
                 num_classes = None,
                 latent_dim=60,
                 use_checkpoint=False,
                 use_fp16 = False,
                 use_scale_shift_norm =False,
                 dropout=0,
                 time_embed_dim=2048,
                 use_encoder=False,
                 state_dataset_config = None
                ):
        super().__init__()
        
        self.use_encoder = use_encoder
        self.time_embed_dim = time_embed_dim
        self.latent_dim = latent_dim
        self.time_embed = None
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        if use_encoder: 
            self.encoder = EncoderMLPModel4(
                perturb_len = state_dataset_config['perturb_len'], 
                                            batch_len = state_dataset_config['batch_len'], 
                                            cell_type_len = state_dataset_config['cell_line_len'], 
                                            input_size = gene_size,
                                            hidden_sizes =self.hidden_sizes, 
                                            use_fp16=use_fp16)
        
        
        if time_embed_dim is not None:
            self.time_embed = nn.Sequential(
                nn.Linear(time_pos_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        if self.use_encoder: 
            layers = []
            for _ in range(num_layers):
                layers.append(MLPBlock(hidden_sizes, hidden_sizes, time_embed_dim, latent_dim))
            self.mlp_blocks = TimestepEmbedSequential(*layers)
        else:
            layers = []
            for _ in range(num_layers):
                layers.append(MLPBlock(hidden_sizes, hidden_sizes, time_embed_dim))
            self.mlp_blocks = TimestepEmbedSequential(*layers)
        
        self.input_layer = nn.Linear(gene_size, hidden_sizes)
        self.output_layer = nn.Linear(hidden_sizes, output_dim)
        
    def forward(self, x, timesteps=None, **model_kwargs):
        
        
        if self.time_embed is not None and timesteps is not None:
            
            emb = self.time_embed(timestep_embedding(timesteps, self.hidden_sizes))
            
        else:
            emb = None
            
        if self.use_encoder: 
            if 'z_mod' in model_kwargs.keys():
                z_sem = model_kwargs['z_mod']
            else:
                #z_sem = self.encoder(model_kwargs['x_start'])
                z_sem = self.encoder(model_kwargs['x_start'],perturb_label = model_kwargs['perturb'],batch = model_kwargs['batch'],cell_type = model_kwargs['cell_line'])

            h = self.input_layer(x)
            
            h = self.mlp_blocks(x=h, emb=emb, z_sem=z_sem)
            h = self.output_layer(h)
        else:
            z_sem = None
            h = self.input_layer(x)
            h = self.mlp_blocks(x=h, emb=emb, z_sem=z_sem)
            h = self.output_layer(h)
        return h




class EncoderMLPModel3(nn.Module):

    def __init__(self, input_size = None, hidden_sizes = None, perturb_len = None, batch_len = None, cell_type_len = None, output_size=60, dropout=0.1, use_fp16=False):
        super(EncoderMLPModel3, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dtype = th.float16 if use_fp16 else th.float32
        self.pert_encoder = nn.Embedding(perturb_len, hidden_sizes)
        self.batch_encoder = nn.Embedding(batch_len, hidden_sizes)
        self.celltype_encoder = nn.Embedding(cell_type_len, hidden_sizes)
        self.x_start_encoder = nn.Linear(input_size, hidden_sizes)
        

        self.fc1 = nn.Linear(hidden_sizes, hidden_sizes)
        self.bn1 = nn.BatchNorm1d(hidden_sizes)
        self.bn2 = nn.BatchNorm1d(hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, output_size)
    
    def forward(self, x_start, perturb_label=None, batch=None, cell_type = None):
        perturb_emb = self.pert_encoder(perturb_label)
        batch_emb = self.batch_encoder(batch)
        celltype_emb = self.celltype_encoder(cell_type)
        x_start = self.x_start_encoder(x_start)
        x_start = x_start + perturb_emb + batch_emb + celltype_emb
        #print(x_start.shape, perturb_emb.shape, batch_emb.shape, celltype_emb.shape)
        h = x_start.type(self.dtype)
        h = F.relu(self.bn1(self.fc1(h)))
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.fc3(h)
        return h
    
class EncoderMLPModel4(nn.Module):

    def __init__(self, input_size = None, hidden_sizes = None, perturb_len = None, batch_len = None, cell_type_len = None, output_size=60, dropout=0.1, use_fp16=False):
        super(EncoderMLPModel4, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dtype = th.float16 if use_fp16 else th.float32
        self.pert_encoder = nn.Embedding(perturb_len, hidden_sizes)
        #self.batch_encoder = nn.Embedding(batch_len, hidden_sizes)
        self.celltype_encoder = nn.Embedding(cell_type_len, hidden_sizes)
        #self.x_start_encoder = nn.Linear(input_size, hidden_sizes)
        

        self.fc1 = nn.Linear(hidden_sizes, hidden_sizes)
        self.bn1 = nn.BatchNorm1d(hidden_sizes)
        self.bn2 = nn.BatchNorm1d(hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, output_size)
    
    def forward(self, x_start, perturb_label=None, batch=None, cell_type = None):
        perturb_emb = self.pert_encoder(perturb_label)
        #batch_emb = self.batch_encoder(batch)
        celltype_emb = self.celltype_encoder(cell_type)
        #x_start = self.x_start_encoder(x_start)
        #x_start = x_start + celltype_emb
        x_start = celltype_emb + perturb_emb
        #print(x_start.shape, perturb_emb.shape, batch_emb.shape, celltype_emb.shape)
        h = x_start.type(self.dtype)
        h = F.relu(self.bn1(self.fc1(h)))
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.fc3(h)
        return h
    
'''
class HierarchicalConditionEncoder(nn.Module):
    def __init__(self, perturb_len, cell_type_len, hidden_size, latent_dim):
        super().__init__()
        
        # 1. 基础 Embedding
        self.perturb_embed = nn.Embedding(perturb_len, hidden_size)
        self.cell_type_embed = nn.Embedding(cell_type_len, hidden_size)
        
        # 2. 特征提取器 (使用 LayerNorm 保证尺度一致)
        self.cell_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        self.pert_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        # 3. 显式交互层 (Interaction Layer)
        # 很多时候 MLP 难以拟合乘法关系，显式构造交互项有助于泛化到 unseen combinations
        self.interaction_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # 4. FiLM 参数生成器
        # 输入是 [Cell, Pert, Cell*Pert] 的拼接
        self.film_generator = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, latent_dim * 2)  # 输出 gamma, beta
        )
        
    def forward(self, x_start, perturb_label, cell_type, batch):
        # 获得基础向量
        p_emb = self.perturb_embed(perturb_label)
        c_emb = self.cell_type_embed(cell_type)
        
        # 提取特征
        p_feat = self.pert_encoder(p_emb)
        c_feat = self.cell_encoder(c_emb)
        
        # 计算显式交互 (Element-wise Product)
        # 物理含义：特定的细胞环境(c)下，扰动(p)的特异性激活
        interaction = self.interaction_proj(p_feat * c_feat)
        
        # 拼接所有信息：基线 + 扰动 + 交互效应
        composed_feat = th.cat([c_feat, p_feat, interaction], dim=-1)
        
        # 生成 FiLM 参数
        film_params = self.film_generator(composed_feat)
        
        
        return film_params
'''
class HierarchicalConditionEncoder(nn.Module):
    def __init__(self, perturb_len, cell_type_len, hidden_size, latent_dim):
        super().__init__()
        
        # 1. 基础 Embedding
        self.perturb_len = perturb_len
        self.perturb_embed = nn.Embedding(perturb_len + 1, hidden_size)
        with th.no_grad():
            self.perturb_embed.weight[perturb_len].zero_()
        self.cell_type_embed = nn.Embedding(cell_type_len, hidden_size)
        
        # 2. 特征预处理
        self.cell_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        self.pert_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        # 3. Path A: Baseline Generator (仅由 Cell Type 决定)
        # 目的：捕捉细胞类型之间的巨大差异 (Global Manifold)
        self.cell_base_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim * 2)  # 输出基准 gamma_c, beta_c
        )

        # 4. Path B: Interaction Delta Generator (Gating Mechanism)
        # 目的：捕捉 Perturbation 在特定 Cell Type 下的微小效应 (Local Shift)
        # 逻辑：Cell Type 生成一个 Gate (0-1)，决定 Perturbation 向量的哪些维度被激活
        self.interaction_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid() 
        )
        
        self.perturb_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        self.delta_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim *2) # 输出偏移 gamma_delta
        )
        
        # 初始化 Trick: 让 Delta 分支的初始权重非常小
        # 这样模型起始状态等同于无扰动，避免强行拟合扰动噪声破坏 Cell 表征
        with th.no_grad():
            self.delta_generator[-1].weight.data.mul_(0.01)
            self.delta_generator[-1].bias.data.zero_()
        
    def forward(self, perturb_label, cell_type, batch):
        p_emb = self.perturb_embed(perturb_label)
        c_emb = self.cell_type_embed(cell_type)
        
        # 提取特征
        p_feat = self.pert_encoder(p_emb)
        c_feat = self.cell_encoder(c_emb)
        
        # 1. 计算 Cell Baseline 参数 (Base)
        base_params = self.cell_base_generator(c_feat)
        base_gamma, base_beta = th.chunk(base_params, 2, dim=-1)
        # 2. 计算 Interaction Delta 参数 (Shift)
        # 物理含义：Specific_Perturbation * Cell_Context_Gate
        gate = self.interaction_gate(c_feat)          # 当前细胞环境允许哪些通路激活
        p_effect = self.perturb_projection(p_feat)    # 扰动本身的潜在机制特征
        
        interaction_feat = p_effect * gate            # 交互特征
        
        delta_params = self.delta_generator(interaction_feat)
        delta_gamma, delta_beta = th.chunk(delta_params, 2, dim=-1)
        gamma, beta = base_gamma + delta_gamma, base_beta + delta_beta  # 仅调整 gamma
        # 3. 显式残差叠加
        # 最终参数 = 细胞基准 + 扰动偏移
        return {
                "gamma": gamma,
                "beta": beta
            }
    
class ControlSetAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, h, control_set):
        """
        h: (B, H)
        control_set: (B, K, H)
        """
        q = h.unsqueeze(1)      # (B, 1, H)
        k = control_set         # (B, K, H)
        v = control_set
        out, _ = self.attn(q, k, v)
        return out.squeeze(1)   # (B, H)


class Our_MLPModel_FiLM(nn.Module):
    """
    使用FiLM条件调制的MLP模型
    """
    def __init__(self, 
                 gene_size, 
                 output_dim, 
                 num_layers, 
                 hidden_sizes=2048,
                 time_pos_dim=2048,
                 perturb_len=None,
                 cell_type_len=None,
                 latent_dim=60,
                 use_fp16=False,
                 dropout=0,
                 time_embed_dim=2048,
                 use_encoder=False,
                 state_dataset_config=None):
        super().__init__()
        
        self.use_encoder = use_encoder
        self.time_embed_dim = time_embed_dim
        self.hidden_sizes = hidden_sizes

        # FiLM条件编码器
        if use_encoder:
            self.encoder = HierarchicalConditionEncoder(
                perturb_len=state_dataset_config['perturb_len'],
                cell_type_len=state_dataset_config['cell_line_len'],
                hidden_size=hidden_sizes,
                latent_dim=hidden_sizes  # 输出与hidden_sizes匹配
            )
        
        # 时间嵌入
        if time_embed_dim is not None:
            self.time_embed = nn.Sequential(
                nn.Linear(time_pos_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        
        # MLP块（使用FiLM调制）
        layers = []
        for i in range(num_layers):
            layers.append(FiLMMLPBlock(
                input_dim=hidden_sizes if i > 0 else hidden_sizes,
                output_dim=hidden_sizes,
                time_embed_dim=time_embed_dim,
                condition_dim=hidden_sizes  # FiLM参数由外部提供
            ))
        self.mlp_blocks = nn.ModuleList(layers)
        
        # 输入输出层
        self.input_layer = nn.Linear(gene_size, hidden_sizes)
        self.output_layer = nn.Linear(hidden_sizes, output_dim)

        # 为每一层创建独立的FiLM参数映射
        self.layer_gamma = nn.ModuleList([
            nn.Linear(hidden_sizes, hidden_sizes) 
            for _ in range(num_layers)
        ])
        self.layer_beta = nn.ModuleList([
            nn.Linear(hidden_sizes, hidden_sizes) 
            for _ in range(num_layers)
        ])

        
    def forward(self, x, timesteps=None, **model_kwargs):
        # 时间嵌入
        if self.time_embed is not None and timesteps is not None:
            emb = self.time_embed(timestep_embedding(timesteps, self.hidden_sizes))
        else:
            emb = None
        
        if 'z_mod' in model_kwargs.keys():
            film_params = model_kwargs['z_mod']
        # 条件编码（生成FiLM参数）
        if self.use_encoder and 'perturb' in model_kwargs and 'cell_line' in model_kwargs:
            film_params = self.encoder.forward(
                perturb_label = model_kwargs['perturb'], 
                cell_type = model_kwargs['cell_line'],
                batch = model_kwargs['batch']
            )
        if film_params is not None:
            gamma, beta = film_params['gamma'], film_params['beta']
        else:
            gamma, beta = None, None
        # 为每一层生成独立的FiLM参数，但依赖于同一条件编码
        film_params_per_layer = []
        for i in range(len(self.mlp_blocks)):
            gamma_i = self.layer_gamma[i](gamma)
            beta_i  = self.layer_beta[i](beta)
            film_params_per_layer.append((gamma_i, beta_i))
        '''
        #固定layer参数？
        film_params_per_layer = [(gamma, beta)] * len(self.mlp_blocks)
        '''
        # 前向传播
        h = self.input_layer(x)
        if 'control_set' in model_kwargs and model_kwargs['control_set'] is not None:
            input_dim = model_kwargs['control_set'].shape
            control_set = model_kwargs['control_set'].reshape(-1, input_dim[-1])  # (B*K, H)
            control_set = self.input_layer(control_set)  # 投影到隐藏维度
            model_kwargs['control_set'] = control_set.reshape(-1, input_dim[1], self.hidden_sizes)  # (B, K, H)

        for i, block in enumerate(self.mlp_blocks):
            film_params = (film_params_per_layer[i][0], film_params_per_layer[i][1]) if film_params_per_layer[i][0] is not None else None
            h = block(x=h, emb=emb, film_params=film_params, control_set=model_kwargs['control_set'])
        
        h = self.output_layer(h)
        return h


class FiLMMLPBlock(TimestepBlock):
    """
    FiLM调制的MLP块
    """
    def __init__(self, input_dim, output_dim, time_embed_dim=None, condition_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.cross_attention = ControlSetAttention(hidden_dim=output_dim) if condition_dim is not None else None
        
        # 时间嵌入
        if time_embed_dim is not None:
            self.time_dense = nn.Linear(time_embed_dim, output_dim)
        
    def forward(self, x, emb, film_params, control_set=None):
        # 如果没有film_params，就使用默认值（gamma=1, beta=0）
        if film_params is None:
            gamma = th.ones_like(x[:, :self.fc1.out_features])
            beta = th.zeros_like(x[:, :self.fc1.out_features])
        else:
            gamma, beta = film_params
            
        # 第一层
        h = self.fc1(x)
        h = self.norm1(h)
        h = h * (1 + gamma) + beta  # FiLM调制
        h = F.silu(h)
        # 时间嵌入
        if emb is not None:
            h = h + self.time_dense(emb)
        if control_set is not None and self.cross_attention is not None:
            h = h + self.cross_attention(h, control_set)
        
        # 第二层
        h = self.fc2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h