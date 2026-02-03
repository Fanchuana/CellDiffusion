import torch

@torch.no_grad()
def _median_heuristic(D2):
    # D2: [N, N] squared distances
    # 去掉对角线 0，取中位数更稳
    N = D2.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=D2.device)
    return torch.median(D2[mask])

def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, device="cuda"):
    source = torch.as_tensor(source, device=device, dtype=torch.float32)
    target = torch.as_tensor(target, device=device, dtype=torch.float32)
    n, m = source.size(0), target.size(0)
    total = torch.cat([source, target], dim=0)  # [N, d]
    N = n + m

    # [N, N] 欧氏距离；cdist返回的是sqrt的距离，这里平方
    D2 = torch.cdist(total, total, p=2) ** 2

    if fix_sigma is None:
        # 更常用的 median heuristic
        sigma2 = _median_heuristic(D2).clamp_min(1e-12)
    else:
        sigma2 = torch.tensor(fix_sigma, device=device, dtype=torch.float32)

    # 多核：sigma2 / kernel_mul^(k//2) 作为基准
    base = sigma2 / (kernel_mul ** (kernel_num // 2))
    K = 0.0
    for i in range(kernel_num):
        gamma = base * (kernel_mul ** i)
        K = K + torch.exp(-D2 / (2.0 * gamma))

    XX = K[:n, :n]
    YY = K[n:, n:]
    XY = K[:n, n:]
    YX = K[n:, :n]
    return XX.mean() + YY.mean() - XY.mean() - YX.mean()

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    
    batch_size = 200
    num_window = int(total0.shape[0]/batch_size)+1
    L2_dis = []
    from tqdm import tqdm
    for i in tqdm(range(num_window)):
        diff = (total0[i*batch_size:(i+1)*batch_size].cuda()-total1[i*batch_size:(i+1)*batch_size].cuda())
        diff.square_()
        L2_dis.append(diff.sum(2).cpu())
    L2_distance = torch.concatenate(L2_dis,dim=0)


    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    source = torch.as_tensor(source, dtype=torch.float32).cuda()
    target = torch.as_tensor(target, dtype=torch.float32).cuda()
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def compute_lisi(x, y, batch_key="batch", type_="knn", n_neighbors=10):
    import scib
    import numpy as np 
    import scanpy as sc
    import anndata as ad
    import pandas as pd
    adata_scib = np.concatenate((x, y),axis=0)
    adata_scib = ad.AnnData(adata_scib, dtype=np.float32)
    adata_scib.obs['batch'] = pd.Categorical([f"true_Cell" for i in range(cell_data.shape[0])]+[f"gen_Cell" for i in range(cell_gen.shape[0])])
    sc.pp.neighbors(adata_scib, n_neighbors=10, n_pcs=20)
    scib.me.ilisi_graph(adata_scib, batch_key="batch", type_="knn")
    lisi_scores = adata_scib.obs['lisi_batch'].values
    return lisi_scores



def return_metrics(x, y):
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr
    return {"r2_score": r2_score(x.mean(axis=0), y.mean(axis=0)), 
            "pearsonr": pearsonr(x.mean(axis=0), y.mean(axis=0))[0], 
            "mmd": mmd_loss(x, y)} 
            #"mmd_rbf": mmd_rbf(x,y)}