from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import time

import matplotlib

matplotlib.use('Agg')  # 禁用图像界面弹窗，直接后台绘图保存
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from config import Config
from models import MAFN
from utils import *
import scanpy as sc


# ----------------- 新增聚类指标计算函数 -----------------
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def evaluate_all_metrics(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    # 获取标签对齐映射以计算 F1, Recall, Precision 和 Kappa
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    map_dict = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    y_pred_aligned = np.array([map_dict.get(label, label) for label in y_pred])

    acc = cluster_acc(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)
    kappa = metrics.cohen_kappa_score(y_true, y_pred_aligned)
    f1 = metrics.f1_score(y_true, y_pred_aligned, average='macro')
    recall = metrics.recall_score(y_true, y_pred_aligned, average='macro')
    precision = metrics.precision_score(y_true, y_pred_aligned, average='macro', zero_division=0)

    return acc, kappa, nmi, ari, purity, f1, recall, precision


# --------------------------------------------------------

def load_data(dataset):
    print("load data:")
    path = "./generate_data/" + dataset + "/MAFN.h5ad"
    if not os.path.exists(path):
        path = "./generate_data/Human_Breast_Cancer/MAFN.h5ad"

    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground']

    fadj_raw = adata.obsm['fadj']
    sadj_raw = adata.obsm['sadj']

    nfadj = normalize_sparse_matrix(fadj_raw + sp.eye(fadj_raw.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    nsadj = normalize_sparse_matrix(sadj_raw + sp.eye(sadj_raw.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])

    fadj = sparse_mx_to_torch_sparse_tensor(fadj_raw) if sp.issparse(fadj_raw) else torch.FloatTensor(fadj_raw)
    sadj = sparse_mx_to_torch_sparse_tensor(sadj_raw) if sp.issparse(sadj_raw) else torch.FloatTensor(sadj_raw)

    print("done")
    return adata, features, labels, fadj, sadj, nfadj, nsadj, graph_nei, graph_neg


def train(epoch, model, optimizer, features, nsadj, nfadj, graph_nei, fadj, config):
    model.train()
    optimizer.zero_grad()

    emb_fuse, pi, disp, mean, emb1, emb2 = model(features, nsadj, nfadj)

    I_N = torch.eye(features.size(0), device=features.device)
    A_target_spatial = torch.clamp(graph_nei.float() + I_N, 0.0, 1.0)
    if fadj.is_sparse:
        A_target_feature = torch.clamp(fadj.to_dense().float() + I_N, 0.0, 1.0)
    else:
        A_target_feature = torch.clamp(fadj.float() + I_N, 0.0, 1.0)

    pos_weight_s = (A_target_spatial.numel() - A_target_spatial.sum()) / A_target_spatial.sum()
    pos_weight_f = (A_target_feature.numel() - A_target_feature.sum()) / A_target_feature.sum()

    gae_loss_spatial = F.binary_cross_entropy_with_logits(torch.matmul(emb1, emb1.T), A_target_spatial,
                                                          pos_weight=pos_weight_s)
    gae_loss_feature = F.binary_cross_entropy_with_logits(torch.matmul(emb2, emb2.T), A_target_feature,
                                                          pos_weight=pos_weight_f)

    L_Base = gae_loss_spatial + gae_loss_feature
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)

    L_HSMA, K1, K2 = hsma_loss(emb1, emb2, sigma=0.5)
    W_star = compute_heat_diffusion_target(K1.detach(), K2.detach(), tau=0.5, k_neighbors=18)
    L_MKCL = manifold_kernel_contrastive_loss(emb1, emb2, W_star, tau_c=1.0)

    warmup_epochs = 50
    if epoch < warmup_epochs:
        current_alpha = 0.0
        current_beta = 0.0
    else:
        current_alpha = config.alpha
        current_beta = config.beta

    total_loss = L_Base + zinb_loss + current_alpha * L_HSMA + current_beta * L_MKCL

    emb_np = pd.DataFrame(emb_fuse.cpu().detach().numpy()).fillna(0).values
    mean_np = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values

    total_loss.backward()
    optimizer.step()

    return emb_np, mean_np, L_Base, zinb_loss, L_HSMA, L_MKCL, total_loss


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    datasets = ['Human_Breast_Cancer']

    for i in range(len(datasets)):
        dataset = datasets[i]
        config_file = './config/' + dataset + '.ini'
        print(f"\n========== Processing {dataset} ==========")
        start_time = time.time()

        adata, features, labels, fadj, sadj, nfadj, nsadj, graph_nei, graph_neg = load_data(dataset)

        plt.rcParams["figure.figsize"] = (3, 3)
        savepath = './result/' + dataset + '/'

        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        title = "Manual annotation (" + dataset + ")"
        sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title, show=False)
        plt.savefig(savepath + 'Manual_Annotation.jpg', bbox_inches='tight', dpi=600)
        plt.close()

        # ----------------- 载入调参后保存的最优参数 -----------------
        config = Config(config_file)
        config.alpha = 5.0
        config.beta = 5.0
        config.lr = 0.0005
        config.weight_decay = 0.0001
        config.seed = 2023
        # --------------------------------------------------------

        cuda = not config.no_cuda and torch.cuda.is_available()

        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())
        config.epochs = 500  # 根据 HBC 调参基准，保持原有完整周期的迭代

        if cuda:
            features = features.cuda()
            fadj = fadj.cuda()
            sadj = sadj.cuda()
            nfadj = nfadj.cuda()
            nsadj = nsadj.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()

        # 固定随机初始化种子
        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)

        if cuda:
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        model = MAFN(nfeat=config.fdim, nhid1=config.nhid1, nhid2=config.nhid2, dropout=config.dropout)

        if cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        ari_max = 0
        best_metrics = {}
        idx_max = []
        mean_max = []
        emb_max = []

        for epoch in range(config.epochs):
            emb, mean, L_Base, zinb_loss, L_HSMA, L_MKCL, total_loss = train(
                epoch, model, optimizer, features, nsadj, nfadj, graph_nei, fadj, config)

            kmeans = KMeans(n_clusters=config.class_num, random_state=config.seed).fit(emb)
            idx = kmeans.labels_

            acc, kappa, nmi, ari, purity, f1, recall, precision = evaluate_all_metrics(ground.numpy(), idx)

            if ari > ari_max:
                ari_max = ari
                idx_max = idx
                mean_max = mean
                emb_max = emb
                best_metrics = {
                    'ACC': acc, 'Kappa': kappa, 'NMI': nmi, 'ARI': ari,
                    'Purity': purity, 'F1': f1, 'Recall': recall, 'Precision': precision
                }

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"\nFinal Best Metrics for {dataset}:")
        print(f"ACC:       {best_metrics['ACC']:.8f}")
        print(f"Kappa:     {best_metrics['Kappa']:.8f}")
        print(f"NMI:       {best_metrics['NMI']:.8f}")
        print(f"ARI:       {best_metrics['ARI']:.8f}")
        print(f"Purity:    {best_metrics['Purity']:.8f}")
        print(f"F1:        {best_metrics['F1']:.8f}")
        print(f"Recall:    {best_metrics['Recall']:.8f}")
        print(f"Precision: {best_metrics['Precision']:.8f}")
        print(f"Time:      {elapsed_time:.2f} s\n")

        title = 'MAFN: ARI={:.2f}'.format(ari_max)
        adata.obs['idx'] = idx_max.astype(str)
        adata.obsm['emb'] = emb_max
        adata.obsm['mean'] = mean_max

        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
        plt.savefig(savepath + 'MAFN.jpg', bbox_inches='tight', dpi=600)
        plt.close()

        sc.pp.neighbors(adata, use_rep='mean')
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.tl.paga(adata, groups='idx')

        # 1. 保持原有输出：带有序号的版本
        sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=title, legend_fontoutline=2,
                           show=False)
        plt.savefig(savepath + 'MAFN_umap_mean.jpg', bbox_inches='tight', dpi=600)
        plt.close()

        # 2. 新增输出：纯 UMAP 图（完全不带图例和序号）
        sc.pl.umap(adata, color='idx', legend_loc=None, frameon=False, size=20, title=title, show=False)
        plt.savefig(savepath + 'MAFN_umap_only_no_label.jpg', bbox_inches='tight', dpi=600)
        plt.close()

        # 3. 新增输出：与原图结构一致（左UMAP右PAGA），但去除UMAP散点上的序号
        sc.pl.paga_compare(adata, legend_loc=None, frameon=False, size=20, title=title, show=False)
        plt.savefig(savepath + 'MAFN_umap_paga_no_label.jpg', bbox_inches='tight', dpi=600)
        plt.close()

        pd.DataFrame(emb_max).to_csv(savepath + 'MAFN_emb.csv')
        pd.DataFrame(idx_max).to_csv(savepath + 'MAFN_idx.csv')
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        adata.write(savepath + 'MAFN.h5ad')